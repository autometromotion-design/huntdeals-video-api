import os
import uuid
import tempfile
import requests
import numpy as np
from flask import Flask, request, jsonify
from moviepy.editor import VideoFileClip, VideoClip, CompositeVideoClip
Image.ANTIALIAS = Image.LANCZOS
import boto3
from botocore.config import Config

app = Flask(__name__)

TARGET_W = 720
TARGET_H = 1280


def download_file(url: str, suffix: str) -> str:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    for chunk in resp.iter_content(chunk_size=65536):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def create_zoom_clip(image_path: str, duration: float) -> VideoClip:
    """Background image with a smooth Ken Burns zoom from 100% to 115%."""
    max_zoom = 1.15
    img = Image.open(image_path).convert("RGB")

    large_w = int(TARGET_W * max_zoom) + 4
    large_h = int(TARGET_H * max_zoom) + 4
    img_large = img.resize((large_w, large_h), Image.Resampling.LANCZOS)
    arr = np.array(img_large)

    def make_frame(t: float) -> np.ndarray:
        zoom = 1.0 + 0.15 * (t / duration) if duration > 0 else 1.0
        # Shrink the crop window as zoom increases so the image appears larger
        crop_w = int(TARGET_W * max_zoom / zoom)
        crop_h = int(TARGET_H * max_zoom / zoom)
        x0 = (large_w - crop_w) // 2
        y0 = (large_h - crop_h) // 2
        cropped = arr[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return np.array(
            Image.fromarray(cropped).resize(
                (TARGET_W, TARGET_H), Image.Resampling.LANCZOS
            )
        )

    return VideoClip(make_frame, duration=duration)


def apply_chroma_key(
    clip: VideoFileClip,
    key_color: tuple = (241, 241, 241),
    tolerance: int = 30,
) -> VideoFileClip:
    """Remove the solid background colour from the avatar clip."""
    key = np.array(key_color, dtype=np.float32)

    def mask_frame(gf, t: float) -> np.ndarray:
        frame = gf(t).astype(np.float32)
        diff = np.max(np.abs(frame[:, :, :3] - key), axis=2)
        # 1.0 = keep pixel, 0.0 = transparent
        return (diff >= tolerance).astype(np.float32)

    mask = VideoClip(
        lambda t: mask_frame(clip.get_frame, t),
        duration=clip.duration,
        ismask=True,
    )
    return clip.set_mask(mask)


def upload_to_r2(file_path: str, filename: str) -> str:
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    bucket = os.environ["R2_BUCKET_NAME"]
    s3.upload_file(
        file_path,
        bucket,
        filename,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    public_base = os.environ["R2_PUBLIC_URL"].rstrip("/")
    return f"{public_base}/{filename}"


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/process-video", methods=["POST"])
def process_video():
    data = request.get_json(silent=True)
    if not data or "avatar_url" not in data or "product_url" not in data:
        return jsonify({"error": "avatar_url and product_url are required"}), 400

    tmp_files: list[str] = []
    try:
        # ── 1. Download inputs ──────────────────────────────────────────────
        avatar_path = download_file(data["avatar_url"], ".mp4")
        product_path = download_file(data["product_url"], ".jpg")
        tmp_files.extend([avatar_path, product_path])

        # ── 2. Load avatar ──────────────────────────────────────────────────
        avatar_clip = VideoFileClip(avatar_path)
        duration = avatar_clip.duration

        # ── 3. Background: product image with zoom effect ───────────────────
        bg_clip = create_zoom_clip(product_path, duration)

        # ── 4. Chroma key: remove grey background ───────────────────────────
        avatar_clip = apply_chroma_key(avatar_clip)

        # ── 5. Resize avatar to 42 % width × 35 % height ────────────────────
        av_w = int(TARGET_W * 0.42)
        av_h = int(TARGET_H * 0.35)
        avatar_clip = avatar_clip.resize((av_w, av_h))

        # ── 6. Position: bottom-left corner ─────────────────────────────────
        avatar_clip = avatar_clip.set_position((0, TARGET_H - av_h))

        # ── 7. Composite ────────────────────────────────────────────────────
        final = CompositeVideoClip(
            [bg_clip, avatar_clip], size=(TARGET_W, TARGET_H)
        ).set_duration(duration)

        if avatar_clip.audio:
            final = final.set_audio(avatar_clip.audio)

        # ── 8. Export ────────────────────────────────────────────────────────
        output_path = tempfile.mktemp(suffix=".mp4")
        tmp_files.append(output_path)
        final.write_videofile(
            output_path,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            threads=4,
            verbose=False,
            logger=None,
        )

        # ── 9. Upload to Cloudflare R2 ───────────────────────────────────────
        filename = f"processed_{uuid.uuid4().hex}.mp4"
        public_url = upload_to_r2(output_path, filename)

        return jsonify({"url": public_url})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    finally:
        for path in tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
