from flask import Flask, request, jsonify
import os
import tempfile
import requests
from PIL import Image
Image.ANTIALIAS = Image.LANCZOS
import boto3
from moviepy.editor import *
import numpy as np
import uuid
from botocore.config import Config

app = Flask(__name__)

TARGET_W = 720
TARGET_H = 1280
PRODUCT_H = int(TARGET_H * 0.65)  # top 65% of frame


def download_file(url: str, suffix: str) -> str:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    for chunk in resp.iter_content(chunk_size=65536):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def create_zoom_clip(image_path: str, duration: float, out_w: int, out_h: int) -> VideoClip:
    max_zoom = 1.15
    img = Image.open(image_path).convert("RGB")

    large_w = int(out_w * max_zoom) + 4
    large_h = int(out_h * max_zoom) + 4
    img_large = img.resize((large_w, large_h), Image.LANCZOS)
    arr = np.array(img_large)

    def make_frame(t: float) -> np.ndarray:
        zoom = 1.0 + 0.15 * (t / duration) if duration > 0 else 1.0
        crop_w = int(out_w * max_zoom / zoom)
        crop_h = int(out_h * max_zoom / zoom)
        x0 = (large_w - crop_w) // 2
        y0 = (large_h - crop_h) // 2
        cropped = arr[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return np.array(
            Image.fromarray(cropped).resize((out_w, out_h), Image.LANCZOS)
        )

    return VideoClip(make_frame, duration=duration)


def apply_chroma_key(
    clip,
    key_color: tuple = (241, 241, 241),
    tolerance: int = 50,
):
    key = np.array(key_color, dtype=np.float32)

    def mask_frame(gf, t: float) -> np.ndarray:
        frame = gf(t).astype(np.float32)
        diff = np.max(np.abs(frame[:, :, :3] - key), axis=2)
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

    tmp_files = []
    try:
        avatar_path = download_file(data["avatar_url"], ".mp4")
        product_path = download_file(data["product_url"], ".jpg")
        tmp_files.extend([avatar_path, product_path])

        avatar_clip = VideoFileClip(avatar_path)
        duration = avatar_clip.duration

        # White background for full frame
        white_bg = ColorClip(size=(TARGET_W, TARGET_H), color=(255, 255, 255)).set_duration(duration)

        # Product image: 100% width, 65% height, pinned to top
        product_clip = create_zoom_clip(product_path, duration, TARGET_W, PRODUCT_H)
        product_clip = product_clip.set_position(("center", 0))

        # Chroma key with tolerance=50
        avatar_clip = apply_chroma_key(avatar_clip, tolerance=50)

        # Avatar: 48% width, maintain aspect ratio
        av_w = int(TARGET_W * 0.48)
        avatar_clip = avatar_clip.resize(width=av_w)
        av_h = avatar_clip.size[1]

        # Position: 2% left margin, 0% bottom margin
        av_x = int(TARGET_W * 0.02)
        av_y = TARGET_H - av_h
        avatar_clip = avatar_clip.set_position((av_x, av_y))

        final = CompositeVideoClip(
            [white_bg, product_clip, avatar_clip], size=(TARGET_W, TARGET_H)
        ).set_duration(duration)

        if avatar_clip.audio:
            final = final.set_audio(avatar_clip.audio)

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
