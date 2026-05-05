import io
import os
import subprocess
import sys
import uuid
import tempfile
import requests
import numpy as np
from flask import Flask, request, jsonify
from moviepy.editor import VideoFileClip, VideoClip, ImageClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont

# Pillow 10+ removed ANTIALIAS; MoviePy still references it internally
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS
import boto3
from botocore.config import Config


def ensure_fonts():
    """Install DejaVu fonts if not present on the system."""
    test_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(test_path):
        try:
            subprocess.run(
                ["apt-get", "install", "-y", "--no-install-recommends", "fonts-dejavu-core"],
                check=True,
                capture_output=True,
            )
        except Exception:
            pass


ensure_fonts()

app = Flask(__name__)

TARGET_W = 720
TARGET_H = 1280

# Font search order for Ubuntu/Railway
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def download_file(url: str, suffix: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, stream=True, timeout=120, headers=headers)
    resp.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    for chunk in resp.iter_content(chunk_size=65536):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def create_static_bg(image_path: str, duration: float) -> ImageClip:
    """Product image fills top 50% of canvas; bottom 50% is white."""
    half_h = TARGET_H // 2
    img = Image.open(image_path).convert("RGB")
    img = img.resize((TARGET_W, half_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))
    canvas.paste(img, (0, 0))
    return ImageClip(np.array(canvas)).set_duration(duration)


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


def create_price_tag_overlay(
    price: str,
    original_price: str | None,
    discount: str | None,
    duration: float,
) -> ImageClip:
    """Load tag.jpg template from R2, remove white bg, paint dynamic prices on top."""
    tag_url = os.environ["R2_PUBLIC_URL"].rstrip("/") + "/tag.jpg"
    resp = requests.get(tag_url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0"
    })
    resp.raise_for_status()
    template = Image.open(io.BytesIO(resp.content)).convert("RGBA")

    # ── Remove white background ──────────────────────────────────────────────
    data = np.array(template, dtype=np.float32)
    white_mask = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
    data[white_mask, 3] = 0
    template = Image.fromarray(data.astype(np.uint8))

    # ── Scale to fit right quadrant (max 320 px wide) ───────────────────────
    TW    = 320
    ratio = TW / template.width
    TH    = int(template.height * ratio)
    template = template.resize((TW, TH), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(template)

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    def measure(text: str, font) -> tuple[int, int]:
        bb = probe.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0], bb[3] - bb[1]

    # Zone proportions (tuned to tag.jpg layout)
    # Black top zone:  y  8% – 38%   original price
    # Yellow zone:     y 38% – 64%   current price
    # Red zone:        y 64% – 95%   discount circle
    top_y1, top_y2 = int(TH * 0.08), int(TH * 0.38)
    mid_y1, mid_y2 = int(TH * 0.38), int(TH * 0.64)
    bot_y1, bot_y2 = int(TH * 0.64), int(TH * 0.95)

    # ── Erase baked-in numbers with section background colours ───────────────
    pad_x = int(TW * 0.08)
    if original_price:
        draw.rectangle([pad_x, top_y1, TW - pad_x, top_y2], fill=(18, 18, 18, 255))
    draw.rectangle([pad_x, mid_y1, TW - pad_x, mid_y2], fill=(255, 200, 0, 255))
    if discount:
        cx, cy = TW // 2, (bot_y1 + bot_y2) // 2
        cr = (bot_y2 - bot_y1) // 2 - int(TH * 0.03)
        draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(18, 18, 18, 255))

    # ── Draw original price (white, red diagonal strikethrough) ─────────────
    if original_price:
        font_orig = _load_font(int((top_y2 - top_y1) * 0.52))
        ow, oh    = measure(original_price, font_orig)
        ox = (TW - ow) // 2
        oy = top_y1 + ((top_y2 - top_y1) - oh) // 2
        draw.text((ox, oy), original_price, font=font_orig, fill=(255, 255, 255, 255))
        sy = oy + oh // 2
        draw.line([(ox - 8, sy - 5), (ox + ow + 8, sy + 5)], fill=(220, 30, 30, 255), width=5)

    # ── Draw current price (black on yellow) ─────────────────────────────────
    font_price = _load_font(int((mid_y2 - mid_y1) * 0.72))
    pw, ph     = measure(price, font_price)
    px = (TW - pw) // 2
    py = mid_y1 + ((mid_y2 - mid_y1) - ph) // 2
    draw.text((px + 3, py + 3), price, font=font_price, fill=(80, 60, 0, 160))
    draw.text((px, py),         price, font=font_price, fill=(18, 18, 18, 255))

    # ── Draw discount (white in black circle) ────────────────────────────────
    if discount:
        font_disc = _load_font(int(cr * 0.72))
        dw, dh    = measure(discount, font_disc)
        draw.text((cx - dw // 2, cy - dh // 2), discount, font=font_disc,
                  fill=(255, 255, 255, 255))

    # ── Wrap as MoviePy clip ─────────────────────────────────────────────────
    arr   = np.array(template)
    rgb   = arr[:, :, :3]
    alpha = arr[:, :, 3] / 255.0
    clip  = ImageClip(rgb).set_duration(duration)
    mask  = ImageClip(alpha, ismask=True).set_duration(duration)
    return clip.set_mask(mask)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/process-video", methods=["POST"])
def process_video():
    data = request.get_json(silent=True)
    if not data or "avatar_url" not in data or "product_url" not in data:
        return jsonify({"error": "avatar_url and product_url are required"}), 400

    price          = data.get("price")           # e.g. "$29.99"
    original_price = data.get("original_price")  # e.g. "$49.99"
    discount       = data.get("discount")        # e.g. "40%"

    tmp_files: list[str] = []
    try:
        # ── 1. Download inputs ──────────────────────────────────────────────
        avatar_path  = download_file(data["avatar_url"], ".mp4")
        product_path = download_file(data["product_url"], ".jpg")
        tmp_files.extend([avatar_path, product_path])

        # ── 2. Load avatar ──────────────────────────────────────────────────
        avatar_clip = VideoFileClip(avatar_path)
        duration    = avatar_clip.duration

        # ── 3. Background: static product image ────────────────────────────
        bg_clip = create_static_bg(product_path, duration)

        # ── 4. Chroma key: remove grey background ───────────────────────────
        avatar_clip = apply_chroma_key(avatar_clip)

        # ── 5. Resize avatar — 10 % larger than previous ────────────────────
        av_w        = int(TARGET_W * 0.4507)
        avatar_clip = avatar_clip.resize(width=av_w)
        av_h        = avatar_clip.h

        # ── 6. Center avatar in bottom-left quadrant ─────────────────────────
        margin_bottom = int(TARGET_H * 0.02)
        x = (TARGET_W // 2 - av_w) // 2        # centered within left half
        y = TARGET_H - av_h - margin_bottom
        avatar_clip = avatar_clip.set_position((x, y))

        # ── 7. Price overlay (bottom-right, optional) ────────────────────────
        layers = [bg_clip, avatar_clip]

        if price:
            overlay    = create_price_tag_overlay(price, original_price, discount, duration)
            ov_w, ov_h = overlay.size
            # Center in bottom-right quadrant, clamped to stay on screen
            ov_x = TARGET_W // 2 + (TARGET_W // 2 - ov_w) // 2
            ov_y = TARGET_H // 2 + (TARGET_H // 2 - ov_h) // 2
            ov_x = max(TARGET_W // 2, min(ov_x, TARGET_W - ov_w))
            ov_y = max(TARGET_H // 2, min(ov_y, TARGET_H - ov_h))
            overlay = overlay.set_position((ov_x, ov_y))
            layers.append(overlay)

        # ── 8. Composite ────────────────────────────────────────────────────
        final = CompositeVideoClip(layers, size=(TARGET_W, TARGET_H)).set_duration(duration)

        if avatar_clip.audio:
            final = final.set_audio(avatar_clip.audio)

        # ── 9. Export ────────────────────────────────────────────────────────
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

        # ── 10. Upload to Cloudflare R2 ──────────────────────────────────────
        filename   = f"processed_{uuid.uuid4().hex}.mp4"
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
