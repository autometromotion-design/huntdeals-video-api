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


def create_price_overlay(
    price: str,
    original_price: str | None,
    discount: str | None,
    duration: float,
) -> ImageClip:
    """Render a price card (white rounded rect) as a MoviePy ImageClip."""
    PAD_H = 42
    PAD_V = 30
    GAP = 24
    BADGE_PAD_H = 36
    BADGE_PAD_V = 15
    RADIUS = 48

    font_price = _load_font(156)
    font_orig  = _load_font(84)
    font_badge = _load_font(78)

    # ── Measure text extents ────────────────────────────────────────────────
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    def measure(text: str, font) -> tuple[int, int]:
        bb = probe.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0], bb[3] - bb[1]

    price_w, price_h = measure(price, font_price)

    orig_w = orig_h = 0
    if original_price:
        orig_w, orig_h = measure(original_price, font_orig)

    badge_w = badge_h = 0
    badge_text = ""
    if discount:
        badge_text = f"{discount} OFF"
        btw, bth = measure(badge_text, font_badge)
        badge_w = btw + BADGE_PAD_H * 2
        badge_h = bth + BADGE_PAD_V * 2

    # ── Block dimensions ────────────────────────────────────────────────────
    content_w = max(price_w, orig_w, badge_w)
    content_h = price_h
    if original_price:
        content_h += orig_h + GAP
    if discount:
        content_h += badge_h + GAP

    block_w = content_w + PAD_H * 2
    block_h = content_h + PAD_V * 2

    # ── Draw card ───────────────────────────────────────────────────────────
    img  = Image.new("RGBA", (block_w, block_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        [0, 0, block_w - 1, block_h - 1],
        radius=RADIUS,
        fill=(255, 255, 255, 240),
    )

    y = PAD_V

    if original_price:
        x = PAD_H + (content_w - orig_w) // 2
        draw.text((x, y), original_price, font=font_orig, fill=(153, 153, 153, 255))
        # Strikethrough line
        strike_y = y + orig_h // 2
        draw.line([(x, strike_y), (x + orig_w, strike_y)], fill=(153, 153, 153, 255), width=2)
        y += orig_h + GAP

    x = PAD_H + (content_w - price_w) // 2
    draw.text((x, y), price, font=font_price, fill=(26, 26, 26, 255))
    y += price_h + GAP

    if discount:
        badge_x = PAD_H + (content_w - badge_w) // 2
        draw.rounded_rectangle(
            [badge_x, y, badge_x + badge_w, y + badge_h],
            radius=badge_h // 2,
            fill=(229, 57, 53, 255),
        )
        btw, bth = measure(badge_text, font_badge)
        draw.text(
            (badge_x + BADGE_PAD_H, y + BADGE_PAD_V),
            badge_text,
            font=font_badge,
            fill=(255, 255, 255, 255),
        )

    # ── Wrap as MoviePy clip with alpha mask ────────────────────────────────
    arr   = np.array(img)
    rgb   = arr[:, :, :3]
    alpha = arr[:, :, 3] / 255.0

    clip = ImageClip(rgb).set_duration(duration)
    mask = ImageClip(alpha, ismask=True).set_duration(duration)
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

        # ── 5. Resize avatar to 38 % width — 20 % smaller than before ───────
        av_w        = int(TARGET_W * 0.384)
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
            overlay     = create_price_overlay(price, original_price, discount, duration)
            ov_w, ov_h  = overlay.size
            margin_right  = int(TARGET_W * 0.03)
            margin_bottom_price = int(TARGET_H * 0.04)
            ov_x = TARGET_W - ov_w - margin_right
            ov_y = TARGET_H - ov_h - margin_bottom_price
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
