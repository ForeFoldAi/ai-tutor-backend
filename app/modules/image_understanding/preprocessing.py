"""Gentle preprocessing for educational images (readability over compression)."""

from __future__ import annotations

import logging
from io import BytesIO

from app.config import IMAGE_MAX_HEIGHT, IMAGE_MAX_WIDTH

logger = logging.getLogger(__name__)


def preprocess_image(data: bytes, mime_type: str) -> tuple[bytes, str, int, int]:
    """
    Orient, RGB-normalize, soft-resize if oversized, strip EXIF.
    Returns (bytes, mime_type, width, height). Output is always JPEG or PNG/WEBP preserved lightly.
    """
    from PIL import Image, ImageOps

    with Image.open(BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "L"):
            if im.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", im.size, (255, 255, 255))
                rgba = im.convert("RGBA")
                background.paste(rgba, mask=rgba.split()[-1])
                im = background
            else:
                im = im.convert("RGB")
        elif im.mode == "L":
            im = im.convert("RGB")

        w, h = im.size
        max_w, max_h = IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT
        if w > max_w or h > max_h:
            im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            w, h = im.size

        # Strip metadata by re-encoding without exif
        out = BytesIO()
        out_mime = mime_type if mime_type in ("image/png", "image/webp", "image/jpeg") else "image/jpeg"
        if out_mime == "image/png":
            im.save(out, format="PNG", optimize=True)
        elif out_mime == "image/webp":
            im.save(out, format="WEBP", quality=90, method=4)
        else:
            out_mime = "image/jpeg"
            im.save(out, format="JPEG", quality=92, optimize=True)
        return out.getvalue(), out_mime, w, h
