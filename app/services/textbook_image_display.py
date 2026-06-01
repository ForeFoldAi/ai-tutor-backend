"""
Convert extracted textbook assets to browser-safe JPEG and detect blank frames.

NCERT PDFs often embed JPEG 2000 (JP2) and grayscale+alpha (LA) images that
browsers cannot render when saved with a .png extension.
"""

from __future__ import annotations

import logging
import statistics
from io import BytesIO

import numpy as np

logger = logging.getLogger(__name__)

_MIN_LUMINANCE_STDDEV = 4.0
_JPEG_QUALITY = 88

# Full-page "© NCERT / not to be republished" slabs: huge pale-gray image, ~95% white, no chroma.
_ANALYSIS_MAX_DIM = 256
_DISCLAIMER_MIN_AREA = 2_200_000
_DISCLAIMER_BRIGHT_FRAC = 0.88
_DISCLAIMER_MAX_SAT_MEAN = 6.0
_DISCLAIMER_LUM_STD_MIN = 10.0
_DISCLAIMER_LUM_STD_MAX = 55.0


def _pil_to_display_rgb(im) -> "object | None":
    from PIL import Image

    try:
        im.load()
    except Exception:
        return None

    mode = im.mode
    if mode == "LA":
        gray, alpha = im.split()
        background = Image.new("RGBA", im.size, (255, 255, 255, 255))
        gray_rgb = Image.merge("RGB", (gray, gray, gray))
        background.paste(gray_rgb, mask=alpha)
        return background.convert("RGB")
    if mode == "RGBA":
        background = Image.new("RGBA", im.size, (255, 255, 255, 255))
        background.alpha_composite(im)
        return background.convert("RGB")
    if mode in ("CMYK", "P", "L", "1"):
        return im.convert("RGB")
    if mode == "RGB":
        return im
    try:
        return im.convert("RGB")
    except Exception:
        return None


def _is_blank_rgb(rgb_im) -> bool:
    w, h = rgb_im.size
    if w < 1 or h < 1:
        return True
    step = max(1, (w * h) // 5000)
    sample = list(rgb_im.getdata())[::step]
    if len(sample) < 2:
        return True
    lum = [sum(px) / 3.0 for px in sample]
    return statistics.pstdev(lum) < _MIN_LUMINANCE_STDDEV


def _rgb_looks_like_disclaimer_plate(rgb_im) -> bool:
    """
    Detect NCERT-style copyright slabs: very large, almost entirely white/light gray,
    essentially grayscale (diagonal text), with modest luminance variance from the text.
    """
    from PIL import Image

    w, h = rgb_im.size
    area = w * h
    if area < _DISCLAIMER_MIN_AREA:
        return False

    m = max(w, h)
    small = rgb_im
    if m > _ANALYSIS_MAX_DIM:
        s = _ANALYSIS_MAX_DIM / m
        small = rgb_im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.BILINEAR)

    if small.mode != "RGB":
        small = small.convert("RGB")

    a = np.asarray(small, dtype=np.float32)
    lum = (a[:, :, 0] + a[:, :, 1] + a[:, :, 2]) / 3.0
    bright_frac = float((lum >= 230).mean())
    sat_mean = float((np.max(a, axis=2) - np.min(a, axis=2)).mean())
    lum_std = float(np.std(lum))

    return (
        bright_frac >= _DISCLAIMER_BRIGHT_FRAC
        and sat_mean <= _DISCLAIMER_MAX_SAT_MEAN
        and _DISCLAIMER_LUM_STD_MIN <= lum_std <= _DISCLAIMER_LUM_STD_MAX
    )


def image_has_visible_content(path: str) -> bool:
    """False for empty frames, copyright slabs, and other non-figure assets."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            rgb = _pil_to_display_rgb(im)
            if rgb is None or _is_blank_rgb(rgb):
                return False
            if _rgb_looks_like_disclaimer_plate(rgb):
                return False
            return True
    except Exception:
        return False


def encode_browser_jpeg(path: str) -> bytes | None:
    """Decode any Pillow-supported textbook asset and return JPEG bytes."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            rgb = _pil_to_display_rgb(im)
            if rgb is None or _is_blank_rgb(rgb) or _rgb_looks_like_disclaimer_plate(rgb):
                return None
            buf = BytesIO()
            rgb.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
            return buf.getvalue()
    except Exception as exc:
        logger.debug("encode_browser_jpeg failed for %s: %s", path, exc)
        return None


def normalize_image_blob(blob: bytes) -> bytes | None:
    """Normalize raw PDF/DOCX image bytes to browser-safe JPEG."""
    try:
        from PIL import Image

        with Image.open(BytesIO(blob)) as im:
            rgb = _pil_to_display_rgb(im)
            if rgb is None or _is_blank_rgb(rgb) or _rgb_looks_like_disclaimer_plate(rgb):
                return None
            w, h = rgb.size
            if w < 36 or h < 36 or w * h < 2800:
                return None
            buf = BytesIO()
            rgb.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
            return buf.getvalue()
    except Exception:
        return None


def can_serve_file_directly(path: str) -> bool:
    """True when the on-disk file is already a non-blank RGB JPEG."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.load()
            return im.format == "JPEG" and im.mode == "RGB" and not _is_blank_rgb(im)
    except Exception:
        return False
