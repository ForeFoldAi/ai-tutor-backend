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

# Uniform white margins left/right (and light bottom) after PDF clip render.
_MARGIN_LUM_THRESH = 248.0
_MARGIN_MIN_AXIS_FRAC = 0.006
_MARGIN_PAD_PX = 3
_MARGIN_MAX_SIDE_FRAC = 0.38
_MARGIN_MAX_TOP_FRAC = 0.12
_MARGIN_MAX_BOTTOM_FRAC = 0.18


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


def _is_blue_diagram_rgb(rgb_im) -> bool:
    """True for atmosphere-style teaching diagrams (skip aggressive text-edge shaving)."""
    w, h = rgb_im.size
    if w < 48 or h < 48:
        return False
    arr = np.asarray(rgb_im, dtype=np.float32)
    if arr.ndim != 3:
        return False
    blue = (arr[..., 2] > arr[..., 0] + 14.0) & (arr[..., 2] > 88.0)
    return float(blue.mean()) >= 0.20


def trim_text_side_margins(rgb_im) -> "object":
    """
    Shave vertical strips of body-text columns on the left/right edges.

    Conservative: only on reasonably wide crops, max ~12% per side, and never
    below 72% of the original width (avoids collapsing maps to a thin strip).
    """
    w, h = rgb_im.size
    if w < 120 or h < 48 or w < h * 0.55:
        return rgb_im

    arr = np.asarray(rgb_im, dtype=np.float32)
    if arr.ndim != 3:
        return rgb_im
    lum = (arr[..., 0] + arr[..., 1] + arr[..., 2]) / 3.0
    dark = lum < 185.0
    strip_w = max(8, w // 64)
    max_trim = int(w * 0.18)

    def _col_is_text(x0: int, x1: int) -> bool:
        strip = dark[:, x0:x1]
        if strip.size == 0:
            return False
        return float(strip.mean()) > 0.028 and float(lum[:, x0:x1].mean()) < 242.0

    x0 = 0
    trimmed = 0
    while trimmed + strip_w <= max_trim:
        if _col_is_text(x0, x0 + strip_w):
            x0 += strip_w
            trimmed += strip_w
        else:
            break

    x1 = w
    trimmed = 0
    while trimmed + strip_w <= max_trim:
        if _col_is_text(x1 - strip_w, x1):
            x1 -= strip_w
            trimmed += strip_w
        else:
            break

    if x1 - x0 < int(w * 0.72) or x1 - x0 < 64:
        return rgb_im
    if x0 == 0 and x1 == w:
        return rgb_im
    return rgb_im.crop((x0, 0, x1, h))


def trim_top_heading_band(rgb_im) -> "object":
    """Remove page headings and body-text lines above the diagram graphic."""
    w, h = rgb_im.size
    if h < 80:
        return rgb_im

    arr = np.asarray(rgb_im, dtype=np.float32)
    lum = (arr[..., 0] + arr[..., 1] + arr[..., 2]) / 3.0
    blue = (arr[..., 2] > arr[..., 0] + 12.0) & (arr[..., 2] > 90.0)
    dark = lum < 190.0
    y0 = 0
    scan = min(int(h * 0.18), 110)
    for y in range(scan):
        row_blue = float(blue[y, :].mean())
        row_dark = float(dark[y, :].mean())
        row_lum = float(lum[y, :].mean())
        if row_blue > 0.10:
            break
        if row_dark > 0.008 and row_lum > 225.0:
            y0 = y + 1
            continue
        if row_dark > 0.004 and row_lum > 210.0 and row_blue < 0.04:
            y0 = y + 1
            continue
        break

    if y0 < 6 or y0 > scan:
        return rgb_im
    if h - y0 < 48:
        return rgb_im
    return rgb_im.crop((0, y0, w, h))


def trim_display_margins(rgb_im, *, max_side_frac: float | None = None) -> "object":
    """
    Remove near-white uniform margins on all sides of a rendered figure.

    Applied after PDF region render so NCERT page gutters and side body-text
    columns do not appear in stored textbook images.
    """
    from PIL import Image

    w, h = rgb_im.size
    if w < 48 or h < 48:
        return rgb_im

    arr = np.asarray(rgb_im, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return rgb_im

    lum = (arr[..., 0] + arr[..., 1] + arr[..., 2]) / 3.0
    content = lum < _MARGIN_LUM_THRESH
    row_hit = content.mean(axis=1) >= _MARGIN_MIN_AXIS_FRAC
    col_hit = content.mean(axis=0) >= _MARGIN_MIN_AXIS_FRAC
    rows = np.flatnonzero(row_hit)
    cols = np.flatnonzero(col_hit)
    if rows.size < 2 or cols.size < 2:
        return rgb_im

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    side_cap = int(w * (max_side_frac if max_side_frac is not None else _MARGIN_MAX_SIDE_FRAC))
    top_cap = int(h * _MARGIN_MAX_TOP_FRAC)
    bot_cap = int(h * _MARGIN_MAX_BOTTOM_FRAC)
    if x0 > side_cap:
        x0 = 0
    if w - x1 > side_cap:
        x1 = w
    if y0 > top_cap:
        y0 = 0
    if h - y1 > bot_cap:
        y1 = h

    pad = _MARGIN_PAD_PX
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return rgb_im
    if x0 == 0 and y0 == 0 and x1 == w and y1 == h:
        return rgb_im
    return rgb_im.crop((x0, y0, x1, y1))


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
            rgb = trim_display_margins(rgb)
            if not _is_blue_diagram_rgb(rgb):
                rgb = trim_top_heading_band(rgb)
                rgb = trim_text_side_margins(rgb)
            buf = BytesIO()
            rgb.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
            return buf.getvalue()
    except Exception as exc:
        logger.debug("encode_browser_jpeg failed for %s: %s", path, exc)
        return None


def normalize_image_blob(blob: bytes, *, preserve_figure_crop: bool = False) -> bytes | None:
    """Normalize raw PDF/DOCX image bytes to browser-safe JPEG."""
    try:
        from PIL import Image

        with Image.open(BytesIO(blob)) as im:
            rgb = _pil_to_display_rgb(im)
            if rgb is None or _is_blank_rgb(rgb) or _rgb_looks_like_disclaimer_plate(rgb):
                return None
            if not preserve_figure_crop:
                rgb = trim_display_margins(rgb)
            if not preserve_figure_crop:
                diagram = _is_blue_diagram_rgb(rgb)
                if not diagram:
                    rgb = trim_top_heading_band(rgb)
                    rgb = trim_text_side_margins(rgb)
            w, h = rgb.size
            if w < 64 or h < 64 or w * h < 4096:
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
