"""Cheap Pillow-only image quality checks used to route between Tesseract and vision."""

from __future__ import annotations

from io import BytesIO

from app.modules.image_understanding.schemas import ImageQualityResult

# ponytail: thresholds calibrated on synthetic Pillow fixtures (tests/test_image_ocr_routing.py);
# recalibrate on real student photos. Only "too small" / "blank" stop analysis — the rest just
# steer the image to the vision model, so a false positive costs one vision call, not an answer.
MIN_SIDE_PX = 32
MIN_AREA_PX = 64 * 64
ANALYSIS_MAX_SIDE = 1024
BLANK_STDDEV = 3.0
BLANK_RANGE = 16
# Exposure: judged on the darkest / brightest 0.2% so sparse text on white is not "too bright".
DARK_MEAN, DARK_MAX_HIGHLIGHT = 60.0, 100
BRIGHT_MEAN, BRIGHT_MIN_SHADOW = 235.0, 180
# Blur: detail removed by a 1px blur relative to a 4px blur. Content-independent; sharp ≈ 0.3–0.5.
BLUR_RATIO = 0.2


def _percentile(hist: list[int], p: float) -> int:
    target, seen = sum(hist) * p, 0
    for value, count in enumerate(hist):
        seen += count
        if seen >= target:
            return value
    return 255


def assess_quality(data: bytes) -> ImageQualityResult:
    from PIL import Image, ImageChops, ImageFilter, ImageStat

    with Image.open(BytesIO(data)) as im:
        w, h = im.size
        gray = im.convert("L")
    too_small = min(w, h) < MIN_SIDE_PX or w * h < MIN_AREA_PX
    gray.thumbnail((ANALYSIS_MAX_SIDE, ANALYSIS_MAX_SIDE))

    stat = ImageStat.Stat(gray)
    mean, stddev = stat.mean[0], stat.stddev[0]
    hist = gray.histogram()
    shadow, highlight = _percentile(hist, 0.002), _percentile(hist, 0.998)
    lo, hi = gray.getextrema()
    mostly_blank = stddev < BLANK_STDDEV or (hi - lo) < BLANK_RANGE
    too_dark = not mostly_blank and mean < DARK_MEAN and highlight < DARK_MAX_HIGHLIGHT
    too_bright = not mostly_blank and mean > BRIGHT_MEAN and shadow > BRIGHT_MIN_SHADOW

    blurry = False
    if not (mostly_blank or too_small):
        fine = ImageStat.Stat(ImageChops.difference(gray, gray.filter(ImageFilter.GaussianBlur(1)))).mean[0]
        coarse = ImageStat.Stat(ImageChops.difference(gray, gray.filter(ImageFilter.GaussianBlur(4)))).mean[0]
        blurry = coarse > 0 and fine / coarse < BLUR_RATIO

    flags = {
        "too_small": too_small,
        "mostly_blank": mostly_blank,
        "too_dark": too_dark,
        "too_bright": too_bright,
        "blurry": blurry,
    }
    penalty = {"too_small": 1.0, "mostly_blank": 1.0, "too_dark": 0.4, "too_bright": 0.4, "blurry": 0.4}
    score = max(0.0, 1.0 - sum(penalty[k] for k, v in flags.items() if v))
    return ImageQualityResult(
        score=round(score, 2),
        is_usable=not (too_small or mostly_blank),
        reason=", ".join(k for k, v in flags.items() if v),
        **flags,
    )
