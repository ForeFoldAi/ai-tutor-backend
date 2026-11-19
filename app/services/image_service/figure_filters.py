"""
Shared figure-crop filters for layout-based PDF extraction (all subjects).
"""

from __future__ import annotations

import re

import fitz

from app.services.image_service.math_extraction import text_from_pdf_region

_ACTIVITY_HEADER_RE = re.compile(
    r"(?i)\b(let'?s explore|activity|do you know|think about|in groups)\b"
)


def boxes_overlap_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    min_area = min(
        max(1, (a[2] - a[0]) * (a[3] - a[1])),
        max(1, (b[2] - b[0]) * (b[3] - b[1])),
    )
    return inter / min_area


def is_near_duplicate_box(
    candidate: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    *,
    threshold: float = 0.55,
) -> bool:
    return any(boxes_overlap_ratio(candidate, other) >= threshold for other in others)


def is_prose_figure_region(
    page: fitz.Page,
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    dpi: int,
    *,
    caption: str = "",
) -> bool:
    """
    Reject layout boxes that are mostly body text (false-positive figure detections).
    """
    cap = (caption or "").strip()
    if cap and _ACTIVITY_HEADER_RE.search(cap):
        return True

    raw = text_from_pdf_region(page, box, image_size, dpi)
    if not raw:
        return False

    words = re.findall(r"\b\w+\b", raw)
    word_count = len(words)
    if word_count < 28:
        return False

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    long_lines = sum(1 for ln in lines if len(ln.split()) >= 8)
    if long_lines >= 3 and word_count >= 35:
        return True

    xmin, ymin, xmax, ymax = box
    img_w, img_h = image_size
    box_w = max(1, xmax - xmin)
    box_h = max(1, ymax - ymin)
    width_frac = box_w / max(1, img_w)
    # Full-width shallow crops with dense prose (e.g. forest-fire paragraph).
    if width_frac >= 0.82 and box_h < img_h * 0.55 and word_count >= 30:
        return True

    if word_count >= 45 and not re.search(r"(?i)\bfig\.?\s*\d", raw):
        return True

    return False


def figure_asset_priority(source: str, bbox: tuple[int, int, int, int] | None) -> tuple[int, int]:
    """Higher is better for deduplicating assets that share a figure number."""
    rank = {
        "layout": 4,
        "layout_orphan": 3,
        "layout_unnumbered": 3,
        "layout_fallback": 1,
    }.get(source, 2)
    if bbox:
        area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
    else:
        area = 0
    return rank, area
