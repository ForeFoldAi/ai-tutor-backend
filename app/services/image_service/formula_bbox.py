"""Formula bounding-box utilities (standalone — no pipeline imports)."""

from __future__ import annotations

FORMULA_PADDING_RATIO = 0.14
FORMULA_PADDING_MIN_PX = 10
FORMULA_PADDING_MAX_PX = 56
FORMULA_VERTICAL_EXTRA = 0.10


def expand_formula_bbox(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    ratio: float = FORMULA_PADDING_RATIO,
    min_pad: int = FORMULA_PADDING_MIN_PX,
    max_pad: int = FORMULA_PADDING_MAX_PX,
    vertical_extra: float = FORMULA_VERTICAL_EXTRA,
) -> tuple[int, int, int, int]:
    """Expand a formula detection box so multi-line expressions are not clipped."""
    xmin, ymin, xmax, ymax = box
    w, h = max(1, xmax - xmin), max(1, ymax - ymin)
    pad_x = max(min_pad, min(max_pad, int(w * ratio)))
    pad_y = max(min_pad, min(max_pad, int(h * (ratio + vertical_extra))))
    img_w, img_h = image_size
    return (
        max(0, xmin - pad_x),
        max(0, ymin - pad_y),
        min(img_w, xmax + pad_x),
        min(img_h, ymax + pad_y),
    )
