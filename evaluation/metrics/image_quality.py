"""Image quality heuristics for extracted figures."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any


@dataclass
class ImageQuality:
    ok: bool
    score: float
    width: int
    height: int
    issues: list[str]
    stats: dict[str, Any]


def analyze_image_bytes(
    data: bytes,
    *,
    min_side: int = 40,
    min_area: int = 2500,
    blank_std_max: float = 8.0,
    page_width: int | None = None,
    page_height: int | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    crop_edge_margin_ratio: float = 0.02,
) -> ImageQuality:
    issues: list[str] = []
    stats: dict[str, Any] = {"bytes": len(data)}

    if not data or len(data) < 32:
        return ImageQuality(False, 0.0, 0, 0, ["empty_or_tiny_bytes"], stats)

    try:
        from PIL import Image, ImageStat
        import numpy as np

        img = Image.open(io.BytesIO(data))
        img.load()
        w, h = img.size
        stats.update({"width": w, "height": h, "mode": img.mode, "format": img.format})
    except Exception as exc:  # noqa: BLE001
        return ImageQuality(False, 0.0, 0, 0, [f"corrupt:{exc}"], stats)

    if w < min_side or h < min_side:
        issues.append("low_resolution_side")
    if w * h < min_area:
        issues.append("low_resolution_area")

    # Blank / near-blank
    try:
        gray = img.convert("L")
        st = ImageStat.Stat(gray)
        std = float(st.stddev[0]) if st.stddev else 0.0
        mean = float(st.mean[0]) if st.mean else 0.0
        stats["pixel_std"] = std
        stats["pixel_mean"] = mean
        if std <= blank_std_max:
            issues.append("blank_or_uniform")
    except Exception:
        pass

    # Half / edge-cropped heuristics using bbox vs page
    if bbox and page_width and page_height and page_width > 0 and page_height > 0:
        x0, y0, x1, y1 = bbox
        margin_x = page_width * crop_edge_margin_ratio
        margin_y = page_height * crop_edge_margin_ratio
        touches_left = x0 <= margin_x
        touches_right = x1 >= page_width - margin_x
        touches_top = y0 <= margin_y
        touches_bottom = y1 >= page_height - margin_y
        edge_hits = sum([touches_left, touches_right, touches_top, touches_bottom])
        stats["edge_hits"] = edge_hits
        aspect = (x1 - x0) / max(1, (y1 - y0))
        stats["bbox_aspect"] = aspect
        # Likely split/half image: touches one vertical edge heavily and is unusually narrow/wide strip
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        if (touches_left ^ touches_right) and bw < page_width * 0.35 and bh > page_height * 0.25:
            issues.append("possible_half_or_split_image")
        if edge_hits >= 3 and (bw * bh) < (page_width * page_height * 0.15):
            issues.append("possible_cropped_fragment")
        if x0 >= x1 or y0 >= y1:
            issues.append("invalid_bbox")
        if x0 < 0 or y0 < 0 or x1 > page_width * 1.05 or y1 > page_height * 1.05:
            issues.append("bbox_out_of_page")

    # Aspect extremes often indicate bad crops / strips
    aspect = w / max(1, h)
    stats["aspect"] = aspect
    if aspect > 8 or aspect < 0.125:
        issues.append("extreme_aspect_ratio")

    # Score: start 1.0, subtract per issue
    score = 1.0
    weights = {
        "empty_or_tiny_bytes": 1.0,
        "corrupt": 1.0,
        "blank_or_uniform": 0.6,
        "low_resolution_side": 0.35,
        "low_resolution_area": 0.35,
        "possible_half_or_split_image": 0.45,
        "possible_cropped_fragment": 0.35,
        "invalid_bbox": 0.5,
        "bbox_out_of_page": 0.25,
        "extreme_aspect_ratio": 0.25,
    }
    for iss in issues:
        key = iss.split(":")[0]
        score -= weights.get(key, 0.2)
    score = max(0.0, min(1.0, score))
    ok = score >= 0.8 and "blank_or_uniform" not in issues and not any(
        i.startswith("corrupt") for i in issues
    )
    return ImageQuality(ok=ok, score=score, width=w, height=h, issues=issues, stats=stats)


def phash_hex(data: bytes) -> str | None:
    """Perceptual hash for duplicate detection (requires imagehash if available)."""
    try:
        import imagehash
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        return str(imagehash.phash(img))
    except Exception:
        # Fallback: size + rough average hash via PIL
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(io.BytesIO(data)).convert("L").resize((8, 8))
            arr = np.asarray(img, dtype=float)
            mean = arr.mean()
            bits = "".join("1" if v > mean else "0" for v in arr.flatten())
            return hex(int(bits, 2))
        except Exception:
            return None


def hamming(a: str, b: str) -> int:
    try:
        if a.startswith("0x"):
            ia, ib = int(a, 16), int(b, 16)
            return (ia ^ ib).bit_count()
        return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    except Exception:
        return 64
