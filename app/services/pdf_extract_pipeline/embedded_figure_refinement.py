"""
Refine layout figure crops using embedded PDF image positions (NCERT XObjects).

Layout YOLO boxes often include body text or clip photos. Embedded JPEG rects
from PyMuPDF are tighter; when those are missing or are full-column plates,
crop the figure band above the Fig. label (excluding prose).
"""

from __future__ import annotations

import re

import fitz

from app.services.image_service.textbook_image_extraction import reject_figure_rect
from app.services.pdf_extract_pipeline.raster import image_box_to_pdf_rect, pdf_rect_to_image_box

_FIG_LABEL_RE = re.compile(r"(?i)fig\.\s*(\d+(?:\.\d+)*)")


def _is_diagram_marker_line(text: str) -> bool:
    """Short diagram labels (panel letters, monsoon arrows) — not body sentences."""
    t = text.strip()
    if re.match(r"^[A-Z]\.\s+\w", t):
        return True
    if re.match(r"(?i)^(southwest|northeast)\s+monsoon$", t):
        return True
    return False
# Column-wide background plates that bundle text + figures (reject for direct crop).
_COLUMN_PLATE_AREA_FRAC = 0.28
_COLUMN_PLATE_WIDTH_FRAC = 0.62
_COLUMN_PLATE_HEIGHT_FRAC = 0.40


def _horizontal_overlap_pdf(a: fitz.Rect, b: fitz.Rect) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def _vertical_overlap_pdf(a: fitz.Rect, b: fitz.Rect) -> float:
    return max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def _rect_intersection_area(a: fitz.Rect, b: fitz.Rect) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _rect_area(rect: fitz.Rect) -> float:
    return max(0.0, rect.width * rect.height)


def _is_column_template_plate(rect: fitz.Rect, page: fitz.Page) -> bool:
    """Reject tall column plates that mix paragraphs and figures in one XObject."""
    pw, ph = page.rect.width, page.rect.height
    if pw <= 0 or ph <= 0:
        return True
    area_frac = _rect_area(rect) / (pw * ph)
    if (
        area_frac >= _COLUMN_PLATE_AREA_FRAC
        and rect.width >= _COLUMN_PLATE_WIDTH_FRAC * pw
        and rect.height >= _COLUMN_PLATE_HEIGHT_FRAC * ph
    ):
        return True
    rejected, _ = reject_figure_rect(rect.x0, rect.y0, rect.x1, rect.y1, pw, ph)
    return rejected


def list_embedded_figure_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Teaching-sized embedded images on a page (not full-page background plates)."""
    pw, ph = page.rect.width, page.rect.height
    rects: list[fitz.Rect] = []
    for info in page.get_images(full=True):
        xref = int(info[0])
        try:
            image_rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in image_rects:
            if rect.width < 28 or rect.height < 28:
                continue
            if rect.x0 < -4 or rect.y0 < -4:
                continue
            if _is_column_template_plate(rect, page):
                continue
            rejected, _ = reject_figure_rect(
                rect.x0, rect.y0, rect.x1, rect.y1, pw, ph
            )
            if rejected:
                continue
            rects.append(rect)
    return rects


def _merge_overlapping_rects(rects: list[fitz.Rect], *, threshold: float = 0.55) -> list[fitz.Rect]:
    """Collapse duplicate image layers, but keep small teaching photos inside larger plates."""
    if not rects:
        return []
    ordered = sorted(rects, key=lambda r: -_rect_area(r))
    merged: list[fitz.Rect] = []
    for rect in ordered:
        dominated = False
        for kept in merged:
            inter = _rect_intersection_area(rect, kept)
            min_area = min(_rect_area(rect), _rect_area(kept))
            if min_area <= 0:
                continue
            overlap = inter / min_area
            size_similarity = min_area / max(_rect_area(rect), _rect_area(kept))
            if overlap >= threshold and size_similarity >= 0.28:
                dominated = True
                break
        if not dominated:
            merged.append(rect)
    return merged


def _label_rect_for_figure(page: fitz.Page, fig_number: str) -> fitz.Rect | None:
    needle = f"Fig. {fig_number}"
    hits = page.search_for(needle)
    if not hits:
        hits = page.search_for(f"Fig.{fig_number}")
    if not hits:
        return None
    return min(hits, key=lambda r: (r.y0, r.x0))


def _extend_for_caption(
    page: fitz.Page,
    crop: fitz.Rect,
    fig_number: str,
) -> fitz.Rect:
    """Include only the Fig. label line directly beneath the image (not body text above)."""
    label = _label_rect_for_figure(page, fig_number)
    if label is None:
        return crop

    return fitz.Rect(
        min(crop.x0, label.x0 - 6),
        crop.y0,
        max(crop.x1, label.x1 + 6),
        label.y1 + 4,
    )


def _score_embedded_rect(
    rect: fitz.Rect,
    *,
    label_rect: fitz.Rect | None,
    layout_rect: fitz.Rect | None,
) -> float:
    score = 0.0
    if label_rect is not None:
        gap = label_rect.y0 - rect.y1
        if -12 <= gap <= 110:
            score += 120.0 - abs(gap) * 0.8
        elif gap > 110:
            score -= 80.0
        overlap = _horizontal_overlap_pdf(rect, label_rect)
        if overlap > 20:
            score += overlap * 0.6
        cx_rect = (rect.x0 + rect.x1) / 2
        cx_label = (label_rect.x0 + label_rect.x1) / 2
        score -= abs(cx_rect - cx_label) * 0.15

    if layout_rect is not None:
        inter = _rect_intersection_area(rect, layout_rect)
        min_area = min(max(1.0, _rect_area(rect)), max(1.0, _rect_area(layout_rect)))
        score += (inter / min_area) * 80.0

    score += min(_rect_area(rect), 120_000) / 2000.0
    return score


def _same_row(a: fitz.Rect, b: fitz.Rect, *, overlap_frac: float = 0.45) -> bool:
    v = _vertical_overlap_pdf(a, b)
    return v >= overlap_frac * min(a.height, b.height)


def _horizontally_adjacent(a: fitz.Rect, b: fitz.Rect, *, max_gap: float = 36.0) -> bool:
    if not _same_row(a, b):
        return False
    gap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    return gap <= max_gap


def _union_panel_cluster(rects: list[fitz.Rect]) -> fitz.Rect:
    if not rects:
        return fitz.Rect(0, 0, 0, 0)
    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)
    return fitz.Rect(x0, y0, x1, y1)


def _cluster_panels(rects: list[fitz.Rect]) -> list[list[fitz.Rect]]:
    """Group side-by-side embedded photos on the same row."""
    if not rects:
        return []
    remaining = sorted(rects, key=lambda r: (r.y0, r.x0))
    clusters: list[list[fitz.Rect]] = []
    for rect in remaining:
        placed = False
        for cluster in clusters:
            if any(_horizontally_adjacent(rect, other) for other in cluster):
                cluster.append(rect)
                placed = True
                break
        if not placed:
            clusters.append([rect])
    return clusters


def _panels_above_label(
    embedded: list[fitz.Rect],
    label_rect: fitz.Rect,
    *,
    max_gap: float = 130.0,
) -> list[fitz.Rect]:
    """Embedded teaching photos sitting directly above a Fig. label."""
    hits: list[fitz.Rect] = []
    for rect in embedded:
        gap = label_rect.y0 - rect.y1
        if -10 <= gap <= max_gap:
            hits.append(rect)
    return hits


def _figure_vertical_bounds(
    page: fitz.Page,
    *,
    x0: float,
    x1: float,
    y_top: float,
    y_label: float,
) -> tuple[float, float]:
    """
    Top/bottom PDF-y bounds for figure content above a Fig. label.

    Stops prose scan at diagram markers (e.g. 'A. Summer') so map labels are
    not treated as body paragraphs.
    """
    band = fitz.Rect(x0, y_top, x1, y_label)
    marker_tops: list[float] = []
    last_prose_bottom = y_top

    for block in page.get_text("dict", clip=band).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            line_text = "".join(span["text"] for span in line["spans"]).strip()
            if not line_text:
                continue
            if _FIG_LABEL_RE.search(line_text):
                continue
            words = line_text.split()
            line_rect = fitz.Rect(line["bbox"])
            if _is_diagram_marker_line(line_text):
                marker_tops.append(line_rect.y0)
                continue
            if len(words) >= 5:
                last_prose_bottom = max(last_prose_bottom, line_rect.y1)

    if marker_tops:
        y0 = min(marker_tops) - 8.0
    else:
        y0 = last_prose_bottom + 6.0
    y1 = y_label + 4.0
    return y0, y1


def _crop_band_above_label(
    page: fitz.Page,
    *,
    layout_rect: fitz.Rect,
    label_rect: fitz.Rect,
) -> fitz.Rect:
    """
    When only a column plate exists, crop the figure region between prose and Fig. label.
    Uses PDF text layer to strip paragraphs (fixes Fig 3.9-style text bleed).
    """
    y0, y1 = _figure_vertical_bounds(
        page,
        x0=layout_rect.x0,
        x1=layout_rect.x1,
        y_top=layout_rect.y0,
        y_label=label_rect.y0,
    )
    y0 = max(layout_rect.y0, y0)
    if y1 - y0 < 40:
        y0 = max(layout_rect.y0, label_rect.y0 - 220)
    sidebar_top = _margin_sidebar_y0(page, y0, y1)
    if sidebar_top is not None:
        y1 = min(y1, sidebar_top - 3)
    right = _figure_diagram_right_bound(
        page, y0=y0, y1=y1, x_min=layout_rect.x0
    )
    for rule_x in _vertical_rules_in_band(page, y0, y1):
        if layout_rect.x0 < rule_x < layout_rect.x1 and right <= rule_x + 4:
            right = max(right, rule_x + 56)
    right = _cap_before_margin_sidebar(page, y0, y1, right + 4)
    right = max(right, layout_rect.x1)
    return fitz.Rect(layout_rect.x0, y0, right, y1)


def _pick_embedded_cluster(
    embedded: list[fitz.Rect],
    *,
    label_rect: fitz.Rect | None,
    layout_rect: fitz.Rect | None,
) -> tuple[fitz.Rect, int] | None:
    """Return (union_rect, panel_count) for the best embedded photo cluster."""
    if not embedded:
        return None

    if label_rect is not None:
        above = _panels_above_label(embedded, label_rect)
        if above:
            clusters = _cluster_panels(above)
            if clusters:
                best_cluster = max(
                    clusters,
                    key=lambda c: sum(_rect_area(r) for r in c),
                )
                return _union_panel_cluster(best_cluster), len(best_cluster)

    best_rect: fitz.Rect | None = None
    best_score = 0.0
    for rect in embedded:
        score = _score_embedded_rect(rect, label_rect=label_rect, layout_rect=layout_rect)
        if score > best_score:
            best_score = score
            best_rect = rect
    if best_rect is None or best_score < 45.0:
        return None
    return fitz.Rect(best_rect), 1


def _layout_has_prose_bleed(page: fitz.Page, layout_rect: fitz.Rect) -> bool:
    """True when the layout YOLO box contains paragraph text, not just a photo caption."""
    text = page.get_text("text", clip=layout_rect) or ""
    words = len(text.split())
    if words > 12:
        return True
    long_lines = sum(1 for ln in text.splitlines() if len(ln.split()) >= 8)
    return long_lines >= 2


def _trim_layout_to_label(layout_rect: fitz.Rect, label_rect: fitz.Rect) -> fitz.Rect:
    return fitz.Rect(layout_rect.x0, layout_rect.y0, layout_rect.x1, label_rect.y1 + 4)


def _split_by_vertical_gutter(embedded_rect: fitz.Rect, layout_rect: fitz.Rect) -> bool:
    """
    NCERT pages often place a vertical column rule between a diagram fragment and
    vector overlays (e.g. Fig 3.4 globe left, solar-radiation arrows right).
    """
    if layout_rect.width <= 0:
        return False
    return layout_rect.width > embedded_rect.width * 1.08


def _vertical_rules_in_band(page: fitz.Page, y0: float, y1: float) -> list[float]:
    """Tall thin column rules crossing a figure band (x center of each rule)."""
    page_h = page.rect.height
    rules: list[float] = []
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"])
        if rect.height < page_h * 0.35:
            continue
        if rect.width > 4:
            continue
        if rect.y1 < y0 + 16 or rect.y0 > y1 - 16:
            continue
        rules.append((rect.x0 + rect.x1) / 2)
    return rules


def _cap_before_margin_sidebar(
    page: fitz.Page, y0: float, y1: float, right: float
) -> float:
    """Trim right edge only when the chapter margin title overlaps the crop."""
    page_w = page.rect.width
    capped = right
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            bbox = fitz.Rect(line["bbox"])
            if bbox.y1 < y0 or bbox.y0 > y1:
                continue
            if bbox.x0 < page_w * 0.8 or bbox.width > 22:
                continue
            if bbox.x0 < capped:
                capped = min(capped, bbox.x0 - 4)
    return capped


def _margin_sidebar_y0(page: fitz.Page, y0: float, y1: float) -> float | None:
    """Top of vertical chapter-title margin text inside a figure band."""
    page_w = page.rect.width
    top: float | None = None
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            bbox = fitz.Rect(line["bbox"])
            if bbox.y1 < y0 or bbox.y0 > y1:
                continue
            if bbox.x0 < page_w * 0.8 or bbox.width > 22:
                continue
            top = bbox.y0 if top is None else min(top, bbox.y0)
    return top


def _figure_diagram_right_bound(
    page: fitz.Page,
    *,
    y0: float,
    y1: float,
    x_min: float,
) -> float:
    """Rightmost ink for diagram vectors/labels, excluding column rules."""
    max_x = x_min
    for drawing in page.get_drawings():
        rect = fitz.Rect(drawing["rect"])
        if rect.y1 < y0 or rect.y0 > y1:
            continue
        if rect.x0 < x_min - 20:
            continue
        if rect.height > rect.width * 8 and rect.width <= 25:
            continue
        color = drawing.get("color")
        width = drawing.get("width") or 0
        if color and len(color) >= 3:
            is_grey = max(color) - min(color) < 0.08
            if is_grey and width <= 1.5 and rect.height > rect.width * 8:
                continue
        if rect.width < 5 and rect.height < 5:
            continue
        if rect.width >= rect.height * 2 or rect.width >= 12:
            max_x = max(max_x, rect.x1)

    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            bbox = fitz.Rect(line["bbox"])
            if bbox.y1 < y0 or bbox.y0 > y1:
                continue
            if bbox.x0 < x_min - 20:
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text or len(text.split()) > 5:
                continue
            max_x = max(max_x, bbox.x1)
    return max_x


def _crop_across_vertical_gutter(
    page: fitz.Page,
    embedded_rect: fitz.Rect,
    layout_rect: fitz.Rect,
) -> fitz.Rect:
    """
    Span embedded + vector overlays across a column rule.

    Use diagram ink bounds on the right (not layout x1) so the margin rule is
    not parked on the crop edge (Fig 3.4).
    """
    y0 = embedded_rect.y0
    y1 = embedded_rect.y1
    sidebar_top = _margin_sidebar_y0(page, y0, y1)
    if sidebar_top is not None:
        y1 = min(y1, sidebar_top - 3)
    x0 = max(layout_rect.x0, embedded_rect.x0)
    right = _figure_diagram_right_bound(
        page, y0=y0, y1=y1, x_min=embedded_rect.x0
    )
    for rule_x in _vertical_rules_in_band(page, y0, y1):
        if embedded_rect.x0 < rule_x < layout_rect.x1 and right <= rule_x + 4:
            right = max(right, rule_x + 56)
    right = _cap_before_margin_sidebar(page, y0, y1, right + 4)
    right = max(right, embedded_rect.x1 + 8)
    return fitz.Rect(x0, y0, right, y1)


def try_extract_embedded_jpeg_bytes(
    page: fitz.Page,
    fig_number: str,
    layout_rect: fitz.Rect | None,
) -> bytes | None:
    """
  Return the full embedded JPEG stream for a teaching photo (Fig 3.3).

  Page clipping at the placed rect often cuts the photo at a vertical column line;
  the PDF XObject bytes contain the complete image.
    """
    label_rect = _label_rect_for_figure(page, fig_number)
    if label_rect is None:
        return None
    if layout_rect is not None and _layout_has_prose_bleed(page, layout_rect):
        return None

    best: tuple[float, int, fitz.Rect] | None = None
    for info in page.get_images(full=True):
        xref = int(info[0])
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            if _is_column_template_plate(rect, page):
                continue
            if rect.width < 40 or rect.height < 40:
                continue
            gap = label_rect.y0 - rect.y1
            if not (-10 <= gap <= 130):
                continue
            score = _score_embedded_rect(
                rect, label_rect=label_rect, layout_rect=layout_rect
            )
            if score < 45.0:
                continue
            if best is None or score > best[0]:
                best = (score, xref, fitz.Rect(rect))

    if best is None:
        return None
    if layout_rect is not None and _split_by_vertical_gutter(best[2], layout_rect):
        return None
    try:
        doc = page.parent
        if doc is None:
            return None
        return doc.extract_image(best[1])["image"]
    except Exception:
        return None


def refine_figure_box(
    page: fitz.Page,
    *,
    layout_box: tuple[int, int, int, int],
    label_box: tuple[int, int, int, int] | None,
    fig_number: str | None,
    image_size: tuple[int, int],
    dpi: int,
) -> tuple[int, int, int, int]:
    """
    Choose the best figure bounds without disturbing clean layout detections.

    - Clean layout photo boxes (Fig 3.3) → keep layout YOLO bounds.
    - Single tight embedded JPEGs (Fig 3.4) → embedded rect only, no layout bleed.
    - Multi-panel collages → union embedded panels.
    - Column plates with text (Fig 3.9) → text-trimmed band above the label.
    """
    embedded = _merge_overlapping_rects(list_embedded_figure_rects(page))
    page_rect = page.rect
    layout_rect = image_box_to_pdf_rect(layout_box, page_rect, image_size, dpi)
    label_rect = None
    if fig_number:
        label_rect = _label_rect_for_figure(page, fig_number)
    if label_rect is None and label_box is not None:
        label_rect = image_box_to_pdf_rect(label_box, page_rect, image_size, dpi)

    pick = _pick_embedded_cluster(
        embedded, label_rect=label_rect, layout_rect=layout_rect
    )
    embedded_rect = pick[0] if pick else None
    panel_count = pick[1] if pick else 0

    crop: fitz.Rect | None = None
    split_gutter = (
        embedded_rect is not None
        and layout_rect is not None
        and _split_by_vertical_gutter(embedded_rect, layout_rect)
    )

    if split_gutter:
        crop = _crop_across_vertical_gutter(page, embedded_rect, layout_rect)
    elif label_rect is not None and not _layout_has_prose_bleed(page, layout_rect):
        crop = _trim_layout_to_label(layout_rect, label_rect)
    elif embedded_rect is not None and panel_count >= 2:
        crop = fitz.Rect(embedded_rect)
    elif embedded_rect is not None and panel_count == 1:
        crop = fitz.Rect(embedded_rect)
    elif label_rect is not None:
        crop = _crop_band_above_label(
            page, layout_rect=layout_rect, label_rect=label_rect
        )
    else:
        crop = fitz.Rect(layout_rect)

    crop = crop & page_rect

    pad = 6 / (dpi / 72.0)
    if split_gutter:
        crop = fitz.Rect(
            max(0, crop.x0 - pad),
            max(0, crop.y0 - pad),
            min(page_rect.width, crop.x1),
            min(page_rect.height, crop.y1 + pad),
        )
    else:
        crop = fitz.Rect(
            max(0, crop.x0 - pad),
            max(0, crop.y0 - pad),
            min(page_rect.width, crop.x1 + pad),
            min(page_rect.height, crop.y1 + pad),
        )
    return pdf_rect_to_image_box(crop, dpi)
