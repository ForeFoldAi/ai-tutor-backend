"""
Refine layout figure crops using embedded PDF image positions (NCERT XObjects).

Layout YOLO boxes often clip photos or bleed into neighbouring text columns.
Embedded image rects from PyMuPDF are usually tighter and better aligned with
the printed figure.
"""

from __future__ import annotations

import re

import fitz

from app.services.image_service.textbook_image_extraction import reject_figure_rect
from app.services.pdf_extract_pipeline.raster import image_box_to_pdf_rect, pdf_rect_to_image_box

_FIG_LABEL_RE = re.compile(r"(?i)fig\.\s*(\d+(?:\.\d+)*)")


def _horizontal_overlap_pdf(a: fitz.Rect, b: fitz.Rect) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


def _rect_intersection_area(a: fitz.Rect, b: fitz.Rect) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


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
            rejected, _ = reject_figure_rect(
                rect.x0, rect.y0, rect.x1, rect.y1, pw, ph
            )
            if rejected:
                continue
            rects.append(rect)
    return rects


def _rect_area(rect: fitz.Rect) -> float:
    return max(0.0, rect.width * rect.height)


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
    """Include the Fig. label and any caption lines directly beneath the image."""
    label = _label_rect_for_figure(page, fig_number)
    if label is None:
        return crop

    band = fitz.Rect(
        max(0, min(crop.x0, label.x0) - 18),
        max(0, crop.y1 - 6),
        min(page.rect.width, max(crop.x1, label.x1) + 18),
        min(page.rect.height, label.y1 + 4),
    )
    max_y = label.y1
    for block in page.get_text("dict", clip=band).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            line_text = "".join(span["text"] for span in line["spans"]).strip()
            if not line_text:
                continue
            line_rect = fitz.Rect(line["bbox"])
            overlap = _horizontal_overlap_pdf(crop, line_rect)
            if overlap < max(18.0, crop.width * 0.22):
                continue
            if line_rect.y1 > label.y1 + 2:
                continue
            if _FIG_LABEL_RE.search(line_text) and fig_number in line_text:
                max_y = max(max_y, line_rect.y1)
                continue
            if line_rect.y0 >= crop.y1 - 6 and line_rect.y1 <= label.y0 + 6:
                if len(line_text) >= 8:
                    max_y = max(max_y, line_rect.y1)

    return fitz.Rect(
        min(crop.x0, label.x0 - 6),
        crop.y0,
        max(crop.x1, label.x1 + 6),
        max_y + 4,
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
    Prefer embedded PDF image rect + caption over a raw layout-detector crop.
    """
    embedded = _merge_overlapping_rects(list_embedded_figure_rects(page))
    if not embedded:
        return layout_box

    page_rect = page.rect
    layout_rect = image_box_to_pdf_rect(layout_box, page_rect, image_size, dpi)
    label_rect = None
    if fig_number:
        label_rect = _label_rect_for_figure(page, fig_number)
    if label_rect is None and label_box is not None:
        label_rect = image_box_to_pdf_rect(label_box, page_rect, image_size, dpi)

    best_rect: fitz.Rect | None = None
    best_score = 0.0
    for rect in embedded:
        score = _score_embedded_rect(rect, label_rect=label_rect, layout_rect=layout_rect)
        if score > best_score:
            best_score = score
            best_rect = rect

    if best_rect is None or best_score < 45.0:
        return layout_box

    crop = fitz.Rect(best_rect)
    if layout_rect is not None and _horizontal_overlap_pdf(crop, layout_rect) > 30:
        extra = layout_rect & fitz.Rect(
            crop.x0 - 12,
            crop.y0 - 12,
            min(page_rect.width, layout_rect.x1),
            max(crop.y1, layout_rect.y1) + 8,
        )
        if extra.width > 12 and extra.height > 12:
            crop = crop | extra

    if fig_number:
        crop = _extend_for_caption(page, crop, fig_number)

    crop = crop & page_rect

    pad = 6 / (dpi / 72.0)
    crop = fitz.Rect(
        max(0, crop.x0 - pad),
        max(0, crop.y0 - pad),
        min(page_rect.width, crop.x1 + pad),
        min(page_rect.height, crop.y1 + pad),
    )
    return pdf_rect_to_image_box(crop, dpi)
