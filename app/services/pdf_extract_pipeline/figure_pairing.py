"""
Layout-first figure/table/formula asset extraction from raster pages.

Adapted for NCERT/CBSE tutoring: pairs Fig. labels with layout bboxes,
with fallbacks for unnumbered figures.
"""

from __future__ import annotations

import io
import re
from typing import Any

import fitz
from PIL import Image

from app.services.image_service.figure_filters import (
    is_near_duplicate_box,
    is_prose_figure_region,
)
from app.services.image_service.math_extraction import (
    extract_formula_text_from_pdf,
    merge_formula_structured_text,
    section_heading_above_figure,
)
from app.services.image_service.formula_bbox import expand_formula_bbox
from app.services.pdf_extract_pipeline.embedded_figure_refinement import (
    refine_figure_box,
    try_extract_embedded_jpeg_bytes,
)
from app.services.pdf_extract_pipeline.raster import image_box_to_pdf_rect, pdf_rect_to_image_box
from app.services.pdf_extract_pipeline.types import ExtractedAsset, LayoutElement, PageExtraction

FIGURE_LABEL_RE = re.compile(r"(?i)fig\.\s*(\d+(?:\.\d+)*)")
TABLE_NUMBER_RE = re.compile(r"(?i)\b(?:table|tbl\.?)\s*(\d+(?:\.\d+)*)")


def _box_area(box: tuple[int, int, int, int]) -> int:
    xmin, ymin, xmax, ymax = box
    return max(0, xmax - xmin) * max(0, ymax - ymin)


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = box
    return (xmin + xmax) / 2, (ymin + ymax) / 2


def _horizontal_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return max(0, max(a[0], b[0]) - min(a[2], b[2]))


def _horizontal_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))


def _crop_formula_box(image: Image.Image, box: tuple[int, int, int, int]) -> bytes:
    expanded = expand_formula_bbox(box, image.size)
    return _crop_box(image, expanded, padding=0)


def _crop_box(image: Image.Image, box: tuple[int, int, int, int], padding: int = 5) -> bytes:
    xmin, ymin, xmax, ymax = box
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(image.width, xmax + padding)
    ymax = min(image.height, ymax + padding)
    cropped = image.crop((xmin, ymin, xmax, ymax))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def _render_figure_bytes(
    page: fitz.Page,
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    dpi: int,
    *,
    padding: int = 2,
) -> bytes:
    """
    Render figure pixels from the PDF vector page (not a pre-rasterized PIL crop).

    NCERT diagrams keep correct colours/masks when rendered via PyMuPDF clip;
    cropping the layout raster at pipeline DPI often disturbs Fig 3.3 / 3.4 quality.
    """
    rect = image_box_to_pdf_rect(box, page.rect, image_size, dpi)
    pad = padding * (72.0 / max(dpi, 72))
    clip = fitz.Rect(
        max(0, rect.x0 - pad),
        max(0, rect.y0 - pad),
        min(page.rect.width, rect.x1 + pad),
        min(page.rect.height, rect.y1 + pad),
    )
    scale = max(2.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    return pix.tobytes("png")


def _figure_image_bytes(
    page: fitz.Page,
    image: Image.Image,
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    dpi: int,
    *,
    fig_number: str | None = None,
) -> bytes:
    if fig_number:
        layout_rect = image_box_to_pdf_rect(box, page.rect, image_size, dpi)
        raw = try_extract_embedded_jpeg_bytes(page, fig_number, layout_rect)
        if raw:
            return raw
    try:
        return _render_figure_bytes(page, box, image_size, dpi)
    except Exception:
        return _crop_box(image, box)


def _text_from_region(page: fitz.Page, box: tuple[int, int, int, int], image_size: tuple[int, int], dpi: int) -> str:
    rect = image_box_to_pdf_rect(box, page.rect, image_size, dpi)
    if rect.is_empty:
        return ""
    return (page.get_text("text", clip=rect) or "").strip()


def _expand_label_box(page: fitz.Page, rect: fitz.Rect, dpi: int) -> tuple[int, int, int, int]:
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            line_rect = fitz.Rect(line["bbox"])
            if line_rect.intersects(rect) or (
                abs(line_rect.y0 - rect.y0) < 4 and line_rect.x0 <= rect.x0 <= line_rect.x1
            ):
                return pdf_rect_to_image_box(line_rect, dpi)
    return pdf_rect_to_image_box(rect, dpi)


def _score_label_candidate(
    page: fitz.Page,
    rect: fitz.Rect,
    fig_number: str,
    figures: list[dict[str, Any]],
    figure_captions: list[dict[str, Any]],
    dpi: int,
) -> tuple[int, tuple[int, int, int, int], str]:
    label_box = _expand_label_box(page, rect, dpi)
    context_rect = fitz.Rect(
        max(0, rect.x0 - 80),
        max(0, rect.y0 - 8),
        rect.x1 + 260,
        rect.y1 + 40,
    )
    context = page.get_text("text", clip=context_rect)
    score = 0
    if re.search(rf"(?i)fig\.\s*{re.escape(fig_number)}\s*\.\s+\w", context):
        score += 120
    if re.search(rf"(?i)\(fig\.\s*{re.escape(fig_number)}\)", context):
        score -= 150
    for caption in figure_captions:
        if _horizontal_gap(label_box, caption["box"]) < 80:
            score += 90
    for figure in figures:
        fig_box = figure["box"]
        if label_box[1] >= fig_box[1] - 30:
            gap = max(0, label_box[1] - fig_box[3])
            if gap < 120 and _horizontal_gap(label_box, fig_box) < 140:
                score += 70 - int(gap * 0.2)
    return score, label_box, context.strip()


def _find_figure_labels(
    page: fitz.Page,
    figures: list[dict[str, Any]],
    figure_captions: list[dict[str, Any]],
    dpi: int,
) -> list[dict[str, Any]]:
    labels_by_number: dict[str, dict[str, Any]] = {}
    for match in FIGURE_LABEL_RE.finditer(page.get_text()):
        label_text = match.group(0)
        fig_number = match.group(1)
        for rect in page.search_for(label_text):
            score, label_box, context = _score_label_candidate(
                page, rect, fig_number, figures, figure_captions, dpi
            )
            candidate = {
                "fig_number": fig_number,
                "label_text": label_text,
                "label_box": label_box,
                "score": score,
                "context": context,
            }
            current = labels_by_number.get(fig_number)
            if current is None or score > current["score"]:
                labels_by_number[fig_number] = candidate
    return list(labels_by_number.values())


def _figure_match_score(label_box: tuple[int, int, int, int], figure_box: tuple[int, int, int, int]) -> float:
    if label_box[3] < figure_box[1] - 40:
        return float("inf")
    vertical_gap = max(0, label_box[1] - figure_box[3])
    if label_box[1] < figure_box[3]:
        vertical_gap = max(0, figure_box[1] - label_box[3]) * 0.5
    horizontal_penalty = _horizontal_gap(label_box, figure_box)
    if _horizontal_overlap(label_box, figure_box) > 0:
        horizontal_penalty *= 0.35
    return vertical_gap * 2.5 + horizontal_penalty * 0.4


def _assign_labels_to_figures(
    labels: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    max_score: float = 500,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairings: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for label in labels:
        for figure in figures:
            score = _figure_match_score(label["label_box"], figure["box"])
            if score < max_score:
                pairings.append((score, label, figure))
    pairings.sort(key=lambda x: x[0])
    used_numbers: set[str] = set()
    used_figures: set[int] = set()
    assignments: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _, label, figure in pairings:
        if label["fig_number"] in used_numbers:
            continue
        fid = id(figure)
        if fid in used_figures:
            continue
        used_numbers.add(label["fig_number"])
        used_figures.add(fid)
        assignments.append((label, figure))
    return assignments


def _fallback_figure_box(
    label_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    figure_height: int = 300,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = label_box
    width = max(220, xmax - xmin + 40)
    cx = (xmin + xmax) / 2
    crop_xmin = max(0, int(cx - width / 2))
    crop_xmax = min(image_size[0], int(cx + width / 2))
    crop_ymax = max(0, int(ymin - 8))
    crop_ymin = max(0, int(crop_ymax - figure_height))
    return crop_xmin, crop_ymin, crop_xmax, crop_ymax


def _caption_for_label(
    page: fitz.Page,
    label_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    dpi: int,
    *,
    fig_number: str | None = None,
) -> str:
    xmin, ymin, xmax, ymax = label_box
    line_box = (
        max(0, xmin - 40),
        max(0, ymin - 8),
        min(image_size[0], xmax + 360),
        min(image_size[1], ymax + 12),
    )
    line_text = _text_from_region(page, line_box, image_size, dpi)
    if fig_number and re.search(rf"(?i)fig\.?\s*{re.escape(fig_number)}\b", line_text):
        cleaned = re.sub(r"\s+", " ", line_text.strip())
        if len(cleaned) >= 8:
            return cleaned[:500]

    search_box = (
        max(0, xmin - 30),
        ymax,
        min(image_size[0], xmax + 220),
        min(image_size[1], ymax + 180),
    )
    below = _text_from_region(page, search_box, image_size, dpi)
    return below or line_text


def _nearby_text_for_figure(
    page: fitz.Page,
    figure_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    dpi: int,
) -> tuple[str, str]:
    xmin, ymin, xmax, ymax = figure_box
    above_box = (
        max(0, xmin - 50),
        max(0, ymin - 460),
        min(image_size[0], xmax + 50),
        ymin,
    )
    below_box = (
        max(0, xmin - 40),
        ymax,
        min(image_size[0], xmax + 220),
        min(image_size[1], ymax + 180),
    )
    return (
        _text_from_region(page, above_box, image_size, dpi),
        _text_from_region(page, below_box, image_size, dpi),
    )


def _caption_from_layout(
    figure_box: tuple[int, int, int, int],
    figure_captions: list[dict[str, Any]],
    page: fitz.Page,
    image_size: tuple[int, int],
    dpi: int,
) -> str:
    best: tuple[float, str] | None = None
    for cap in figure_captions:
        cap_box = cap["box"]
        if cap_box[1] < figure_box[3] - 30:
            continue
        gap = cap_box[1] - figure_box[3]
        if gap > 220:
            continue
        overlap = _horizontal_overlap(figure_box, cap_box)
        if overlap <= 0:
            continue
        text = cap.get("text") or _text_from_region(page, cap_box, image_size, dpi)
        score = overlap - gap * 0.2
        if text.strip() and (best is None or score > best[0]):
            best = (score, text.strip())
    return best[1][:500] if best else ""


def _find_table_number(text: str) -> str | None:
    m = TABLE_NUMBER_RE.search(text or "")
    return m.group(1) if m else None


def _caption_from_embedded_context(
    page: fitz.Page,
    rect: fitz.Rect,
    nearby_before: str,
    nearby_after: str,
) -> str:
    """Best caption for an embedded photo from PDF text layer (not layout OCR)."""
    for blob in (nearby_after, nearby_before):
        for line in (blob or "").splitlines():
            line = re.sub(r"\s+", " ", line.strip())
            if len(line) >= 24:
                return line[:500]
    below = page.get_text(
        "text",
        clip=fitz.Rect(rect.x0 - 20, rect.y1, rect.x1 + 40, min(page.rect.height, rect.y1 + 140)),
    )
    below = re.sub(r"\s+", " ", (below or "").strip())
    if len(below) >= 16:
        return below[:500]
    return ""


def _embedded_teaching_rects_in_image_box(
    page: fitz.Page,
    figure_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    dpi: int,
) -> list[fitz.Rect]:
    """Embedded PDF photos whose center lies inside a layout figure box."""
    from app.services.pdf_extract_pipeline.embedded_figure_refinement import (
        list_embedded_figure_rects,
    )

    layout_rect = image_box_to_pdf_rect(figure_box, page.rect, image_size, dpi)
    pw, ph = page.rect.width, page.rect.height
    page_area = pw * ph
    hits: list[fitz.Rect] = []
    for rect in list_embedded_figure_rects(page):
        if page_area > 0 and (rect.width * rect.height) / page_area > 0.40:
            continue
        if rect.width * rect.height < 8000:
            continue
        cx, cy = (rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2
        if layout_rect.contains(fitz.Point(cx, cy)):
            hits.append(rect)
    return hits


def _layout_box_spans_multiple_photos(
    page: fitz.Page,
    figure_box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    dpi: int,
) -> bool:
    return len(_embedded_teaching_rects_in_image_box(page, figure_box, image_size, dpi=dpi)) >= 2


def _extract_embedded_pdf_figures(
    page: fitz.Page,
    image: Image.Image,
    *,
    dpi: int,
    page_no: int,
    page_assigned_boxes: list[tuple[int, int, int, int]],
) -> list[ExtractedAsset]:
    """
    Supplement layout YOLO with embedded PDF XObject images.

    NCERT collage pages often place teaching photos as embedded JPEGs that YOLO
  misses while detecting unrelated regions (e.g. activity photos at page bottom).
    """
    from app.services.pdf_extract_pipeline.embedded_figure_refinement import (
        list_embedded_figure_rects,
    )

    pw, ph = page.rect.width, page.rect.height
    page_area = pw * ph
    out: list[ExtractedAsset] = []
    assigned = list(page_assigned_boxes)

    for rect in sorted(
        list_embedded_figure_rects(page),
        key=lambda r: -(r.width * r.height),
    ):
        if page_area > 0:
            area_frac = (rect.width * rect.height) / page_area
            if area_frac > 0.40:
                continue
        if rect.width * rect.height < 12000:
            continue

        box = pdf_rect_to_image_box(rect, dpi)
        if _box_area(box) < 4000:
            continue
        if is_near_duplicate_box(box, assigned):
            continue

        nearby_before, nearby_after = _nearby_text_for_figure(page, box, image.size, dpi)
        caption = _caption_from_embedded_context(page, rect, nearby_before, nearby_after)
        if is_prose_figure_region(page, box, image.size, dpi, caption=caption):
            continue

        out.append(
            ExtractedAsset(
                asset_type="figure",
                page_no=page_no,
                image_bytes=_figure_image_bytes(page, image, box, image.size, dpi),
                number=None,
                caption=caption,
                nearby_before=nearby_before,
                nearby_after=nearby_after,
                bbox=box,
                source="embedded_pdf",
            )
        )
        assigned.append(box)

    return out


def extract_assets_from_page(
    page_extraction: PageExtraction,
    page: fitz.Page,
    *,
    dpi: int,
    document_has_fig_numbers: bool,
    table_structured: dict[int, str] | None = None,
    assigned_fig_numbers: set[str] | None = None,
    assigned_figure_boxes: list[tuple[int, int, int, int]] | None = None,
) -> list[ExtractedAsset]:
    image = page_extraction.pil_image
    if image is None:
        return []
    image_size = image.size
    page_no = page_extraction.page_no + 1

    figures: list[dict[str, Any]] = []
    figure_captions: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    table_captions: list[dict[str, Any]] = []
    formulas: list[dict[str, Any]] = []

    for det in page_extraction.layout_dets:
        box = det.bbox
        item = {"det": det, "box": box}
        cat = det.category_type
        if cat == "figure":
            figures.append(item)
        elif cat == "figure_caption":
            figure_captions.append(item)
        elif cat == "table":
            tables.append(item)
        elif cat == "table_caption":
            table_captions.append(item)
        elif cat in ("isolate_formula", "isolated", "inline"):
            formulas.append(item)

    for caption in table_captions:
        caption["text"] = _text_from_region(page, caption["box"], image_size, dpi)

    assets: list[ExtractedAsset] = []
    global_assigned_numbers = assigned_fig_numbers or set()
    page_assigned_boxes = list(assigned_figure_boxes or [])
    figure_labels = _find_figure_labels(page, figures, figure_captions, dpi)
    assignments = _assign_labels_to_figures(figure_labels, figures)
    assigned_numbers = {lbl["fig_number"] for lbl, _ in assignments}
    assigned_ids = {id(fig) for _, fig in assignments}

    for label, figure in assignments:
        box = figure["box"]
        if _box_area(box) < 4000:
            continue
        fig_number = label["fig_number"]
        box = refine_figure_box(
            page,
            layout_box=box,
            label_box=label["label_box"],
            fig_number=fig_number,
            image_size=image_size,
            dpi=dpi,
        )
        nearby_before, nearby_after = _nearby_text_for_figure(page, box, image_size, dpi)
        layout_cap = _caption_from_layout(box, figure_captions, page, image_size, dpi)
        caption = (
            layout_cap
            or _caption_for_label(page, label["label_box"], image_size, dpi, fig_number=fig_number)
            or label.get("context")
            or label["label_text"]
        )
        assets.append(
            ExtractedAsset(
                asset_type="figure",
                page_no=page_no,
                image_bytes=_figure_image_bytes(
                    page, image, box, image_size, dpi, fig_number=fig_number
                ),
                number=fig_number,
                caption=caption,
                nearby_before=nearby_before,
                nearby_after=nearby_after,
                bbox=box,
                source="layout",
            )
        )
        page_assigned_boxes.append(box)

    for label in figure_labels:
        fig_number = label["fig_number"]
        if fig_number in assigned_numbers or fig_number in global_assigned_numbers:
            continue
        box = _fallback_figure_box(label["label_box"], image_size)
        if _box_area(box) < 4000:
            continue
        caption = (
            _caption_for_label(page, label["label_box"], image_size, dpi)
            or label.get("context")
            or label["label_text"]
        )
        if is_prose_figure_region(page, box, image_size, dpi, caption=caption):
            continue
        assets.append(
            ExtractedAsset(
                asset_type="figure",
                page_no=page_no,
                image_bytes=_figure_image_bytes(
                    page, image, box, image_size, dpi, fig_number=fig_number
                ),
                number=fig_number,
                caption=caption,
                bbox=box,
                source="layout_fallback",
            )
        )
        page_assigned_boxes.append(box)

    orphan_source = "layout_unnumbered" if not document_has_fig_numbers else "layout_orphan"
    for fig in figures:
        if id(fig) in assigned_ids:
            continue
        box = fig["box"]
        if _box_area(box) < 4000:
            continue
        if is_near_duplicate_box(box, page_assigned_boxes):
            continue
        if _layout_box_spans_multiple_photos(page, box, image_size, dpi=dpi):
            continue
        nearby_before, nearby_after = _nearby_text_for_figure(page, box, image_size, dpi)
        section = section_heading_above_figure(page, box, image_size, dpi)
        caption = section or nearby_before.splitlines()[-1].strip()[:200] if nearby_before else ""
        if is_prose_figure_region(page, box, image_size, dpi, caption=caption):
            continue
        assets.append(
            ExtractedAsset(
                asset_type="figure",
                page_no=page_no,
                image_bytes=_figure_image_bytes(page, image, box, image_size, dpi),
                number=None,
                caption=caption,
                nearby_before=nearby_before,
                nearby_after=nearby_after,
                bbox=box,
                source=orphan_source,
            )
        )
        page_assigned_boxes.append(box)

    for formula in formulas:
        box = formula["box"]
        if _box_area(box) < 400:
            continue
        latex = formula["det"].latex or ""
        pdf_math = extract_formula_text_from_pdf(page, box, image_size, dpi)
        structured = merge_formula_structured_text(pdf_math, latex)
        expanded_box = expand_formula_bbox(box, image_size)
        assets.append(
            ExtractedAsset(
                asset_type="formula",
                page_no=page_no,
                image_bytes=_crop_formula_box(image, box),
                structured_text=structured,
                bbox=expanded_box,
                source="layout",
            )
        )

    for idx, table in enumerate(tables, start=1):
        box = table["box"]
        caption_item = None
        if table_captions:
            caption_item = min(
                table_captions,
                key=lambda c: abs(_box_center(c["box"])[1] - _box_center(box)[1]),
            )
        caption_text = caption_item["text"] if caption_item else ""
        table_number = _find_table_number(caption_text)
        structured = ""
        if table_structured and idx - 1 in table_structured:
            structured = table_structured[idx - 1]
        assets.append(
            ExtractedAsset(
                asset_type="table",
                page_no=page_no,
                image_bytes=_figure_image_bytes(page, image, box, image_size, dpi),
                number=table_number,
                caption=caption_text,
                structured_text=structured,
                bbox=box,
                source="layout",
            )
        )

    assets.extend(
        _extract_embedded_pdf_figures(
            page,
            image,
            dpi=dpi,
            page_no=page_no,
            page_assigned_boxes=page_assigned_boxes,
        )
    )

    return assets
