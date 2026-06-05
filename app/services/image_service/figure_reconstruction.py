"""
Caption-anchored figure reconstruction for educational PDFs.

  Caption anchors → caption-centered search → collect/fuse objects →
  single-figure bounds → region render (3×) → validate → orphan recovery
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import DEBUG_FIGURE_BBOXES, PROJECT_ROOT
from app.services.image_service.pdf_layout_extraction import (
    BBox,
    DocumentExtractionResult,
    LayoutCaption,
    LayoutTextBlock,
    PageExtractionLog,
    PairedFigure,
    _FIG_MARKER_RE,
    _MIN_IMAGE_PT,
    _MIN_PIXEL_AREA,
    _MIN_PIXEL_DIM,
    _text_near_figure,
    build_figure_context_layout,
    collect_text_blocks,
    collect_vector_drawing_candidates,
    detect_captions_from_blocks,
    detect_layout_regions,
    detect_section_titles_from_blocks,
    is_page_background,
    pairing_confidence_from_distance,
    pairing_distance,
)

logger = logging.getLogger(__name__)

# --- Stage 2: search window ---
_SEARCH_HEIGHT_FRAC = 0.50
_SEARCH_HEIGHT_MAX_PT = 500.0
_SEARCH_MARGIN_X = 180.0
_SEARCH_BESIDE_Y_PAD = 100.0
_MAX_EMBED_AREA_FRAC = 0.28
# NCERT full-width map/diagram plates (e.g. Fig 2.13) — still teaching figures, not page shells.
_MAX_PLATE_EMBED_AREA_FRAC = 0.58
_WIDE_CAPTION_WIDTH_FRAC = 0.30
_MAX_PAIR_DISTANCE = 320.0
_MIN_CROP_HEIGHT_PT = 72.0

# --- Stage 3: bbox fusion ---
_FUSION_IOU_THRESHOLD = 0.30
_MATCH_IOU = 0.30

# --- Stage 4: whitespace expansion ---
_BBOX_PADDING_PT = 12.0
_CAPTION_GAP_PT = 6.0
_SIDE_TEXT_TRIM_FRAC = 0.14
_BODY_TEXT_BAND_FRAC = 0.28
_BODY_TEXT_DENSITY_TRIM = 0.42

# --- Stage 5: region rendering ---
_RENDER_MATRIX_SCALE = 3.0

# --- Stage 6: completeness (Change 1) ---
_COMPLETENESS_MIN_FULL = 0.55
_COMPLETENESS_MIN_MINIMAL = 0.35

# --- Stage 7: failure rejection ---
_STRIP_WIDTH_FRAC = 0.15
_STRIP_HEIGHT_FRAC = 0.10
_MIN_AREA_FRAC = 0.01
_ASPECT_MAX = 5.0
_ASPECT_MIN = 0.15
_TEXT_DENSITY_MAX = 0.70
_LAYOUT_ZONE_MAX_TEXT = 0.55

# --- Change 5: orphan recovery ---
_ORPHAN_SEARCH_PT = 150.0

_DEBUG_DIR = os.path.join(PROJECT_ROOT, "debug_figure_detection")


@dataclass
class FigureAnchor:
    figure_number: str
    caption_bbox: BBox
    caption_text: str
    page_number: int
    caption_index_on_page: int = 0
    from_orphan_recovery: bool = False


@dataclass
class VisualRegion:
    bbox: BBox
    source: str


@dataclass
class FinalFigureRegion:
    figure_bbox: BBox
    object_count: int
    sources: list[str] = field(default_factory=list)
    figure_completeness: float = 0.0


@dataclass
class ReconstructedFigure:
    anchor: FigureAnchor
    region: FinalFigureRegion
    image_bytes: bytes
    caption: str
    caption_source: str
    nearby_before: str = ""
    nearby_after: str = ""
    section_title: str | None = None
    subsection_title: str | None = None
    pairing_distance: float = 0.0
    validation_flags: list[str] = field(default_factory=list)


@dataclass
class ExtractionDiagnostics:
    total_captions_found: int = 0
    total_anchors_created: int = 0
    total_visual_regions_found: int = 0
    total_matched: int = 0
    total_orphan_regions: int = 0
    total_extracted: int = 0
    rejections: list[dict[str, Any]] = field(default_factory=list)

    def log_summary(self, pdf_path: str) -> None:
        logger.info(
            "[RECON-STATS] file=%s captions=%d anchors=%d visual_regions=%d "
            "matched=%d orphan_regions=%d extracted=%d rejected=%d",
            os.path.basename(pdf_path),
            self.total_captions_found,
            self.total_anchors_created,
            self.total_visual_regions_found,
            self.total_matched,
            self.total_orphan_regions,
            self.total_extracted,
            len(self.rejections),
        )


def _log_reject(
    diag: ExtractionDiagnostics | None,
    *,
    figure_number: str,
    reason: str,
    page_number: int = 0,
    completeness: float | None = None,
    object_count: int | None = None,
    text_density: float | None = None,
) -> None:
    parts = [f"REJECT Fig {figure_number}", f"reason={reason}"]
    if completeness is not None:
        parts.append(f"completeness={completeness:.2f}")
    if object_count is not None:
        parts.append(f"objects={object_count}")
    if text_density is not None:
        parts.append(f"text_density={text_density:.2f}")
    if page_number:
        parts.append(f"page={page_number}")
    logger.info(" ".join(parts))
    if diag is not None:
        diag.rejections.append(
            {
                "figure_number": figure_number,
                "reason": reason,
                "page": page_number,
                "completeness": completeness,
                "object_count": object_count,
                "text_density": text_density,
            }
        )


def anchor_from_layout_caption(cap: LayoutCaption) -> FigureAnchor:
    return FigureAnchor(
        figure_number=cap.figure_number,
        caption_bbox=cap.bbox,
        caption_text=cap.caption_text,
        page_number=cap.page_number,
        caption_index_on_page=cap.caption_index_on_page,
    )


def figure_number_to_filename(figure_number: str | None, page_index: int, seq: int) -> str:
    if figure_number:
        parts = re.sub(r"[^\d.]", "", figure_number.strip()).strip(".").split(".")
        parts = [p for p in parts if p]
        if parts:
            return f"fig_{'_'.join(parts)}.jpg"
    return f"p{page_index}_{seq}.jpg"


def _completeness_threshold(caption_text: str) -> float:
    from app.services.image_service.figure_context_gates import is_minimal_figure_caption

    if is_minimal_figure_caption(caption_text):
        return _COMPLETENESS_MIN_MINIMAL
    return _COMPLETENESS_MIN_FULL


def _bbox_iou(a: BBox, b: BBox) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _bbox_intersects(a: BBox, b: BBox) -> bool:
    return a.intersects(b)


def _bbox_center_inside(inner: BBox, outer: BBox) -> bool:
    cx, cy = inner.center
    return outer.x0 <= cx <= outer.x1 and outer.y0 <= cy <= outer.y1


def _bbox_union_many(boxes: list[BBox]) -> BBox:
    return BBox(
        min(b.x0 for b in boxes),
        min(b.y0 for b in boxes),
        max(b.x1 for b in boxes),
        max(b.y1 for b in boxes),
    )


def _clamp_bbox(bbox: BBox, page_bbox: BBox) -> BBox:
    return BBox(
        max(page_bbox.x0, bbox.x0),
        max(page_bbox.y0, bbox.y0),
        min(page_bbox.x1, bbox.x1),
        min(page_bbox.y1, bbox.y1),
    )


def _figure_bottom_y(anchor: FigureAnchor) -> float:
    """Top edge of caption band — figure raster must end above this (caption-below layout)."""
    return anchor.caption_bbox.y0 - _CAPTION_GAP_PT


def _caption_layout_beside_figure(visual: BBox, anchor: FigureAnchor) -> bool:
    """True when the caption sits beside the figure (shared row), not under it."""
    cap = anchor.caption_bbox
    if cap.y0 >= visual.y1 - 24:
        return False
    vy0 = max(visual.y0, cap.y0 - 16)
    vy1 = min(visual.y1, cap.y1 + 16)
    if vy1 - vy0 < 12:
        return False
    h_sep = abs(visual.center[0] - cap.center[0])
    return h_sep > max(48.0, visual.width * 0.18)


def _figure_crop_bottom_y(visual: BBox, anchor: FigureAnchor) -> float:
    """Bottom of raster crop — above caption when below; keep full image when beside."""
    if _caption_layout_beside_figure(visual, anchor):
        return visual.y1
    return _figure_bottom_y(anchor)


def _caption_below_figure_in_column(anchor: FigureAnchor, page_bbox: BBox) -> bool:
    """Caption under the figure in the same column (NCERT text-left / figure-right pages)."""
    cap = anchor.caption_bbox
    ph = page_bbox.height
    pw = page_bbox.width
    if cap.y0 < ph * 0.48:
        return False
    cx = cap.center[0]
    return cx > pw * 0.52 or cx < pw * 0.48


def _search_region(anchor: FigureAnchor, page_bbox: BBox) -> BBox:
    """
    Search band for figure XObjects: above the caption and beside it (NCERT grid layouts).
    """
    ph = page_bbox.height
    pw = page_bbox.width
    search_h = min(ph * _SEARCH_HEIGHT_FRAC, _SEARCH_HEIGHT_MAX_PT)
    cap = anchor.caption_bbox
    caption_cx = (cap.x0 + cap.x1) / 2.0

    # Vertical: figures can sit above or on the same row as the caption (side-by-side).
    y0 = max(page_bbox.y0, cap.y0 - search_h)
    y1 = max(_figure_bottom_y(anchor), cap.y1 + _SEARCH_BESIDE_Y_PAD)

    # Full-width captions (weather maps, wide diagrams) — search the whole content width.
    if cap.width > pw * _WIDE_CAPTION_WIDTH_FRAC:
        return BBox(page_bbox.x0 + 16.0, y0, pw - 16.0, y1)

    # Figure above caption in the same column (e.g. rain gauge with caption bottom-right).
    if _caption_below_figure_in_column(anchor, page_bbox):
        if caption_cx > pw * 0.52:
            x0 = max(page_bbox.x0 + 16.0, cap.x0 - _SEARCH_MARGIN_X)
            x1 = min(pw - 16.0, cap.x1 + 80.0)
        else:
            x0 = max(page_bbox.x0 + 16.0, cap.x0 - 80.0)
            x1 = min(pw - 16.0, cap.x1 + _SEARCH_MARGIN_X)
        return BBox(x0, y0, x1, y1)

    # Side-by-side row: caption in margin column → search opposite column for the photo.
    if caption_cx > pw * 0.52:
        x0 = max(page_bbox.x0 + 16.0, 0.0)
        x1 = min(pw, cap.x0 + 48.0)
    elif caption_cx < pw * 0.48:
        x0 = page_bbox.x0 + 16.0
        x1 = min(pw - 16.0, cap.x1 + _SEARCH_MARGIN_X)
    else:
        x0 = max(page_bbox.x0, caption_cx - _SEARCH_MARGIN_X)
        x1 = min(pw, caption_cx + _SEARCH_MARGIN_X)

    return BBox(x0, y0, x1, y1)


def _anchor_layout_zone(anchor: FigureAnchor, page_bbox: BBox) -> BBox:
    """Narrow fallback zone (not full search band) when no objects intersect."""
    cap = anchor.caption_bbox
    caption_cx = (cap.x0 + cap.x1) / 2.0
    x0 = max(page_bbox.x0, caption_cx - _SEARCH_MARGIN_X)
    x1 = min(page_bbox.width, caption_cx + _SEARCH_MARGIN_X)
    zone_h = min(320.0, page_bbox.height * 0.40)
    y0 = max(page_bbox.y0, cap.y0 - zone_h)
    y1 = max(page_bbox.y0, cap.y0 - _CAPTION_GAP_PT)
    return BBox(x0, y0, x1, y1)


def _collect_embedded_boxes(page: Any, search: BBox, page_bbox: BBox) -> list[BBox]:
    boxes: list[BBox] = []
    for info in page.get_images(full=True):
        xref = int(info[0])
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in rects:
            bbox = BBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
            if bbox.width < _MIN_IMAGE_PT or bbox.height < _MIN_IMAGE_PT:
                continue
            if is_page_background(bbox, page_bbox):
                continue
            if _bbox_intersects(bbox, search):
                boxes.append(bbox)
    return boxes


def _filter_teaching_embedded_boxes(boxes: list[BBox], page_bbox: BBox) -> list[BBox]:
    """Drop full-page shells and repeated content-area plates; keep diagram XObjects."""
    page_area = page_bbox.area or 1.0
    kept: list[BBox] = []
    for bbox in boxes:
        if is_page_background(bbox, page_bbox):
            continue
        if bbox.area / page_area > _MAX_EMBED_AREA_FRAC:
            continue
        kept.append(bbox)
    return kept


def _is_oversized_content_plate(bbox: BBox, page_bbox: BBox) -> bool:
    page_area = page_bbox.area or 1.0
    return bbox.area / page_area > _MAX_EMBED_AREA_FRAC


def _pick_tight_layout_bbox(
    anchor: FigureAnchor,
    page: Any,
    page_number: int,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
    search: BBox,
) -> BBox | None:
    """Prefer a layout/vector zone over a full-page content plate (e.g. Fig 2.6 rain gauge)."""
    cap = anchor.caption_bbox
    page_area = page_bbox.area or 1.0
    candidates: list[BBox] = []
    for bbox in _collect_layout_boxes(page, page_number, page_bbox, text_blocks, search):
        if bbox.area / page_area > _MAX_EMBED_AREA_FRAC:
            continue
        if bbox.y1 > cap.y0 + 24.0:
            continue
        candidates.append(bbox)
    for bbox in _collect_vector_boxes(page, page_number, page_bbox, text_blocks, search):
        if bbox.area / page_area > _MAX_EMBED_AREA_FRAC * 0.85:
            continue
        if bbox.y1 > cap.y0 + 24.0:
            continue
        candidates.append(bbox)
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda b: pairing_distance(b, cap, page_bbox=page_bbox),
    )
    best = ranked[0]
    if pairing_distance(best, cap, page_bbox=page_bbox) > _MAX_PAIR_DISTANCE:
        return None
    return best


def _crop_plate_to_caption_figure_band(
    plate: BBox,
    anchor: FigureAnchor,
    text_blocks: list[LayoutTextBlock],
    page_bbox: BBox,
) -> BBox:
    """Shrink a full-page content plate to the figure column above the caption."""
    from app.services.image_service.pdf_layout_extraction import _find_figure_top_y

    cap = anchor.caption_bbox
    y1 = min(plate.y1, cap.y0 - _CAPTION_GAP_PT)
    y0 = _find_figure_top_y(text_blocks, cap.y0, page_bbox.y0, max_height=320.0)
    y0 = max(plate.y0, y0)
    if cap.width > page_bbox.width * 0.28:
        x0 = page_bbox.x0 + 36.0
        x1 = page_bbox.x1 - 36.0
    else:
        x0 = max(plate.x0, cap.x0 - _SEARCH_MARGIN_X)
        x1 = min(plate.x1, cap.x1 + 80.0)
    return _clamp_bbox(BBox(x0, y0, x1, y1), page_bbox)


def _recover_caption_plate_embedded(
    page: Any,
    search: BBox,
    page_bbox: BBox,
    anchor: FigureAnchor,
) -> BBox | None:
    """
    Recover a large caption-adjacent raster plate dropped by _MAX_EMBED_AREA_FRAC.

    NCERT often embeds full-width maps as a single XObject (~30–55 % of page area).
    """
    cap = anchor.caption_bbox
    page_area = page_bbox.area or 1.0
    best: BBox | None = None
    best_dist = float("inf")
    for bbox in _collect_embedded_boxes(page, search, page_bbox):
        if is_page_background(bbox, page_bbox):
            continue
        frac = bbox.area / page_area
        if frac <= _MAX_EMBED_AREA_FRAC or frac > _MAX_PLATE_EMBED_AREA_FRAC:
            continue
        if bbox.y1 > cap.y0 + 48.0:
            continue
        dist = pairing_distance(bbox, cap, page_bbox=page_bbox)
        if dist > _MAX_PAIR_DISTANCE * 1.35:
            continue
        if dist < best_dist:
            best_dist = dist
            best = bbox
    return best


def _union_embedded_with_diagram_vectors(
    embedded: BBox,
    vectors: list[BBox],
) -> BBox:
    """Include vector label art (e.g. 'Tropopause' arrows) that sits beside the XObject."""
    boxes = [embedded]
    for v in vectors:
        iy0 = max(embedded.y0, v.y0)
        iy1 = min(embedded.y1, v.y1)
        if iy1 - iy0 < embedded.height * 0.30:
            continue
        if v.x1 <= embedded.x0 + 80 or v.x0 >= embedded.x1 - 40:
            boxes.append(v)
        elif _bbox_intersects(v, embedded):
            boxes.append(v)
    return _bbox_union_many(boxes)


def _pick_nearest_embedded_box(
    boxes: list[BBox],
    anchor: FigureAnchor,
    page_bbox: BBox,
) -> BBox | None:
    if not boxes:
        return None
    cap = anchor.caption_bbox
    ranked = sorted(
        boxes,
        key=lambda b: pairing_distance(b, cap, page_bbox=page_bbox),
    )
    best = ranked[0]
    dist = pairing_distance(best, cap, page_bbox=page_bbox)
    if dist > _MAX_PAIR_DISTANCE:
        return None
    return best


def _collect_vector_boxes(
    page: Any,
    page_number: int,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
    search: BBox,
) -> list[BBox]:
    candidates = collect_vector_drawing_candidates(page, page_number, page_bbox, text_blocks)
    return [c.bbox for c in candidates if _bbox_intersects(c.bbox, search)]


def _collect_layout_boxes(
    page: Any,
    page_number: int,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
    search: BBox,
) -> list[BBox]:
    candidates = detect_layout_regions(page, page_number, page_bbox, text_blocks)
    return [c.bbox for c in candidates if _bbox_intersects(c.bbox, search)]


def _collect_all_visual_regions(
    page: Any,
    page_number: int,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
) -> list[VisualRegion]:
    """All visual regions on page (for orphan recovery)."""
    full_search = page_bbox
    regions: list[VisualRegion] = []
    for bbox in _collect_embedded_boxes(page, full_search, page_bbox):
        regions.append(VisualRegion(bbox=bbox, source="embedded"))
    for bbox in _collect_vector_boxes(page, page_number, page_bbox, text_blocks, full_search):
        regions.append(VisualRegion(bbox=bbox, source="vector"))
    for bbox in _collect_layout_boxes(page, page_number, page_bbox, text_blocks, full_search):
        regions.append(VisualRegion(bbox=bbox, source="layout"))
    return regions


def _cluster_bboxes_iou(bboxes: list[BBox], *, threshold: float = _FUSION_IOU_THRESHOLD) -> list[list[BBox]]:
    if not bboxes:
        return []
    clusters: list[list[BBox]] = []
    used = [False] * len(bboxes)

    for i, bbox in enumerate(bboxes):
        if used[i]:
            continue
        cluster = [bbox]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(bboxes)):
                if used[j]:
                    continue
                other = bboxes[j]
                if any(_bbox_iou(other, member) >= threshold for member in cluster):
                    cluster.append(other)
                    used[j] = True
                    changed = True
        clusters.append(cluster)
    return clusters


def _pick_cluster_for_anchor(
    clusters: list[list[BBox]],
    anchor: FigureAnchor,
    search: BBox,
    *,
    page_bbox: BBox | None = None,
) -> list[BBox]:
    if not clusters:
        return []
    cap = anchor.caption_bbox
    pb = page_bbox or search
    best: list[BBox] | None = None
    best_score = float("-inf")

    for cluster in clusters:
        union = _bbox_union_many(cluster)
        dist = pairing_distance(union, cap, page_bbox=pb)
        area_frac = union.area / pb.area if pb.area > 0 else 0.0
        score = 1000.0 - dist - area_frac * 500.0
        if union.y1 > cap.y0 + 50:
            score -= 120.0
        if score > best_score:
            best_score = score
            best = cluster
    if best is None:
        return []
    return _tighten_cluster_boxes(best, anchor, pb)


def _tighten_cluster_boxes(
    cluster: list[BBox],
    anchor: FigureAnchor,
    page_bbox: BBox,
) -> list[BBox]:
    """
    When a cluster contains separate photos (low mutual overlap), keep only the
    XObject nearest the caption instead of merging into one wide strip.
    """
    if len(cluster) <= 1:
        return cluster
    for i, a in enumerate(cluster):
        for b in cluster[i + 1 :]:
            if _bbox_iou(a, b) >= 0.25:
                return cluster
    nearest = _pick_nearest_embedded_box(cluster, anchor, page_bbox)
    return [nearest] if nearest else cluster


def _try_fallback_objects(
    page: Any,
    page_number: int,
    anchor: FigureAnchor,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
    search: BBox,
) -> list[BBox]:
    """Change 2 — layout zone then page-wide vector/layout, never full search render."""
    layout_in_search = _collect_layout_boxes(
        page, page_number, page_bbox, text_blocks, search
    )
    if layout_in_search:
        return layout_in_search

    zone = _anchor_layout_zone(anchor, page_bbox)
    if (
        zone.width >= 40
        and zone.height >= 40
        and _text_density(zone, text_blocks) < _LAYOUT_ZONE_MAX_TEXT
        and not is_page_background(zone, page_bbox)
    ):
        return [zone]

    vectors = _collect_vector_boxes(page, page_number, page_bbox, text_blocks, search)
    if vectors:
        return vectors

    layout_page = _collect_layout_boxes(
        page, page_number, page_bbox, text_blocks, page_bbox
    )
    for bbox in layout_page:
        if _bbox_intersects(bbox, search):
            return [bbox]

    return []


def find_captions_inside_bbox(
    figure_bbox: BBox,
    anchors: list[FigureAnchor],
    *,
    page_number: int,
) -> list[FigureAnchor]:
    """Change 4 — anchors whose caption lies inside figure_bbox."""
    inside: list[FigureAnchor] = []
    for a in anchors:
        if a.page_number != page_number:
            continue
        cap = a.caption_bbox
        if _bbox_center_inside(cap, figure_bbox) or _bbox_iou(cap, figure_bbox) > 0.15:
            inside.append(a)
    return inside


def _bound_single_figure_vertical(
    visual_bbox: BBox,
    anchor: FigureAnchor,
    all_anchors: list[FigureAnchor],
    page_bbox: BBox,
) -> BBox:
    """Clamp vertical extent so only this figure's caption band is included."""
    cap = anchor.caption_bbox
    same_page = [a for a in all_anchors if a.page_number == anchor.page_number]
    sorted_caps = sorted(same_page, key=lambda a: a.caption_bbox.y0)

    y0 = visual_bbox.y0
    if _caption_layout_beside_figure(visual_bbox, anchor):
        y1 = visual_bbox.y1
    else:
        y1 = min(cap.y0 - _CAPTION_GAP_PT, visual_bbox.y1)

    for a in sorted_caps:
        if a.figure_number == anchor.figure_number:
            break
        prev = a.caption_bbox
        if prev.y1 <= cap.y0:
            candidate = prev.y1 + 6.0
            # Tall diagrams often extend above an earlier caption on the same page.
            if visual_bbox.y0 < candidate - 12.0 and visual_bbox.height >= min(
                200.0, page_bbox.height * 0.22
            ):
                continue
            y0 = max(y0, candidate)

    for a in sorted_caps:
        if a.caption_bbox.y0 <= cap.y0:
            continue
        if a.caption_bbox.y0 < visual_bbox.y1 + 80:
            y1 = min(y1, a.caption_bbox.y0 - 6.0)
            break

    if y1 - y0 < 24:
        y1 = cap.y0 - _CAPTION_GAP_PT
        y0 = max(page_bbox.y0, cap.y0 - min(280.0, page_bbox.height * 0.35))

    return BBox(visual_bbox.x0, y0, visual_bbox.x1, y1)


def _caption_zone_fallback_bbox(
    anchor: FigureAnchor,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
) -> BBox | None:
    """
    Rasterize the teaching region above the caption when only page-shell XObjects exist
    (e.g. vector weather maps embedded in the content plate).
    """
    from app.services.image_service.pdf_layout_extraction import _find_figure_top_y

    cap = anchor.caption_bbox
    y1 = _figure_bottom_y(anchor)
    y0 = _find_figure_top_y(text_blocks, cap.y0, page_bbox.y0, max_height=400.0)
    if cap.width > page_bbox.width * 0.32:
        x0 = page_bbox.x0 + 36.0
        x1 = page_bbox.x1 - 36.0
    else:
        x0 = max(page_bbox.x0 + 20.0, cap.x0 - 24.0)
        x1 = min(page_bbox.x1 - 20.0, cap.x1 + 24.0)
    zone = BBox(x0, y0, x1, y1)
    if zone.height < _MIN_CROP_HEIGHT_PT or zone.width < 40:
        return None
    if is_page_background(zone, page_bbox):
        return None
    return zone


def fuse_figure_region(
    anchor: FigureAnchor,
    page_bbox: BBox,
    object_boxes: list[BBox],
    *,
    page: Any,
    page_number: int,
    text_blocks: list[LayoutTextBlock],
) -> FinalFigureRegion | None:
    """Prefer nearest teaching XObject; fall back to vectors/layout/caption zone."""
    search = _search_region(anchor, page_bbox)

    embedded = _filter_teaching_embedded_boxes(
        _collect_embedded_boxes(page, search, page_bbox), page_bbox
    )
    nearest = _pick_nearest_embedded_box(embedded, anchor, page_bbox)
    if nearest is None:
        nearest = _recover_caption_plate_embedded(page, search, page_bbox, anchor)

    tight = _pick_tight_layout_bbox(
        anchor, page, page_number, page_bbox, text_blocks, search
    )
    if nearest is not None and _is_oversized_content_plate(nearest, page_bbox):
        if tight is not None:
            nearest = tight
            sources = ["layout"]
        else:
            nearest = _crop_plate_to_caption_figure_band(
                nearest, anchor, text_blocks, page_bbox
            )
            sources = ["embedded_image", "plate_crop"]
    elif nearest is not None:
        sources = ["embedded_image"]
    elif tight is not None:
        nearest = tight
        sources = ["layout"]
    else:
        sources = []

    if nearest is not None:
        if "layout" not in sources:
            vectors = _collect_vector_boxes(
                page, anchor.page_number, page_bbox, text_blocks, search
            )
            if vectors:
                nearest = _union_embedded_with_diagram_vectors(nearest, vectors)
        return FinalFigureRegion(
            figure_bbox=nearest,
            object_count=1,
            sources=sources or ["embedded_image"],
        )

    if not object_boxes:
        object_boxes = _try_fallback_objects(
            page, page_number, anchor, page_bbox, text_blocks, search
        )

    if not object_boxes:
        zone = _caption_zone_fallback_bbox(anchor, page_bbox, text_blocks)
        if zone is None:
            return None
        return FinalFigureRegion(
            figure_bbox=zone,
            object_count=0,
            sources=["caption_zone"],
        )

    clusters = _cluster_bboxes_iou(object_boxes)
    chosen = _pick_cluster_for_anchor(clusters, anchor, search, page_bbox=page_bbox)
    if not chosen:
        zone = _caption_zone_fallback_bbox(anchor, page_bbox, text_blocks)
        if zone is None:
            return None
        return FinalFigureRegion(
            figure_bbox=zone,
            object_count=0,
            sources=["caption_zone"],
        )

    visual = _bbox_union_many(chosen)
    plate = _recover_caption_plate_embedded(page, search, page_bbox, anchor)
    if plate is not None and plate.y0 < visual.y0 - 12.0:
        visual = _bbox_union_many([visual, plate])
    return FinalFigureRegion(
        figure_bbox=visual,
        object_count=len(chosen),
        sources=["embedded", "vector", "layout"],
    )


def _trim_body_text_above(bbox: BBox, text_blocks: list[LayoutTextBlock]) -> BBox:
    """Raise y0 past body-text bands sitting above the diagram (not part of the figure)."""
    y0 = bbox.y0
    for block in text_blocks:
        text = block.text.strip()
        if len(text) < 10 or _FIG_MARKER_RE.search(text):
            continue
        # Only bands entirely above the figure top (not labels inside the diagram).
        if block.bbox.y1 <= bbox.y0 + 10.0 and block.bbox.y0 < bbox.y0:
            y0 = max(y0, block.bbox.y1 + 4.0)

    original_y0 = y0
    band_h = min(96.0, max(32.0, bbox.height * _BODY_TEXT_BAND_FRAC))
    for _ in range(12):
        if bbox.y1 - y0 < 48:
            break
        trial = BBox(bbox.x0, y0, bbox.x1, min(bbox.y1, y0 + band_h))
        if _text_density(trial, text_blocks) < _BODY_TEXT_DENSITY_TRIM:
            break
        y0 += min(28.0, band_h * 0.45)
    # Do not shred teaching diagrams — keep at least ~55% of vertical extent.
    if bbox.y1 - y0 < (bbox.y1 - original_y0) * 0.55:
        return bbox
    return BBox(bbox.x0, y0, bbox.x1, bbox.y1)


def _trim_side_text_columns(
    bbox: BBox,
    text_blocks: list[LayoutTextBlock],
    anchor: FigureAnchor,
) -> BBox:
    """Shrink horizontal extent when body-text columns overlap the figure strip."""
    cap = anchor.caption_bbox
    x0, x1 = bbox.x0, bbox.x1
    width = max(1.0, x1 - x0)
    left_limit = x0 + width * _SIDE_TEXT_TRIM_FRAC
    right_limit = x1 - width * _SIDE_TEXT_TRIM_FRAC

    for block in text_blocks:
        text = block.text.strip()
        if len(text) < 10 or _FIG_MARKER_RE.search(text):
            continue
        if block.bbox.y1 < bbox.y0 + 8 or block.bbox.y0 > bbox.y1 - 8:
            continue
        if block.bbox.intersects(cap):
            continue
        if block.bbox.x1 <= left_limit and block.bbox.x1 > x0:
            x0 = max(x0, block.bbox.x1 + 3.0)
        if block.bbox.x0 >= right_limit and block.bbox.x0 < x1:
            x1 = min(x1, block.bbox.x0 - 3.0)

    if x1 - x0 < 24:
        return bbox
    return BBox(x0, bbox.y0, x1, bbox.y1)


def _refine_figure_crop_bbox(
    bbox: BBox,
    anchor: FigureAnchor,
    text_blocks: list[LayoutTextBlock],
    page_bbox: BBox,
) -> BBox:
    """Exclude caption band, body text above, and side-column bleed from the raster."""
    bottom = _figure_crop_bottom_y(bbox, anchor)
    refined = BBox(bbox.x0, bbox.y0, bbox.x1, min(bbox.y1, bottom))
    refined = _trim_body_text_above(refined, text_blocks)
    page_area = page_bbox.area or 1.0
    if refined.area / page_area < 0.42:
        refined = _trim_side_text_columns(refined, text_blocks, anchor)
    return _clamp_bbox(refined, page_bbox)


def apply_single_figure_bounds(
    region: FinalFigureRegion,
    anchor: FigureAnchor,
    all_anchors: list[FigureAnchor],
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
) -> BBox | None:
    """One figure number per crop; bottom edge stops above the caption."""
    visual = _bound_single_figure_vertical(region.figure_bbox, anchor, all_anchors, page_bbox)
    bottom = _figure_crop_bottom_y(visual, anchor)
    final = BBox(visual.x0, visual.y0, visual.x1, min(visual.y1, bottom))

    inside = find_captions_inside_bbox(final, all_anchors, page_number=anchor.page_number)
    fig_nums = {a.figure_number for a in inside}
    if len(fig_nums) > 1:
        visual = _bound_single_figure_vertical(region.figure_bbox, anchor, all_anchors, page_bbox)
        final = BBox(visual.x0, visual.y0, visual.x1, min(visual.y1, bottom))
        inside2 = find_captions_inside_bbox(final, all_anchors, page_number=anchor.page_number)
        if len({a.figure_number for a in inside2}) > 1:
            return None

    return _refine_figure_crop_bbox(final, anchor, text_blocks, page_bbox)


def expand_figure_bbox(
    bbox: BBox,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
    all_anchors: list[FigureAnchor],
    anchor: FigureAnchor,
    *,
    padding: float = _BBOX_PADDING_PT,
) -> BBox:
    bottom = min(_figure_crop_bottom_y(bbox, anchor), bbox.y1)
    top = bbox.y0
    bbox = BBox(bbox.x0, top, bbox.x1, min(bbox.y1, bottom))
    page_area = page_bbox.area or 1.0
    page_frac = bbox.area / page_area
    top_pad = padding if bbox.height >= 200.0 else 0.0
    # Wide map/diagram plates: avoid horizontal pad that trips large_content_plate rejection.
    if page_frac > 0.38:
        side_pad = 0.0
        top_pad = min(top_pad, 8.0)
        bot_pad = min(padding, 4.0)
    else:
        side_pad = padding
        bot_pad = padding
    expanded = BBox(
        bbox.x0 - side_pad,
        max(page_bbox.y0, top - top_pad),
        bbox.x1 + side_pad,
        min(bbox.y1 + bot_pad, bottom),
    )
    from app.services.image_service.textbook_image_extraction import _MARGIN_TOUCH_PT

    expanded = _clamp_bbox(expanded, page_bbox)
    inset = _MARGIN_TOUCH_PT + 4.0
    expanded = BBox(
        max(expanded.x0, inset),
        max(expanded.y0, inset),
        min(expanded.x1, page_bbox.x1 - inset),
        min(expanded.y1, page_bbox.y1 - inset),
    )
    same_page = sorted(
        [a for a in all_anchors if a.page_number == anchor.page_number],
        key=lambda a: a.caption_bbox.y0,
    )
    for a in same_page:
        if a.figure_number == anchor.figure_number:
            break
        prev_cap = a.caption_bbox
        # Only separate rows stacked vertically — not captions on the same grid row.
        if prev_cap.y1 <= anchor.caption_bbox.y0 - 12.0:
            expanded = BBox(
                expanded.x0,
                max(expanded.y0, prev_cap.y1 + 4),
                expanded.x1,
                expanded.y1,
            )
    return expanded


def _text_density(bbox: BBox, text_blocks: list[LayoutTextBlock]) -> float:
    if bbox.area <= 0:
        return 0.0
    covered = 0.0
    for block in text_blocks:
        if _FIG_MARKER_RE.search(block.text):
            continue
        ix0 = max(bbox.x0, block.bbox.x0)
        iy0 = max(bbox.y0, block.bbox.y0)
        ix1 = min(bbox.x1, block.bbox.x1)
        iy1 = min(bbox.y1, block.bbox.y1)
        if ix1 > ix0 and iy1 > iy0:
            covered += (ix1 - ix0) * (iy1 - iy0)
    return min(1.0, covered / bbox.area)


def compute_figure_completeness(
    region: FinalFigureRegion,
    anchor: FigureAnchor,
    page_bbox: BBox,
) -> float:
    bbox = region.figure_bbox
    cap = anchor.caption_bbox

    from app.services.image_service.figure_context_gates import is_minimal_figure_caption

    caption_score = 0.55 if is_minimal_figure_caption(anchor.caption_text) else 1.0
    object_score = min(1.0, region.object_count / 3.0) if region.object_count else 0.0

    page_area = page_bbox.area or 1.0
    area_frac = bbox.area / page_area
    if area_frac < _MIN_AREA_FRAC:
        area_score = 0.0
    elif area_frac > 0.55:
        area_score = 0.25
    else:
        area_score = min(1.0, area_frac / 0.25)

    w, h = bbox.width, bbox.height
    if w <= 0 or h <= 0:
        aspect_score = 0.0
    else:
        ar = w / h
        if ar > _ASPECT_MAX or ar < _ASPECT_MIN:
            aspect_score = 0.0
        elif 0.4 <= ar <= 2.5:
            aspect_score = 1.0
        else:
            aspect_score = 0.6

    gap = max(0.0, cap.y0 - bbox.y1)
    if gap < 40:
        caption_score = min(1.0, caption_score + 0.15)

    score = (
        0.30 * caption_score
        + 0.25 * object_score
        + 0.25 * area_score
        + 0.20 * aspect_score
    )
    region.figure_completeness = round(score, 4)
    return region.figure_completeness


def reject_region_failure(
    bbox: BBox,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
    *,
    allow_diagram_labels: bool = False,
) -> str | None:
    pw, ph = page_bbox.width, page_bbox.height
    if pw <= 0 or ph <= 0:
        return "invalid_page"

    w, h = bbox.width, bbox.height
    if w / pw < _STRIP_WIDTH_FRAC:
        return "thin_vertical_strip"
    if h / ph < _STRIP_HEIGHT_FRAC:
        return "thin_horizontal_strip"
    if h < _MIN_CROP_HEIGHT_PT:
        return "crop_too_short"

    page_area = page_bbox.area
    if page_area > 0 and bbox.area / page_area < _MIN_AREA_FRAC:
        return "tiny_crop"

    if h > 0:
        ar = w / h
        if ar > _ASPECT_MAX or ar < _ASPECT_MIN:
            return "extreme_aspect_ratio"

    text_cap = 0.92 if allow_diagram_labels else _TEXT_DENSITY_MAX
    if _text_density(bbox, text_blocks) > text_cap:
        return "text_heavy_region"

    if is_page_background(bbox, page_bbox):
        # Caption-anchored full-width teaching plates (e.g. weather maps).
        if (
            allow_diagram_labels
            and page_area > 0
            and 0.34 <= bbox.area / page_area <= _MAX_PLATE_EMBED_AREA_FRAC
        ):
            pass
        else:
            return "page_background"

    return None


def render_figure_region(page: Any, bbox: BBox) -> bytes | None:
    import fitz

    from app.services.image_service.textbook_image_display import normalize_image_blob

    if bbox.width < 20 or bbox.height < 20:
        return None
    clip = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    try:
        mat = fitz.Matrix(_RENDER_MATRIX_SCALE, _RENDER_MATRIX_SCALE)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        if pix.width * pix.height < _MIN_PIXEL_AREA or pix.width < _MIN_PIXEL_DIM or pix.height < _MIN_PIXEL_DIM:
            return None
        blob = pix.tobytes("jpeg")
        return blob if normalize_image_blob(blob) else None
    except Exception as exc:
        logger.debug("render_figure_region failed: %s", exc)
        return None


def enrich_caption_if_minimal(
    anchor: FigureAnchor,
    *,
    nearby_before: str,
    nearby_after: str,
    section_title: str | None,
    subsection_title: str | None,
    chapter_title: str | None,
) -> tuple[str, str]:
    from app.services.image_service.figure_context_gates import is_minimal_figure_caption
    from app.services.image_service.caption_generator import generate_contextual_caption

    cap = anchor.caption_text
    if not is_minimal_figure_caption(cap):
        return cap, "detected"

    gen = generate_contextual_caption(
        figure_number=anchor.figure_number,
        image_type="unknown",
        section_title=section_title,
        subsection_title=subsection_title,
        chapter_title=chapter_title,
        nearby_before=nearby_before,
        nearby_after=nearby_after,
    )
    if gen:
        label = cap.split()[0] if cap else "Fig."
        enriched = f"{label} {anchor.figure_number}. {gen}"[:500]
        return enriched, "context_generated"
    return cap, "detected_minimal"


def _find_fig_marker_near_region(
    region: VisualRegion,
    text_blocks: list[LayoutTextBlock],
    page_bbox: BBox,
) -> tuple[str, str, BBox] | None:
    """Search ±150pt vertically for a fig marker adjacent to orphan region."""
    band = region.bbox.expand(_ORPHAN_SEARCH_PT)
    band = _clamp_bbox(
        BBox(band.x0, band.y0 - _ORPHAN_SEARCH_PT, band.x1, band.y1 + _ORPHAN_SEARCH_PT),
        page_bbox,
    )
    best: tuple[str, str, BBox, float] | None = None
    for block in text_blocks:
        if not _bbox_intersects(block.bbox, band):
            continue
        for m in _FIG_MARKER_RE.finditer(block.text):
            fig_num = m.group(2).strip()
            tail = block.text[m.end() :].strip(" .:;-")[:220]
            from app.services.image_service.pdf_layout_extraction import _format_caption

            cap_text = _format_caption(m.group(1), fig_num, tail)
            dist = abs(block.bbox.center[1] - region.bbox.center[1])
            if best is None or dist < best[3]:
                best = (fig_num, cap_text, block.bbox, dist)
    if best:
        return best[0], best[1], best[2]
    return None


def _region_is_matched(region: VisualRegion, matched_bboxes: list[BBox]) -> bool:
    return any(_bbox_iou(region.bbox, m) >= _MATCH_IOU for m in matched_bboxes)


def _save_debug_page_overlay(
    page: Any,
    page_index: int,
    page_bbox: BBox,
    *,
    anchors: list[FigureAnchor],
    candidate_boxes: list[BBox],
    final_boxes: list[tuple[BBox, str]],
    pdf_basename: str,
) -> None:
    """Change 7 — RED=caption, GREEN=candidates, BLUE=final."""
    import fitz

    try:
        os.makedirs(_DEBUG_DIR, exist_ok=True)

        def _draw(bbox: BBox, color: tuple[float, float, float], width: float = 2) -> None:
            page.draw_rect(
                fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                color=color,
                width=width,
                overlay=True,
            )

        for a in anchors:
            if a.page_number != page_index + 1:
                continue
            _draw(a.caption_bbox, (1, 0, 0), 2.5)
        for bbox in candidate_boxes:
            _draw(bbox, (0, 0.75, 0), 1.5)
        for bbox, _ in final_boxes:
            _draw(bbox, (0, 0.2, 1), 3)

        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out = os.path.join(_DEBUG_DIR, f"{pdf_basename}_p{page_index + 1}.png")
        pix.save(out)
        logger.info("[RECON-DEBUG] saved overlay %s", out)
    except Exception as exc:
        logger.warning("[RECON-DEBUG] overlay failed p%d: %s", page_index + 1, exc)


def reconstruct_figure_for_anchor(
    page: Any,
    anchor: FigureAnchor,
    page_bbox: BBox,
    page_number: int,
    text_blocks: list[LayoutTextBlock],
    all_anchors: list[FigureAnchor],
    *,
    section_title: str | None = None,
    subsection_title: str | None = None,
    chapter_title: str | None = None,
    diag: ExtractionDiagnostics | None = None,
    debug_candidates: list[BBox] | None = None,
    debug_finals: list[tuple[BBox, str]] | None = None,
) -> ReconstructedFigure | None:
    search = _search_region(anchor, page_bbox)

    object_boxes: list[BBox] = []
    object_boxes.extend(_collect_embedded_boxes(page, search, page_bbox))
    object_boxes.extend(_collect_vector_boxes(page, page_number, page_bbox, text_blocks, search))
    object_boxes.extend(_collect_layout_boxes(page, page_number, page_bbox, text_blocks, search))

    if debug_candidates is not None:
        debug_candidates.extend(object_boxes)

    region = fuse_figure_region(
        anchor,
        page_bbox,
        object_boxes,
        page=page,
        page_number=page_number,
        text_blocks=text_blocks,
    )
    if region is None:
        _log_reject(
            diag,
            figure_number=anchor.figure_number,
            reason="no_objects_found",
            page_number=page_number,
        )
        return None

    bounded = apply_single_figure_bounds(region, anchor, all_anchors, page_bbox, text_blocks)
    if bounded is None:
        _log_reject(
            diag,
            figure_number=anchor.figure_number,
            reason="multiple_captions",
            page_number=page_number,
            object_count=region.object_count,
        )
        return None

    region.figure_bbox = expand_figure_bbox(
        bounded, page_bbox, text_blocks, all_anchors, anchor
    )
    region.figure_bbox = _refine_figure_crop_bbox(
        region.figure_bbox, anchor, text_blocks, page_bbox
    )

    td = _text_density(region.figure_bbox, text_blocks)
    allow_labels = "embedded_image" in (region.sources or [])
    reject_reason = reject_region_failure(
        region.figure_bbox,
        page_bbox,
        text_blocks,
        allow_diagram_labels=allow_labels,
    )
    if reject_reason:
        _log_reject(
            diag,
            figure_number=anchor.figure_number,
            reason=reject_reason,
            page_number=page_number,
            object_count=region.object_count,
            text_density=td,
        )
        return None

    completeness = compute_figure_completeness(region, anchor, page_bbox)
    threshold = _completeness_threshold(anchor.caption_text)
    if completeness < threshold:
        _log_reject(
            diag,
            figure_number=anchor.figure_number,
            reason="low_completeness",
            page_number=page_number,
            completeness=completeness,
            object_count=region.object_count,
            text_density=td,
        )
        return None

    blob = render_figure_region(page, region.figure_bbox)
    if not blob:
        _log_reject(
            diag,
            figure_number=anchor.figure_number,
            reason="render_failed",
            page_number=page_number,
            completeness=completeness,
        )
        return None

    if debug_finals is not None:
        debug_finals.append((region.figure_bbox, anchor.figure_number))

    before, after = _text_near_figure(text_blocks, region.figure_bbox, anchor.caption_bbox)
    caption, caption_source = enrich_caption_if_minimal(
        anchor,
        nearby_before=before,
        nearby_after=after,
        section_title=section_title,
        subsection_title=subsection_title,
        chapter_title=chapter_title,
    )

    flags: list[str] = []
    if completeness < 0.75:
        flags.append("moderate_completeness")
    if anchor.from_orphan_recovery:
        flags.append("orphan_recovery")

    dist = max(0.0, anchor.caption_bbox.y0 - region.figure_bbox.y1)

    if diag is not None:
        diag.total_matched += 1

    return ReconstructedFigure(
        anchor=anchor,
        region=region,
        image_bytes=blob,
        caption=caption,
        caption_source=caption_source,
        nearby_before=before,
        nearby_after=after,
        section_title=section_title,
        subsection_title=subsection_title,
        pairing_distance=dist,
        validation_flags=flags,
    )


def _recover_orphan_figures(
    page: Any,
    page_number: int,
    page_index: int,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
    all_anchors: list[FigureAnchor],
    matched_bboxes: list[BBox],
    extracted_numbers: set[str],
    *,
    section_title: str | None,
    subsection_title: str | None,
    chapter_title: str | None,
    diag: ExtractionDiagnostics | None,
    max_figures: int,
    current_count: int,
    debug_candidates: list[BBox] | None = None,
    debug_finals: list[tuple[BBox, str]] | None = None,
) -> list[ReconstructedFigure]:
    """Change 5 — recover fig-number-only anchors from unmatched visual regions."""
    if current_count >= max_figures:
        return []

    all_regions = _collect_all_visual_regions(page, page_number, page_bbox, text_blocks)
    orphans = [r for r in all_regions if not _region_is_matched(r, matched_bboxes)]
    if diag is not None:
        diag.total_orphan_regions += len(orphans)

    recovered: list[ReconstructedFigure] = []
    for region in orphans:
        if current_count + len(recovered) >= max_figures:
            break
        found = _find_fig_marker_near_region(region, text_blocks, page_bbox)
        if not found:
            continue
        fig_num, cap_text, cap_bbox = found
        if fig_num in extracted_numbers:
            continue

        orphan_anchor = FigureAnchor(
            figure_number=fig_num,
            caption_bbox=cap_bbox,
            caption_text=cap_text,
            page_number=page_number,
            from_orphan_recovery=True,
        )
        all_anchors.append(orphan_anchor)
        if diag is not None:
            diag.total_anchors_created += 1

        rec = reconstruct_figure_for_anchor(
            page,
            orphan_anchor,
            page_bbox,
            page_number,
            text_blocks,
            all_anchors,
            section_title=section_title,
            subsection_title=subsection_title,
            chapter_title=chapter_title,
            diag=diag,
            debug_candidates=debug_candidates,
            debug_finals=debug_finals,
        )
        if rec:
            recovered.append(rec)
            extracted_numbers.add(fig_num)
            matched_bboxes.append(rec.region.figure_bbox)

    return recovered


def extract_document_by_caption_anchors(
    pdf_path: str,
    *,
    max_pages: int | None = None,
    max_figures: int = 96,
    chapter_title: str | None = None,
) -> DocumentExtractionResult:
    import fitz

    doc = fitz.open(pdf_path)
    figures: list[PairedFigure] = []
    page_logs: list[PageExtractionLog] = []
    global_seq = 0
    used_filenames: set[str] = set()
    diag = ExtractionDiagnostics()
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

    try:
        n_pages = len(doc)
        if max_pages is not None:
            n_pages = min(n_pages, max_pages)

        for page_index in range(n_pages):
            if len(figures) >= max_figures:
                break

            page = doc[page_index]
            page_number = page_index + 1
            page_bbox = BBox(0, 0, float(page.rect.width), float(page.rect.height))

            log = PageExtractionLog(page_number=page_number)
            text_blocks = collect_text_blocks(page, page_number)
            captions = detect_captions_from_blocks(page_number, text_blocks)
            anchors = [anchor_from_layout_caption(c) for c in captions]

            diag.total_captions_found += len(captions)
            diag.total_anchors_created += len(anchors)

            page_visuals = _collect_all_visual_regions(page, page_number, page_bbox, text_blocks)
            diag.total_visual_regions_found += len(page_visuals)

            sec, subsec = detect_section_titles_from_blocks(text_blocks)
            log.captions_found = len(anchors)

            debug_candidates: list[BBox] = []
            debug_finals: list[tuple[BBox, str]] = []
            reconstructed: list[ReconstructedFigure] = []
            matched_bboxes: list[BBox] = []
            extracted_numbers: set[str] = set()

            for anchor in anchors:
                if len(figures) + len(reconstructed) >= max_figures:
                    break
                rec = reconstruct_figure_for_anchor(
                    page,
                    anchor,
                    page_bbox,
                    page_number,
                    text_blocks,
                    anchors,
                    section_title=sec,
                    subsection_title=subsec,
                    chapter_title=chapter_title,
                    diag=diag,
                    debug_candidates=debug_candidates if DEBUG_FIGURE_BBOXES else None,
                    debug_finals=debug_finals if DEBUG_FIGURE_BBOXES else None,
                )
                if rec:
                    reconstructed.append(rec)
                    matched_bboxes.append(rec.region.figure_bbox)
                    extracted_numbers.add(anchor.figure_number)
                else:
                    log.orphan_captions.append(anchor.figure_number)

            orphan_recovered = _recover_orphan_figures(
                page,
                page_number,
                page_index,
                page_bbox,
                text_blocks,
                anchors,
                matched_bboxes,
                extracted_numbers,
                section_title=sec,
                subsection_title=subsec,
                chapter_title=chapter_title,
                diag=diag,
                max_figures=max_figures,
                current_count=len(figures) + len(reconstructed),
                debug_candidates=debug_candidates if DEBUG_FIGURE_BBOXES else None,
                debug_finals=debug_finals if DEBUG_FIGURE_BBOXES else None,
            )
            reconstructed.extend(orphan_recovered)

            log.images_found = len(reconstructed)

            if DEBUG_FIGURE_BBOXES:
                _save_debug_page_overlay(
                    page,
                    page_index,
                    page_bbox,
                    anchors=anchors,
                    candidate_boxes=debug_candidates,
                    final_boxes=debug_finals,
                    pdf_basename=pdf_basename,
                )

            for rec in reconstructed:
                if len(figures) >= max_figures:
                    break

                anchor = rec.anchor
                region = rec.region
                pairing_conf = pairing_confidence_from_distance(rec.pairing_distance)
                pairing_conf = min(1.0, pairing_conf * 0.5 + region.figure_completeness * 0.5)

                ctx = build_figure_context_layout(
                    caption=rec.caption,
                    nearby_before=rec.nearby_before,
                    nearby_after=rec.nearby_after,
                    section_title=rec.section_title,
                    subsection_title=rec.subsection_title,
                    chapter_title=chapter_title,
                )

                base_fname = figure_number_to_filename(
                    anchor.figure_number, page_index, global_seq
                )
                fname = base_fname
                n = 2
                while fname in used_filenames:
                    stem = base_fname.rsplit(".", 1)[0]
                    fname = f"{stem}_{n}.jpg"
                    n += 1
                used_filenames.add(fname)

                log.pairs.append(
                    {
                        "page": page_number,
                        "figure_number": anchor.figure_number,
                        "caption": rec.caption[:100],
                        "file_name": fname,
                        "image_bbox": region.figure_bbox.to_list(),
                        "caption_bbox": anchor.caption_bbox.to_list(),
                        "pairing_distance": round(rec.pairing_distance, 1),
                        "pairing_confidence": round(pairing_conf, 3),
                        "figure_completeness": region.figure_completeness,
                        "object_count": region.object_count,
                    }
                )

                page_cov = (
                    round(min(1.0, region.figure_bbox.area / page_bbox.area), 4)
                    if page_bbox.area > 0
                    else 0.0
                )

                figures.append(
                    PairedFigure(
                        page_number=page_number,
                        page_index=page_index,
                        sequence=global_seq,
                        figure_number=anchor.figure_number,
                        caption=rec.caption,
                        figure_context=ctx,
                        image_bbox=region.figure_bbox,
                        caption_bbox=anchor.caption_bbox,
                        pairing_distance=rec.pairing_distance,
                        pairing_confidence=pairing_conf,
                        image_bytes=rec.image_bytes,
                        nearby_before=rec.nearby_before,
                        nearby_after=rec.nearby_after,
                        section_title=rec.section_title,
                        subsection_title=rec.subsection_title,
                        validation_flags=rec.validation_flags,
                        source_type="region_render",
                        caption_source=rec.caption_source,
                        page_coverage=page_cov,
                        confidence=region.figure_completeness,
                        preferred_file_name=fname,
                    )
                )
                global_seq += 1
                diag.total_extracted += 1

            if log.captions_found or log.images_found:
                page_logs.append(log)

    finally:
        doc.close()

    diag.log_summary(pdf_path)
    logger.info(
        "[RECON] extracted %d figures from %s (%d pages logged)",
        len(figures),
        pdf_path,
        len(page_logs),
    )
    return DocumentExtractionResult(figures=figures, page_logs=page_logs)
