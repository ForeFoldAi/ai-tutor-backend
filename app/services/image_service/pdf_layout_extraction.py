"""
Production-grade layout-aware PDF figure extraction (PyMuPDF).

Multi-source pipeline for educational PDFs — NCERT, CBSE, ICSE, State Board.

Sources:
  A. Embedded raster images (XObjects via page.get_images)
  B. Vector drawing regions (clustered paths via page.get_drawings)
  C. Layout-detected figure zones (DocLayout-YOLO or heuristic fallback)

Stages:
  1. Page analysis — image rects, text blocks, drawing objects, page dimensions
  2. Multi-source figure discovery — Sources A, B, C
  3. Candidate merging — IoU / containment / score-based deduplication
  4. Caption detection — Fig./Figure markers with layout coordinates
  5. Spatial pairing — nearest caption per figure candidate
  6. Figure context — paragraphs above/below from text blocks
  7. Validation logs — orphans, suspicious pairs, background removals
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Generic figure label detector — covers NCERT (Fig./Figure), ICSE (Plate, Exhibit),
# state-board (Diagram N, Illustration N, Scheme N) and multilingual textbooks.
# Requires a digit after the label to avoid matching common words like "diagram of…"
_FIG_MARKER_RE = re.compile(
    r"(?i)\b(fig\.?|figure|diagram|illustration|plate|exhibit|scheme)\s*[\.\s:]*\s*(\d+(?:\.\d+)*)"
)
_CAPTION_STOP_RE = re.compile(
    r"(?i)(?:\.indd\b|reprint|not\s+to\s+be\s+republished|chapter\s+\d+\.indd)",
)
_MIN_IMAGE_PT = 28.0  # skip icons / bullets in page space
_MIN_PIXEL_DIM = 64   # minimum width or height in pixels (Stage 9)
_MIN_PIXEL_AREA = 4096  # minimum pixel area (Stage 9)
_MAX_PAIR_DISTANCE = 280.0  # page points; tune for A4/Letter
_RENDER_MATRIX_SCALE = 2.0
_LAYOUT_RENDER_SCALE = 300.0 / 72.0  # 300 DPI for page-level layout detection (Stage 1)
_PAGE_BLEED_Y0 = -15.0
_RECURRING_TEMPLATE_MIN_PAGES = 2
_LARGE_IMAGE_DISTANCE_PENALTY = 140.0
_BELOW_CAPTION_DISTANCE_PENALTY = 220.0
# Source B — vector drawing detection (configurable)
_VECTOR_MIN_PATHS = 3          # minimum clustered paths to form a figure candidate
_VECTOR_CLUSTER_PROXIMITY = 30.0  # pt: paths within this distance are clustered
_VECTOR_MAX_TEXT_OVERLAP = 0.40   # reject vector clusters whose area is >40% body text
# Candidate merging
_MERGE_IOU_THRESHOLD = 0.50
_MERGE_CONTAINMENT_THRESHOLD = 0.80
# Source priority weights for score-based merging (added to confidence)
_SOURCE_WEIGHT = {"embedded_image": 0.30, "vector_drawing": 0.20, "layout_detected": 0.10}


@dataclass
class BBox:
    """Axis-aligned box in PDF page coordinates (points, origin top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_list(self) -> list[float]:
        return [round(self.x0, 2), round(self.y0, 2), round(self.x1, 2), round(self.y1, 2)]

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def expand(self, margin: float) -> BBox:
        return BBox(self.x0 - margin, self.y0 - margin, self.x1 + margin, self.y1 + margin)

    def intersects(self, other: BBox) -> bool:
        return not (
            self.x1 <= other.x0
            or other.x1 <= self.x0
            or self.y1 <= other.y0
            or other.y1 <= self.y0
        )


@dataclass
class LayoutTextBlock:
    page_number: int
    bbox: BBox
    text: str


@dataclass
class LayoutImage:
    page_number: int
    image_index_on_page: int
    bbox: BBox
    xref: int
    pixel_width: int = 0
    pixel_height: int = 0
    rejected_as_background: bool = False


@dataclass
class LayoutCaption:
    page_number: int
    caption_index_on_page: int
    figure_number: str
    caption_text: str
    bbox: BBox


@dataclass
class FigureCandidate:
    """Pre-pairing figure region from any of the three discovery sources."""

    bbox: BBox
    source_type: str          # "embedded_image" | "vector_drawing" | "layout_detected"
    confidence: float         # 0.0–1.0 (detection or extraction confidence)
    image_bytes: bytes | None = None
    xref: int = -1
    pixel_width: int = 0
    pixel_height: int = 0

    @property
    def merge_score(self) -> float:
        """Score used for candidate deduplication — confidence + source weight."""
        return self.confidence + _SOURCE_WEIGHT.get(self.source_type, 0.0)


@dataclass
class PairedFigure:
    """One teaching figure ready for persistence."""

    page_number: int
    page_index: int
    sequence: int
    figure_number: str | None
    caption: str | None
    figure_context: str
    image_bbox: BBox
    caption_bbox: BBox | None
    pairing_distance: float
    pairing_confidence: float
    image_bytes: bytes
    nearby_before: str = ""
    nearby_after: str = ""
    section_title: str | None = None
    subsection_title: str | None = None
    validation_flags: list[str] = field(default_factory=list)
    # Production fields (Stage 2–3)
    source_type: str = "embedded_image"   # discovery source
    caption_source: str = "none"          # "detected" | "none"
    page_coverage: float = 0.0            # image area / page area
    confidence: float = 1.0              # figure detection confidence
    preferred_file_name: str | None = None  # e.g. fig_2_1.jpg from caption-anchored pipeline


@dataclass
class PageExtractionLog:
    page_number: int
    backgrounds_removed: int = 0
    images_found: int = 0
    captions_found: int = 0
    pairs: list[dict[str, Any]] = field(default_factory=list)
    orphan_captions: list[str] = field(default_factory=list)
    orphan_images: list[int] = field(default_factory=list)
    suspicious_pairs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentExtractionResult:
    figures: list[PairedFigure]
    page_logs: list[PageExtractionLog]


def bbox_to_json(bbox: BBox | None) -> str | None:
    if bbox is None:
        return None
    return json.dumps(bbox.to_list())


def bbox_from_json(raw: str | None) -> BBox | None:
    if not raw:
        return None
    try:
        parts = json.loads(raw)
        if isinstance(parts, list) and len(parts) == 4:
            return BBox(float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    except Exception:
        pass
    return None


def is_page_background(bbox: BBox, page_bbox: BBox, *, recurring_template: bool = False) -> bool:
    """Delegate to shared rejection heuristics in textbook_image_extraction."""
    from app.services.image_service.textbook_image_extraction import reject_figure_rect

    rejected, _ = reject_figure_rect(
        bbox.x0,
        bbox.y0,
        bbox.x1,
        bbox.y1,
        page_bbox.width,
        page_bbox.height,
        recurring_template=recurring_template,
    )
    if rejected:
        return True

    page_area = page_bbox.area
    img_area = bbox.area
    if page_area <= 0 or img_area <= 0:
        return True

    # Bleed coordinates (background extends past crop box)
    if bbox.y0 < _PAGE_BLEED_Y0 and img_area > 0.45 * page_area:
        return True
    if bbox.x0 < -10 and img_area > 0.45 * page_area:
        return True

    # Nearly coincident with full page
    ix0 = max(bbox.x0, page_bbox.x0)
    iy0 = max(bbox.y0, page_bbox.y0)
    ix1 = min(bbox.x1, page_bbox.x1)
    iy1 = min(bbox.y1, page_bbox.y1)
    if ix1 > ix0 and iy1 > iy0:
        inter = (ix1 - ix0) * (iy1 - iy0)
        if inter / page_area > 0.80:
            return True

    return False


def normalized_rect_key(bbox: BBox, page_bbox: BBox) -> tuple[float, float, float, float]:
    """Scale-invariant key for detecting repeated page templates across a PDF."""
    pw = page_bbox.width or 1.0
    ph = page_bbox.height or 1.0
    return (
        round(bbox.x0 / pw, 2),
        round(bbox.y0 / ph, 2),
        round(bbox.x1 / pw, 2),
        round(bbox.y1 / ph, 2),
    )


def scan_recurring_template_keys(pdf_path: str, *, max_pages: int | None = None) -> set[tuple[float, float, float, float]]:
    """Rects that repeat on multiple pages (publisher page shells, not figures)."""
    import fitz

    from app.services.image_service.textbook_image_extraction import reject_figure_rect

    doc = fitz.open(pdf_path)
    counts: dict[tuple[float, float, float, float], int] = {}
    try:
        n_pages = len(doc)
        if max_pages is not None:
            n_pages = min(n_pages, max_pages)
        for page_index in range(n_pages):
            page = doc[page_index]
            page_bbox = BBox(0, 0, float(page.rect.width), float(page.rect.height))
            seen_on_page: set[tuple[float, float, float, float]] = set()
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
                    rejected, _ = reject_figure_rect(
                        bbox.x0, bbox.y0, bbox.x1, bbox.y1, page_bbox.width, page_bbox.height
                    )
                    if rejected:
                        continue
                    key = normalized_rect_key(bbox, page_bbox)
                    if key in seen_on_page:
                        continue
                    seen_on_page.add(key)
                    area_frac = bbox.area / page_bbox.area if page_bbox.area > 0 else 0
                    if area_frac < 0.28:
                        continue
                    counts[key] = counts.get(key, 0) + 1
    finally:
        doc.close()
    return {k for k, n in counts.items() if n >= _RECURRING_TEMPLATE_MIN_PAGES}


def pairing_distance(
    image_bbox: BBox,
    caption_bbox: BBox,
    *,
    page_bbox: BBox | None = None,
) -> float:
    """
    Layout-aware image–caption pairing distance.

    Handles two common textbook layouts:

    Type A — caption below (or above) image:
        Uses edge-to-edge vertical gap + 0.35 × center-to-center horizontal.

    Type B — caption beside image (side-by-side, NCERT grid layouts):
        When image and caption share the same y-range (vertically overlapping),
        distance is the horizontal edge-to-edge gap × 0.5, dy=0.
        This correctly pairs Fig 2.3.1 (ants image on left) with its label
        placed to the right at the same y-level rather than with the caption
        of the next row that happens to be vertically adjacent below.
    """
    # Detect side-by-side (Type B): image and caption share a y-range
    vy0 = max(image_bbox.y0, caption_bbox.y0)
    vy1 = min(image_bbox.y1, caption_bbox.y1)
    vertically_overlapping = vy1 > vy0

    if vertically_overlapping:
        # Horizontal edge-to-edge gap: caption to the right or left of image
        dy = 0.0
        if caption_bbox.x0 >= image_bbox.x1:
            dx = caption_bbox.x0 - image_bbox.x1   # caption right of image
        elif image_bbox.x0 >= caption_bbox.x1:
            dx = image_bbox.x0 - caption_bbox.x1   # caption left of image
        else:
            dx = abs(image_bbox.center[0] - caption_bbox.center[0])  # x-overlap
        dist = dy + 0.5 * dx
    else:
        # Type A: caption strictly above or below image (edge-to-edge vertical)
        if caption_bbox.y0 >= image_bbox.y1:
            dy = caption_bbox.y0 - image_bbox.y1
        elif image_bbox.y0 >= caption_bbox.y1:
            dy = image_bbox.y0 - caption_bbox.y1
        else:
            dy = abs(image_bbox.center[1] - caption_bbox.center[1])
        dx = abs(image_bbox.center[0] - caption_bbox.center[0])
        dist = dy + 0.35 * dx

        # Penalty when image is drawn BELOW its caption (rare, usually wrong pairing)
        if image_bbox.y0 > caption_bbox.y0 + 12:
            dist += _BELOW_CAPTION_DISTANCE_PENALTY
        elif image_bbox.center[1] > caption_bbox.center[1] + 20:
            dist += _BELOW_CAPTION_DISTANCE_PENALTY * 0.5

    # Light area penalty — capped so a nearby caption always beats a distant one
    if page_bbox and page_bbox.area > 0:
        area_frac = image_bbox.area / page_bbox.area
        dist += min(60.0, _LARGE_IMAGE_DISTANCE_PENALTY * area_frac)

    return dist


def pairing_confidence_from_distance(distance: float) -> float:
    """Map distance (pt) to 0–1 confidence."""
    if distance <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 / (1.0 + distance / 80.0)))


def _format_caption(label: str, number: str, title: str) -> str:
    label = label.strip()
    number = number.strip()
    title = re.sub(r"\s+", " ", (title or "").strip(" .:;-"))
    prefix = f"{label} {number}".replace("  ", " ")
    if not title or len(title) < 2:
        return prefix[:500]
    sep = ". " if not prefix.endswith(".") else " "
    return f"{prefix}{sep}{title}"[:500]


def _merge_line_bboxes(lines: list[tuple[BBox, str]]) -> BBox:
    x0 = min(b.x0 for b, _ in lines)
    y0 = min(b.y0 for b, _ in lines)
    x1 = max(b.x1 for b, _ in lines)
    y1 = max(b.y1 for b, _ in lines)
    return BBox(x0, y0, x1, y1)


def detect_captions_from_blocks(
    page_number: int,
    text_blocks: list[LayoutTextBlock],
) -> list[LayoutCaption]:
    """
    Find Fig./Figure markers in layout text blocks and build caption bboxes + titles.
    """
    captions: list[LayoutCaption] = []
    cap_idx = 0

    # Collect captions, tagging each as inline or standalone.
    # Inline: the fig marker is preceded by "(" AND that "(" is NOT the first char of
    # the text block.  "gauge (Fig. 2.6). When it rains..." → inline.
    # Standalone: "(Fig. 2.3.1. Ants...)" where "(" is char 0 → keep; or bare label.
    captions_typed: list[tuple[LayoutCaption, bool]] = []  # (caption, is_inline)

    for block in text_blocks:
        text = block.text.strip()
        if not text:
            continue
        for m in _FIG_MARKER_RE.finditer(text):
            fig_num = m.group(2).strip()
            start = m.start()

            # Inline: "(Fig. N)" deep inside a sentence — start > 1 so that
            # "(Fig. 2.3.1. Title)" at position 0 is NOT skipped (that IS a label).
            is_inline = start > 1 and text[start - 1] == "("

            # Title: text after marker until stop pattern or 200 chars
            tail = text[m.end():]
            tail = _CAPTION_STOP_RE.split(tail, maxsplit=1)[0].strip(" .:;-")
            if len(tail) > 220:
                tail = tail[:220].rsplit(" ", 1)[0]
            cap_text = _format_caption(m.group(1), fig_num, tail)
            cap = LayoutCaption(
                page_number=page_number,
                caption_index_on_page=cap_idx,
                figure_number=fig_num,
                caption_text=cap_text,
                bbox=block.bbox,
            )
            captions_typed.append((cap, is_inline))
            cap_idx += 1

    # Deduplicate: for the same figure number prefer standalone over inline,
    # and among equal type prefer the longer descriptive title.
    def _title_len(c: LayoutCaption) -> int:
        idx = c.caption_text.find(c.figure_number)
        if idx == -1:
            return 0
        return len(c.caption_text[idx + len(c.figure_number):].strip(" .:;-()"))

    best: dict[str, tuple[LayoutCaption, bool]] = {}
    for cap, is_inline in sorted(captions_typed, key=lambda x: (x[0].bbox.y0, x[0].bbox.x0)):
        num = cap.figure_number
        if num not in best:
            best[num] = (cap, is_inline)
        else:
            ex_cap, ex_inline = best[num]
            # Non-inline always beats inline
            if is_inline and not ex_inline:
                continue
            if not is_inline and ex_inline:
                best[num] = (cap, is_inline)
            elif _title_len(cap) > _title_len(ex_cap):
                best[num] = (cap, is_inline)

    deduped = sorted((cap for cap, _ in best.values()), key=lambda c: (c.bbox.y0, c.bbox.x0))
    return deduped


def collect_text_blocks(page: Any, page_number: int) -> list[LayoutTextBlock]:
    """PyMuPDF text blocks (excluding figures)."""
    blocks: list[LayoutTextBlock] = []
    try:
        raw = page.get_text("blocks")
    except Exception:
        return blocks
    for item in raw:
        if len(item) < 5:
            continue
        x0, y0, x1, y1, text, block_no, block_type = item[0], item[1], item[2], item[3], item[4], item[5], item[6]
        if block_type != 0:
            continue
        t = (text or "").strip()
        if not t or len(t) < 2:
            continue
        blocks.append(
            LayoutTextBlock(
                page_number=page_number,
                bbox=BBox(float(x0), float(y0), float(x1), float(y1)),
                text=t,
            )
        )
    return blocks


def collect_images(
    page: Any,
    page_number: int,
    page_bbox: BBox,
    *,
    recurring_template_keys: set[tuple[float, float, float, float]] | None = None,
) -> tuple[list[tuple[LayoutImage, bytes]], list[tuple[LayoutImage, bytes]], int]:
    """
    Returns (accepted_images, tentative_recurring_images, backgrounds_removed).

    *accepted_images* passed all filters.
    *tentative_recurring_images* were rejected only because they appear at the same
    normalised position on multiple pages (recurring template).  Some PDFs rasterise
    the entire page content as one large XObject — the same image slot is reused on
    every page, so the genuine teaching diagram is incorrectly flagged as a template.
    Callers can try to pair tentative images with orphan captions before discarding them.
    """
    import fitz

    from app.services.image_service.textbook_image_extraction import log_figure_extract, reject_figure_rect

    results: list[tuple[LayoutImage, bytes]] = []
    tentative: list[tuple[LayoutImage, bytes]] = []
    backgrounds_removed = 0
    img_idx = 0
    render_matrix = fitz.Matrix(_RENDER_MATRIX_SCALE, _RENDER_MATRIX_SCALE)
    recurring = recurring_template_keys or set()

    for info in page.get_images(full=True):
        xref = int(info[0])
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for rect in rects:
            bbox = BBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
            if bbox.width < _MIN_IMAGE_PT or bbox.height < _MIN_IMAGE_PT:
                continue

            rect_key = normalized_rect_key(bbox, page_bbox)
            is_recurring = rect_key in recurring

            # Check for hard background rejection (ignore recurring flag here)
            rejected_hard, reason_hard = reject_figure_rect(
                bbox.x0, bbox.y0, bbox.x1, bbox.y1,
                page_bbox.width, page_bbox.height,
                recurring_template=False,  # check structure only
            )
            if not rejected_hard and is_page_background(bbox, page_bbox):
                rejected_hard = True
                reason_hard = "bleed_or_full_page_overlap"

            if rejected_hard:
                backgrounds_removed += 1
                log_figure_extract(
                    page=page_number, figure=None, caption_bbox=None,
                    image_bbox=bbox.to_list(), distance=None,
                    selected=False, rejected_reason=reason_hard,
                )
                continue

            # Rasterise the image regardless of recurring status
            try:
                pix = page.get_pixmap(matrix=render_matrix, clip=rect, alpha=False)
                w, h = pix.width, pix.height
                if w * h < _MIN_PIXEL_AREA or w < _MIN_PIXEL_DIM or h < _MIN_PIXEL_DIM:
                    continue
                blob = pix.tobytes("jpeg")
            except Exception:
                continue

            from app.services.image_service.textbook_image_display import normalize_image_blob

            if not normalize_image_blob(blob):
                continue

            lim = LayoutImage(
                page_number=page_number,
                image_index_on_page=img_idx,
                bbox=bbox, xref=xref,
                pixel_width=w, pixel_height=h,
            )

            if is_recurring:
                # Defer: only include if it can be paired with a caption
                tentative.append((lim, blob))
                log_figure_extract(
                    page=page_number, figure=None, caption_bbox=None,
                    image_bbox=bbox.to_list(), distance=None,
                    selected=False, rejected_reason="recurring_page_template",
                )
            else:
                results.append((lim, blob))
                img_idx += 1

    return _dedupe_near_duplicate_images(results), _dedupe_near_duplicate_images(tentative), backgrounds_removed


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


def _dedupe_near_duplicate_images(
    images: list[tuple[LayoutImage, bytes]],
    *,
    iou_threshold: float = 0.88,
) -> list[tuple[LayoutImage, bytes]]:
    """
    Drop only near-identical placements (same xref drawn twice).
    Unlike area-sort overlap dedupe, this keeps distinct figures on one page.
    """
    ranked = sorted(images, key=lambda pair: -pair[0].bbox.area)
    kept: list[tuple[LayoutImage, bytes]] = []
    for pair in ranked:
        if any(_bbox_iou(pair[0].bbox, existing[0].bbox) >= iou_threshold for existing in kept):
            continue
        kept.append(pair)
    return kept


# ---------------------------------------------------------------------------
# Geometry helpers for multi-source merging
# ---------------------------------------------------------------------------

def _bbox_intersection_area(a: BBox, b: BBox) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _text_overlap_fraction(bbox: BBox, text_blocks: list[LayoutTextBlock]) -> float:
    """Fraction of bbox area covered by text blocks (0–1)."""
    if bbox.area <= 0:
        return 0.0
    total = sum(_bbox_intersection_area(bbox, tb.bbox) for tb in text_blocks)
    return min(1.0, total / bbox.area)


def _cluster_bboxes(bboxes: list[BBox], *, proximity: float) -> list[list[BBox]]:
    """Group bboxes whose edges are within `proximity` pt of any cluster member."""
    if not bboxes:
        return []
    clusters: list[list[BBox]] = []
    used = [False] * len(bboxes)

    for i, bbox in enumerate(bboxes):
        if used[i]:
            continue
        cluster: list[BBox] = [bbox]
        used[i] = True
        # BFS: keep checking newly added members for additional neighbors
        frontier = [bbox]
        while frontier:
            cur = frontier.pop()
            for j in range(len(bboxes)):
                if used[j]:
                    continue
                other = bboxes[j]
                # Gap between cur and other in x and y (negative = overlap)
                gap_x = max(0.0, max(cur.x0, other.x0) - min(cur.x1, other.x1))
                gap_y = max(0.0, max(cur.y0, other.y0) - min(cur.y1, other.y1))
                if gap_x <= proximity and gap_y <= proximity:
                    cluster.append(other)
                    used[j] = True
                    frontier.append(other)
        clusters.append(cluster)
    return clusters


# ---------------------------------------------------------------------------
# Source B — Vector drawing detection
# ---------------------------------------------------------------------------

def collect_vector_drawing_candidates(
    page: Any,
    page_number: int,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
) -> list[FigureCandidate]:
    """
    Detect figure candidates from vector drawings (page.get_drawings).

    Clusters drawing paths into figure regions, rejecting table borders,
    tiny decorative elements, and drawing clusters that mostly overlay text.
    _VECTOR_MIN_PATHS and _VECTOR_CLUSTER_PROXIMITY are configurable constants.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []

    if not drawings:
        return []

    raw_rects: list[BBox] = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        bbox = BBox(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        if bbox.width < 8 or bbox.height < 8:
            continue
        # Skip near-horizontal rules (table borders, section dividers)
        if bbox.height < 4.0 and bbox.width > 0.40 * page_bbox.width:
            continue
        # Skip near-vertical rules
        if bbox.width < 4.0 and bbox.height > 0.40 * page_bbox.height:
            continue
        raw_rects.append(bbox)

    if len(raw_rects) < _VECTOR_MIN_PATHS:
        return []

    clusters = _cluster_bboxes(raw_rects, proximity=_VECTOR_CLUSTER_PROXIMITY)

    candidates: list[FigureCandidate] = []
    for cluster in clusters:
        if len(cluster) < _VECTOR_MIN_PATHS:
            continue

        envelope = BBox(
            min(r.x0 for r in cluster),
            min(r.y0 for r in cluster),
            max(r.x1 for r in cluster),
            max(r.y1 for r in cluster),
        )
        if envelope.width < _MIN_IMAGE_PT or envelope.height < _MIN_IMAGE_PT:
            continue
        if is_page_background(envelope, page_bbox):
            continue
        if _text_overlap_fraction(envelope, text_blocks) > _VECTOR_MAX_TEXT_OVERLAP:
            continue

        conf = min(0.90, 0.50 + (len(cluster) - _VECTOR_MIN_PATHS) * 0.02)
        candidates.append(FigureCandidate(
            bbox=envelope,
            source_type="vector_drawing",
            confidence=conf,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Source C — Layout detection (DocLayout-YOLO with heuristic fallback)
# ---------------------------------------------------------------------------

_YOLO_MODEL: Any = None
_YOLO_LOADED: bool = False


def _load_doclayout_model() -> Any:
    global _YOLO_MODEL, _YOLO_LOADED
    if _YOLO_LOADED:
        return _YOLO_MODEL
    _YOLO_LOADED = True
    try:
        from doclayout_yolo import YOLOv10  # type: ignore[import]
        _YOLO_MODEL = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
        logger.info("DocLayout-YOLO model loaded")
    except Exception as exc:
        logger.debug("DocLayout-YOLO unavailable (%s); heuristic fallback active", exc)
        _YOLO_MODEL = None
    return _YOLO_MODEL


def _detect_with_yolo(
    model: Any,
    page: Any,
    page_bbox: BBox,
    render_scale: float,
) -> list[FigureCandidate]:
    import fitz
    from io import BytesIO

    try:
        from PIL import Image as _PILImage
        mat = fitz.Matrix(render_scale, render_scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pil_img = _PILImage.open(BytesIO(pix.tobytes("png")))
    except Exception:
        return []

    try:
        results = model.predict(pil_img, imgsz=1024, conf=0.20, verbose=False)
    except Exception as exc:
        logger.warning("DocLayout-YOLO predict failed: %s", exc)
        return []

    candidates: list[FigureCandidate] = []
    _FIGURE_CLASSES = {"figure", "Figure", "picture", "Picture"}
    for result in results:
        for box in result.boxes:
            cls_label = result.names.get(int(box.cls.item()), "")
            if cls_label not in _FIGURE_CLASSES:
                continue
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pdf_bbox = BBox(x1 / render_scale, y1 / render_scale,
                            x2 / render_scale, y2 / render_scale)
            if is_page_background(pdf_bbox, page_bbox):
                continue
            candidates.append(FigureCandidate(
                bbox=pdf_bbox,
                source_type="layout_detected",
                confidence=conf,
            ))
    return candidates


def _detect_layout_heuristic(
    page_number: int,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
) -> list[FigureCandidate]:
    """
    Heuristic layout detection: project figure zones above caption blocks.

    Finds text-sparse regions (< 40% text coverage) that sit above a detected
    figure-marker caption. Used when DocLayout-YOLO is not installed.
    """
    captions = detect_captions_from_blocks(page_number, text_blocks)
    candidates: list[FigureCandidate] = []
    for cap in captions:
        fig_y1 = cap.bbox.y0
        fig_y0 = max(page_bbox.y0, fig_y1 - 300.0)
        fig_x0 = max(page_bbox.x0, cap.bbox.x0 - 20.0)
        fig_x1 = min(page_bbox.x1, cap.bbox.x1 + 20.0)
        zone = BBox(fig_x0, fig_y0, fig_x1, fig_y1)
        if zone.width < 40.0 or zone.height < 40.0:
            continue
        if is_page_background(zone, page_bbox):
            continue
        if _text_overlap_fraction(zone, text_blocks) >= 0.40:
            continue
        candidates.append(FigureCandidate(
            bbox=zone,
            source_type="layout_detected",
            confidence=0.35,
        ))
    return candidates


def detect_layout_regions(
    page: Any,
    page_number: int,
    page_bbox: BBox,
    text_blocks: list[LayoutTextBlock],
    render_scale: float = _LAYOUT_RENDER_SCALE,
) -> list[FigureCandidate]:
    """Try YOLO layout detection; fall back to caption-zone heuristic."""
    model = _load_doclayout_model()
    if model is not None:
        return _detect_with_yolo(model, page, page_bbox, render_scale)
    return _detect_layout_heuristic(page_number, page_bbox, text_blocks)


# ---------------------------------------------------------------------------
# Stage 3 — Multi-source candidate merging
# ---------------------------------------------------------------------------

def merge_figure_candidates(
    sources_dict: dict[str, list[FigureCandidate]],
    page_bbox: BBox,
) -> list[FigureCandidate]:
    """
    Merge figure candidates from Sources A/B/C into a deduplicated list.

    Uses score-based comparison (confidence + source weight) so a
    high-confidence layout detection can beat a low-confidence embedded
    image, rather than blindly preferring by source type.
    """
    all_candidates: list[FigureCandidate] = []
    for src_list in sources_dict.values():
        all_candidates.extend(src_list)

    # Background rejection pass
    all_candidates = [c for c in all_candidates
                      if not is_page_background(c.bbox, page_bbox)]

    # Sort by merge_score descending — highest score wins ties
    all_candidates.sort(key=lambda c: c.merge_score, reverse=True)

    kept: list[FigureCandidate] = []
    for candidate in all_candidates:
        absorbed = False
        for i, existing in enumerate(kept):
            iou = _bbox_iou(candidate.bbox, existing.bbox)
            if iou >= _MERGE_IOU_THRESHOLD:
                absorbed = True
                break
            # Containment: candidate is mostly inside existing
            inter = _bbox_intersection_area(candidate.bbox, existing.bbox)
            if candidate.bbox.area > 0:
                if inter / candidate.bbox.area >= _MERGE_CONTAINMENT_THRESHOLD:
                    absorbed = True
                    break
            # Reverse containment: existing mostly inside candidate
            # If candidate scores higher, replace the existing entry
            if existing.bbox.area > 0:
                if inter / existing.bbox.area >= _MERGE_CONTAINMENT_THRESHOLD:
                    if candidate.merge_score > existing.merge_score:
                        kept[i] = candidate
                    absorbed = True
                    break
        if not absorbed:
            kept.append(candidate)

    return kept


# ---------------------------------------------------------------------------
# Region renderer for non-embedded candidates (Sources B and C)
# ---------------------------------------------------------------------------

def _render_region(
    page: Any,
    bbox: BBox,
    *,
    render_scale: float = _RENDER_MATRIX_SCALE,
) -> bytes | None:
    """Rasterize a PDF page region to JPEG bytes."""
    import fitz
    from app.services.image_service.textbook_image_display import normalize_image_blob

    clip = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    try:
        mat = fitz.Matrix(render_scale, render_scale)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        w, h = pix.width, pix.height
        if w * h < _MIN_PIXEL_AREA or w < _MIN_PIXEL_DIM or h < _MIN_PIXEL_DIM:
            return None
        blob = pix.tobytes("jpeg")
        return blob if normalize_image_blob(blob) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------

def pair_images_to_captions(
    images: list[tuple[LayoutImage, bytes]],
    captions: list[LayoutCaption],
    *,
    page_bbox: BBox | None = None,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """
    Greedy minimum-distance pairing (not index order).

    Returns:
      assignments: list of (image_idx, caption_idx, distance)
      orphan_image_indices
      orphan_caption_indices
    """
    if not images:
        return [], [], list(range(len(captions)))
    if not captions:
        return [], list(range(len(images))), []

    candidates: list[tuple[float, int, int]] = []
    for ii, (img, _) in enumerate(images):
        for jj, cap in enumerate(captions):
            d = pairing_distance(img.bbox, cap.bbox, page_bbox=page_bbox)
            candidates.append((d, ii, jj))
    candidates.sort(key=lambda x: x[0])

    used_i: set[int] = set()
    used_j: set[int] = set()
    assignments: list[tuple[int, int, float]] = []

    for d, ii, jj in candidates:
        if ii in used_i or jj in used_j:
            continue
        if d > _MAX_PAIR_DISTANCE:
            continue
        used_i.add(ii)
        used_j.add(jj)
        assignments.append((ii, jj, d))

    orphan_i = [i for i in range(len(images)) if i not in used_i]
    orphan_j = [j for j in range(len(captions)) if j not in used_j]
    return assignments, orphan_i, orphan_j


def _text_near_figure(
    text_blocks: list[LayoutTextBlock],
    image_bbox: BBox,
    caption_bbox: BBox | None,
    *,
    margin: float = 18.0,
) -> tuple[str, str]:
    """Paragraphs above and below figure (layout-aware)."""
    if caption_bbox is not None:
        region = _bbox_union(image_bbox.expand(margin), caption_bbox.expand(margin))
    else:
        region = image_bbox.expand(margin)

    above_parts: list[tuple[float, str]] = []
    below_parts: list[tuple[float, str]] = []
    icy = image_bbox.center[1]

    for block in text_blocks:
        if _FIG_MARKER_RE.search(block.text):
            continue
        if _CAPTION_STOP_RE.search(block.text):
            continue
        t = block.text.strip()
        if len(t) < 12:
            continue
        cy = block.bbox.center[1]
        if not region.intersects(block.bbox) and abs(cy - icy) > 220:
            continue
        if cy < icy - 5:
            above_parts.append((cy, t))
        elif cy > icy + 5:
            below_parts.append((cy, t))

    above_parts.sort(key=lambda x: -x[0])
    below_parts.sort(key=lambda x: x[0])
    before = " ".join(t for _, t in above_parts[:4])[:1400]
    after = " ".join(t for _, t in below_parts[:4])[:1400]
    return before, after


def _bbox_union(a: BBox, b: BBox) -> BBox:
    return BBox(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))


def detect_section_titles_from_blocks(text_blocks: list[LayoutTextBlock]) -> tuple[str | None, str | None]:
    """
    Heading detection from layout blocks (position + shape).

    Detects:
      - Numbered headings: "2.1 Title" / "2.1.3 Title"
      - ALL-CAPS headings: "WEATHER STATIONS"
      - Lettered sub-sections: "a) Temperature" / "b)\t Precipitation"
        (common in NCERT / CBSE textbooks for topic sub-divisions)
    """
    section: str | None = None
    subsection: str | None = None
    _NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\s+([A-Z][A-Za-z0-9 ,\-/]{3,70})$")
    _CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z\s\-/]{5,70}$")
    # Matches "a) Temperature", "b)\t Precipitation", "c)  Atmospheric pressure"
    _LETTERED_HEADING_RE = re.compile(r"^[a-e]\)[\s\t]+([A-Z][A-Za-z ]{2,40})$")

    for block in sorted(text_blocks, key=lambda b: b.bbox.y0):
        line = block.text.strip()
        if not line or len(line) > 80 or line.endswith("."):
            continue

        m = _NUMBERED_HEADING_RE.match(line)
        if m:
            level = m.group(1).count(".")
            title = m.group(2).strip()
            if level == 1 and section is None:
                section = title
            elif level == 2 and subsection is None:
                subsection = title
            continue

        if _CAPS_HEADING_RE.match(line) and len(line) >= 6 and section is None:
            section = line.title()
            continue

        # Lettered sub-section (e.g. "b) Precipitation") → subsection title
        m2 = _LETTERED_HEADING_RE.match(line)
        if m2 and subsection is None:
            subsection = m2.group(1).strip()

    return section, subsection


def _find_figure_top_y(
    text_blocks: list["LayoutTextBlock"],
    caption_y0: float,
    page_y0: float,
    *,
    max_height: float = 320.0,
) -> float:
    """
    Find the y-coordinate of the nearest section/subsection heading above
    the caption.  The figure lives between this heading and the caption.

    Falls back to `caption_y0 - max_height` when no heading is found.
    """
    nearest_heading_y0 = None
    for block in text_blocks:
        by0 = block.bbox.y0
        if by0 >= caption_y0:
            continue
        text = block.text.strip()
        # Lettered subsections ("b) Precipitation") or ALL-CAPS headings
        if re.match(r"^[a-e]\)[\s\t]", text) or re.match(r"^[A-Z][A-Z\s\-/]{4,}", text):
            if nearest_heading_y0 is None or by0 > nearest_heading_y0:
                nearest_heading_y0 = by0

    if nearest_heading_y0 is not None:
        return max(page_y0, nearest_heading_y0 - 8.0)

    return max(page_y0, caption_y0 - max_height)


def _render_figure_crop_for_caption(
    page: Any,
    caption_bbox: BBox,
    page_bbox: BBox,
    text_blocks: "list[LayoutTextBlock] | None" = None,
    *,
    render_matrix_scale: float = _RENDER_MATRIX_SCALE,
) -> bytes | None:
    """
    Render a focused crop of the PDF page containing only the figure above
    a caption.

    Used when the figure lives inside a full-page raster (recurring template)
    rather than its own XObject.  Two-pass strategy:

    Vertical extent
      Top: nearest section/subsection heading above the caption (e.g. the
        "b) Precipitation" line).  This excludes tables or earlier figures
        that appear above the heading.  Falls back to 320 pt above caption.
      Bottom: caption top edge.

    Horizontal extent
      Start at 50 % of page width to stay in the right-hand column where
      most NCERT diagrams are placed, avoiding the left body-text column.
      End at the right page margin.

    Returns JPEG bytes, or None if the crop is too small or rendering fails.
    """
    import fitz

    from app.services.image_service.textbook_image_display import normalize_image_blob

    fig_y0 = _find_figure_top_y(
        text_blocks or [], caption_bbox.y0, page_bbox.y0
    )
    fig_y1 = caption_bbox.y0

    # Right half of the page: avoids the left body-text column entirely.
    # NCERT two-column pages place the diagram in the right ~50 % of the page.
    fig_x0 = max(caption_bbox.x0, page_bbox.width * 0.50)
    fig_x1 = page_bbox.x1

    if fig_y1 - fig_y0 < 40 or fig_x1 - fig_x0 < 40:
        return None

    clip = fitz.Rect(fig_x0, fig_y0, fig_x1, fig_y1)

    try:
        mat = fitz.Matrix(render_matrix_scale, render_matrix_scale)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        if pix.width * pix.height < 8000:
            return None
        blob = pix.tobytes("jpeg")
        return blob if normalize_image_blob(blob) else None
    except Exception:
        return None


def build_figure_context_layout(
    *,
    caption: str | None,
    nearby_before: str,
    nearby_after: str,
    section_title: str | None,
    subsection_title: str | None,
    chapter_title: str | None,
) -> str:
    from app.services.image_service.textbook_image_extraction import build_figure_context

    return build_figure_context(
        caption=caption,
        nearby_before=nearby_before,
        nearby_after=nearby_after,
        section_title=section_title,
        subsection_title=subsection_title,
        chapter_title=chapter_title,
        page_snippet=None,
    )


def extract_document_layout(
    pdf_path: str,
    *,
    max_pages: int | None = None,
    max_figures: int = 96,
    chapter_title: str | None = None,
) -> DocumentExtractionResult:
    """
    Extract teaching figures via caption-anchored reconstruction.

    Each Fig./Diagram anchor defines a search region; embedded images, vectors,
    and layout boxes are fused, padded, and rendered as one page region (3×).
    """
    from app.services.image_service.figure_reconstruction import (
        extract_document_by_caption_anchors,
    )

    return extract_document_by_caption_anchors(
        pdf_path,
        max_pages=max_pages,
        max_figures=max_figures,
        chapter_title=chapter_title,
    )


def _extract_document_layout_legacy(
    pdf_path: str,
    *,
    max_pages: int | None = None,
    max_figures: int = 96,
    chapter_title: str | None = None,
) -> DocumentExtractionResult:
    """Legacy image-object-first pipeline (kept for reference / debugging)."""
    import fitz

    doc = fitz.open(pdf_path)
    figures: list[PairedFigure] = []
    page_logs: list[PageExtractionLog] = []
    global_seq = 0
    recurring_keys = scan_recurring_template_keys(pdf_path, max_pages=max_pages)

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

            # --- Stage 2: Multi-source figure discovery ---

            # Source A: embedded raster images
            embedded_raw, tentative_images, bg_removed = collect_images(
                page,
                page_number,
                page_bbox,
                recurring_template_keys=recurring_keys,
            )
            embedded_candidates = [
                FigureCandidate(
                    bbox=img.bbox,
                    source_type="embedded_image",
                    confidence=1.0,
                    image_bytes=blob,
                    xref=img.xref,
                    pixel_width=img.pixel_width,
                    pixel_height=img.pixel_height,
                )
                for img, blob in embedded_raw
            ]

            # Source B: vector drawing regions
            vector_candidates = collect_vector_drawing_candidates(
                page, page_number, page_bbox, text_blocks
            )

            # Source C: layout-detected regions (YOLO or heuristic)
            layout_candidates = detect_layout_regions(
                page, page_number, page_bbox, text_blocks
            )

            # --- Stage 3: Merge candidates ---
            merged = merge_figure_candidates(
                {
                    "embedded": embedded_candidates,
                    "vector": vector_candidates,
                    "layout": layout_candidates,
                },
                page_bbox,
            )

            # Render image bytes for non-embedded candidates that survived merging
            for c in merged:
                if c.image_bytes is None:
                    c.image_bytes = _render_region(page, c.bbox)
                    if c.image_bytes is not None:
                        import fitz as _fitz
                        mat = _fitz.Matrix(_RENDER_MATRIX_SCALE, _RENDER_MATRIX_SCALE)
                        try:
                            pix = page.get_pixmap(
                                matrix=mat,
                                clip=_fitz.Rect(c.bbox.x0, c.bbox.y0, c.bbox.x1, c.bbox.y1),
                                alpha=False,
                            )
                            c.pixel_width = pix.width
                            c.pixel_height = pix.height
                        except Exception:
                            pass

            # Reconstruct images_with_blobs (LayoutImage, bytes) for existing pairing logic
            images_with_blobs: list[tuple[LayoutImage, bytes]] = []
            # Also keep a parallel candidate list to retrieve source_type / confidence later
            active_candidates: list[FigureCandidate] = []
            for idx, c in enumerate(merged):
                if c.image_bytes is None:
                    continue
                lim = LayoutImage(
                    page_number=page_number,
                    image_index_on_page=idx,
                    bbox=c.bbox,
                    xref=c.xref,
                    pixel_width=c.pixel_width,
                    pixel_height=c.pixel_height,
                )
                images_with_blobs.append((lim, c.image_bytes))
                active_candidates.append(c)

            log.backgrounds_removed = bg_removed
            log.images_found = len(images_with_blobs)
            log.captions_found = len(captions)

            sec, subsec = detect_section_titles_from_blocks(text_blocks)

            assignments, orphan_i, orphan_j = pair_images_to_captions(
                images_with_blobs, captions, page_bbox=page_bbox
            )

            # Rescue orphan captions using tentative recurring images.
            # Some PDFs rasterise each page's entire content as one large XObject
            # (same normalised position on every page). The genuine teaching figure
            # lives inside that image slot — pair it with any remaining orphan caption.
            if orphan_j and tentative_images:
                orphan_caps = [captions[j] for j in orphan_j]
                extra_assignments, extra_orphan_i, extra_orphan_j = pair_images_to_captions(
                    tentative_images, orphan_caps, page_bbox=page_bbox
                )
                # Translate extra_assignments back to original caption indices.
                # For each rescued figure, render a focused crop of the page
                # (region above the caption) instead of the full content-area
                # raster — otherwise we'd extract the entire page as one image.
                for ti, oci, dist in extra_assignments:
                    orig_cap_idx = orphan_j[oci]
                    tentative_img, tentative_blob = tentative_images[ti]
                    cap_for_crop = captions[orig_cap_idx]

                    focused_blob = _render_figure_crop_for_caption(
                        page, cap_for_crop.bbox, page_bbox, text_blocks
                    )
                    use_blob = focused_blob if focused_blob else tentative_blob

                    new_img_idx = len(images_with_blobs)
                    images_with_blobs.append((tentative_img, use_blob))
                    assignments.append((new_img_idx, orig_cap_idx, dist))
                    # Track candidate for rescued recurring images
                    active_candidates.append(FigureCandidate(
                        bbox=tentative_img.bbox,
                        source_type="embedded_image",
                        confidence=0.70,
                        image_bytes=use_blob,
                        xref=tentative_img.xref,
                        pixel_width=tentative_img.pixel_width,
                        pixel_height=tentative_img.pixel_height,
                    ))
                # Update orphan_j to only truly-unmatched captions
                matched_oci = {oci for _, oci, _ in extra_assignments}
                orphan_j = [j for pos, j in enumerate(orphan_j) if pos not in matched_oci]
                logger.debug(
                    "[LAYOUT] p%d: rescued %d orphan caption(s) via tentative recurring images",
                    page_number, len(extra_assignments),
                )

            for cap_idx in orphan_j:
                log.orphan_captions.append(captions[cap_idx].figure_number)
            for oi in orphan_i:
                log.orphan_images.append(oi)

            for img_idx, cap_idx, dist in assignments:
                if len(figures) >= max_figures:
                    break
                img, blob = images_with_blobs[img_idx]
                cap = captions[cap_idx]
                pairing_conf = pairing_confidence_from_distance(dist)
                flags: list[str] = []
                if pairing_conf < 0.35:
                    flags.append("low_pairing_confidence")
                if dist > 180:
                    flags.append("suspicious_pairing_distance")
                    log.suspicious_pairs.append(
                        {
                            "figure_number": cap.figure_number,
                            "distance": round(dist, 1),
                            "confidence": round(pairing_conf, 3),
                        }
                    )

                before, after = _text_near_figure(
                    text_blocks, img.bbox, cap.bbox
                )
                ctx = build_figure_context_layout(
                    caption=cap.caption_text,
                    nearby_before=before,
                    nearby_after=after,
                    section_title=sec,
                    subsection_title=subsec,
                    chapter_title=chapter_title,
                )

                from app.services.image_service.textbook_image_extraction import log_figure_extract

                log_figure_extract(
                    page=page_number,
                    figure=cap.figure_number,
                    caption_bbox=cap.bbox.to_list(),
                    image_bbox=img.bbox.to_list(),
                    distance=dist,
                    selected=True,
                )

                log.pairs.append(
                    {
                        "page": page_number,
                        "figure_number": cap.figure_number,
                        "caption": cap.caption_text[:100],
                        "image_bbox": img.bbox.to_list(),
                        "caption_bbox": cap.bbox.to_list(),
                        "pairing_distance": round(dist, 1),
                        "pairing_confidence": round(pairing_conf, 3),
                    }
                )

                # Retrieve candidate metadata (source_type, detection confidence)
                cand = active_candidates[img_idx] if img_idx < len(active_candidates) else None
                src_type = cand.source_type if cand else "embedded_image"
                det_conf = cand.confidence if cand else 1.0
                page_cov = round(min(1.0, img.bbox.area / page_bbox.area), 4) if page_bbox.area > 0 else 0.0

                figures.append(
                    PairedFigure(
                        page_number=page_number,
                        page_index=page_index,
                        sequence=global_seq,
                        figure_number=cap.figure_number,
                        caption=cap.caption_text,
                        figure_context=ctx,
                        image_bbox=img.bbox,
                        caption_bbox=cap.bbox,
                        pairing_distance=dist,
                        pairing_confidence=pairing_conf,
                        image_bytes=blob,
                        nearby_before=before,
                        nearby_after=after,
                        section_title=sec,
                        subsection_title=subsec,
                        validation_flags=flags,
                        source_type=src_type,
                        caption_source="detected",
                        page_coverage=page_cov,
                        confidence=det_conf,
                    )
                )
                global_seq += 1

            # Unpaired images (no figure number, no caption) are intentionally NOT
            # extracted.  Only images that have been paired with a figure label
            # (Fig. N, Diagram N, etc.) are educationally indexable.  Orphan images
            # without labels are decorative publisher assets, icons, or clip-art that
            # add noise to retrieval and must be excluded.

            if log.captions_found or log.images_found:
                page_logs.append(log)

    finally:
        doc.close()

    return DocumentExtractionResult(figures=figures, page_logs=page_logs)


def validation_report_dict(result: DocumentExtractionResult) -> dict[str, Any]:
    """Serialize validation logs for scripts / APIs."""
    return {
        "total_figures": len(result.figures),
        "pages": [
            {
                "page": pl.page_number,
                "backgrounds_removed": pl.backgrounds_removed,
                "images_found": pl.images_found,
                "captions_found": pl.captions_found,
                "pairs": pl.pairs,
                "orphan_captions": pl.orphan_captions,
                "orphan_images": pl.orphan_images,
                "suspicious_pairs": pl.suspicious_pairs,
            }
            for pl in result.page_logs
        ],
    }
