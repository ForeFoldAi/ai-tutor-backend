"""
Layout-aware PDF figure extraction (PyMuPDF).

General-purpose pipeline for textbook PDFs — no hardcoded figure numbers or publishers.

Stages:
  1. Page analysis — image rects, text blocks, caption markers with bboxes
  2. Background rejection — full-page plates never enter pairing
  3. Caption detection — Fig./Figure markers with layout coordinates
  4. Spatial pairing — nearest caption per image (not index order)
  5. Figure context — paragraphs above/below from text blocks
  6. Validation logs — orphans, suspicious pairs, background removals
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
_MAX_PAIR_DISTANCE = 280.0  # page points; tune for A4/Letter
_RENDER_MATRIX_SCALE = 2.0
_PAGE_BLEED_Y0 = -15.0
_RECURRING_TEMPLATE_MIN_PAGES = 2
_LARGE_IMAGE_DISTANCE_PENALTY = 140.0
_BELOW_CAPTION_DISTANCE_PENALTY = 220.0


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
        if inter / page_area > 0.85:
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
    Edge-to-edge vertical distance (captions usually just below figures).

    Uses the gap between image bottom and caption top (or caption bottom to
    image top) rather than center-to-center, so large teaching diagrams that
    sit immediately above their label are not penalised by their own height.
    """
    # Vertical component: edge-to-edge gap
    if caption_bbox.y0 >= image_bbox.y1:
        # Caption is below image (normal textbook layout)
        dy = caption_bbox.y0 - image_bbox.y1
    elif image_bbox.y0 >= caption_bbox.y1:
        # Caption is above image (e.g. figure title at top)
        dy = image_bbox.y0 - caption_bbox.y1
    else:
        # Overlapping vertically — use center distance as fallback
        dy = abs(image_bbox.center[1] - caption_bbox.center[1])

    # Horizontal component: center-to-center (captions are usually aligned with figure)
    dx = abs(image_bbox.center[0] - caption_bbox.center[0])
    dist = dy + 0.35 * dx

    # Light area penalty for very large images to reduce spurious long-range pairings,
    # but cap it so a closely-placed caption always wins over a far-away one.
    if page_bbox and page_bbox.area > 0:
        area_frac = image_bbox.area / page_bbox.area
        dist += min(60.0, _LARGE_IMAGE_DISTANCE_PENALTY * area_frac)

    # Penalty when image is drawn BELOW its caption (rare, usually wrong pairing)
    if image_bbox.y0 > caption_bbox.y0 + 12:
        dist += _BELOW_CAPTION_DISTANCE_PENALTY
    elif image_bbox.center[1] > caption_bbox.center[1] + 20:
        dist += _BELOW_CAPTION_DISTANCE_PENALTY * 0.5

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

    for block in text_blocks:
        text = block.text.strip()
        if not text:
            continue
        for m in _FIG_MARKER_RE.finditer(text):
            fig_num = m.group(2).strip()

            # Skip inline cross-references like "(Fig. 2.6)" — these are parenthesised
            # references inside body text, not standalone figure labels.  The actual
            # label ("Fig. 2.6. Rain gauge") appears as its own text block or at the
            # start of a line and is NOT preceded by "(".
            start = m.start()
            if start > 0 and text[start - 1] == "(":
                continue

            # Title: text after marker until stop pattern or 200 chars
            tail = text[m.end() :]
            tail = _CAPTION_STOP_RE.split(tail, maxsplit=1)[0].strip(" .:;-")
            if len(tail) > 220:
                tail = tail[:220].rsplit(" ", 1)[0]
            cap_text = _format_caption(m.group(1), fig_num, tail)
            # Caption bbox: use block bbox (line-level precision when blocks are lines)
            captions.append(
                LayoutCaption(
                    page_number=page_number,
                    caption_index_on_page=cap_idx,
                    figure_number=fig_num,
                    caption_text=cap_text,
                    bbox=block.bbox,
                )
            )
            cap_idx += 1

    # Deduplicate: same figure number can appear multiple times (inline refs, repeated
    # labels).  Prefer the caption with the LONGEST descriptive title — "Fig. 2.6. Rain
    # gauge" wins over a bare "Fig. 2.6" or one whose tail starts with ")".
    def _title_len(c: LayoutCaption) -> int:
        idx = c.caption_text.find(c.figure_number)
        if idx == -1:
            return 0
        tail = c.caption_text[idx + len(c.figure_number):].strip(" .:;-()")
        return len(tail)

    best: dict[str, LayoutCaption] = {}
    for cap in sorted(captions, key=lambda c: (c.bbox.y0, c.bbox.x0)):
        if cap.figure_number not in best or _title_len(cap) > _title_len(best[cap.figure_number]):
            best[cap.figure_number] = cap

    deduped = sorted(best.values(), key=lambda c: (c.bbox.y0, c.bbox.x0))
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
                if w * h < 2800:
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
    """Heading detection from layout blocks (position + shape)."""
    section: str | None = None
    subsection: str | None = None
    _NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\s+([A-Z][A-Za-z0-9 ,\-/]{3,70})$")
    _CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z\s\-/]{5,70}$")

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
    return section, subsection


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
    Extract all teaching figures from a PDF using layout-aware pairing.
    """
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
            images_with_blobs, tentative_images, bg_removed = collect_images(
                page,
                page_number,
                page_bbox,
                recurring_template_keys=recurring_keys,
            )

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
                # Translate extra_assignments back to original caption indices
                for ti, oci, dist in extra_assignments:
                    orig_cap_idx = orphan_j[oci]
                    # Append the tentative image to images_with_blobs
                    new_img_idx = len(images_with_blobs)
                    images_with_blobs.append(tentative_images[ti])
                    assignments.append((new_img_idx, orig_cap_idx, dist))
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
                conf = pairing_confidence_from_distance(dist)
                flags: list[str] = []
                if conf < 0.35:
                    flags.append("low_pairing_confidence")
                if dist > 180:
                    flags.append("suspicious_pairing_distance")
                    log.suspicious_pairs.append(
                        {
                            "figure_number": cap.figure_number,
                            "distance": round(dist, 1),
                            "confidence": round(conf, 3),
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
                        "pairing_confidence": round(conf, 3),
                    }
                )

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
                        pairing_confidence=conf,
                        image_bytes=blob,
                        nearby_before=before,
                        nearby_after=after,
                        section_title=sec,
                        subsection_title=subsec,
                        validation_flags=flags,
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
