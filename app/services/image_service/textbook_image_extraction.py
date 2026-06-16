"""
Extract embedded images from uploaded textbook PDFs and DOCX files.

Stores files under ``uploads/_textbook_images/<upload_id>/{figures,tables,formulas}/`` and persists
``TextbookImage`` rows. Idempotent per upload when rows already exist unless
forced purge is called first (re-processing).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import uuid
from io import BytesIO
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import UPLOADS_DIR
from app.modules.catalog.models import TextbookImage, TextbookUpload

logger = logging.getLogger(__name__)

IMAGE_ROOT = os.path.join(UPLOADS_DIR, "_textbook_images")
FIGURES_SUBDIR = "figures"
TABLES_SUBDIR = "tables"
FORMULAS_SUBDIR = "formulas"
_CONTENT_KIND_SUBDIR = {
    "figure": FIGURES_SUBDIR,
    "table": TABLES_SUBDIR,
    "formula": FORMULAS_SUBDIR,
}
_MAX_IMAGES_PER_UPLOAD = 96
_MIN_PIXELS = 4096  # skip tiny icons / bullets (Stage 9)
_MIN_WH = 64        # minimum pixel dimension (Stage 9)
_PHASH_HAMMING_THRESHOLD = 4  # Hamming distance for near-duplicate detection (configurable)
# Reject only full-page raster plates (not large teaching diagrams).
_MAX_FIGURE_AREA = 12_000_000

# Labels: NCERT (Fig./Figure), ICSE (Plate, Exhibit), state-board (Diagram N, Illustration N, Scheme N).
# Requires a digit after the label to avoid matching prose occurrences of "diagram".
# Figure label regex for DOCX slot parsing and caption normalization.
_FIG_MARKER_RE = re.compile(
    r"(?i)\b(fig\.?|figure|diagram|illustration|plate|exhibit|scheme)\s*[\.\s:]*\s*(\d+(?:\.\d+)*)"
)
_TITLE_STOP_RE = re.compile(
    r"(?i)(?:don['\u2019]?t\s+miss|chapter\s+\d|\.indd\b|reprint|not\s+to\s+be\s+republished)",
)

# ---------------------------------------------------------------------------
# Pedagogy-centric metadata helpers
# ---------------------------------------------------------------------------

_IMAGE_TYPE_PATTERNS: list[tuple[str, str]] = [
    # Fine-grained specific types first (most → least specific)
    (r"\b(automated\s+weather\s+station|aws\b|met\s+station|meteorological\s+station|weather\s+station)\b",
     "weather_station"),
    (r"\b(instrument|sensor|gauge|thermometer|barometer|anemometer|hygrometer|apparatus|equipment|device|"
     r"setup|calibrat|measur)\b", "instrument"),
    (r"\b(satellite|aerial|from\s+above|top.?view|remote\s+sensing)\b", "satellite_image"),
    (r"\b(map|distribution|region|boundary|border|territory|location\s+of)\b", "map"),
    (r"\b(gharial|crocodile|tiger|lion|elephant|leopard|peacock|macaque|cobra|"
     r"bird|animal|species|fauna|wildlife|reptile|insect|fish|deer|bear|yak|camel)\b", "wildlife"),
    (r"\b(graph|chart|bar|pie|line\s+graph|data|statistics|percentage)\b", "chart"),
    (r"\b(step\s+\d|stage\s+\d|phase\s+\d|procedure|sequence|method|flowchart|cycle|"
     r"formation|process\s+of|how\s+it\s+works)\b", "process"),
    (r"\b(students?|children|people\s+performing|activity|exercise|experiment|practis|practis)\b",
     "activity"),
    (r"\b(diagram|cross.?section|structure|illustrat|schematic|labelled)\b", "diagram"),
    (r"\b(fort|palace|temple|mosque|church|monument|heritage|ancient|ruins|museum|historical)\b",
     "historical_photo"),
    (r"\b(mountain|himalaya|peak|glacier|waterfall|falls|desert|sand\s+dune|dunes|"
     r"plain|plateau|valley|lake|river|coast|beach|island|forest|cave|cliff|volcano|"
     r"city|town|village|market|landscape|landform|terrain)\b", "landscape"),
]

# Educational role mapping from image_type (query-independent heuristic)
_EDUCATIONAL_ROLE_BY_TYPE: dict[str, str] = {
    "weather_station": "primary_concept",
    "instrument": "primary_concept",
    "diagram": "primary_concept",
    "process": "primary_concept",
    "map": "supporting_example",
    "chart": "supporting_example",
    "satellite_image": "supporting_example",
    "landscape": "supporting_example",
    "historical_photo": "supporting_example",
    "activity": "supporting_example",
    "wildlife": "sidebar_example",
    "unknown": "sidebar_example",
}

# Patterns that indicate a line is a section/subsection heading
_NUMBERED_HEADING_RE = re.compile(
    r"^(\d+(?:\.\d+){0,2})\s+([A-Z][A-Za-z0-9 ,\-/]{3,70})$"
)
_CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z\s\-/]{5,70}$")
_TITLE_CASE_HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9 \-/,]{4,70}$")

_CAPTION_PREFIX_RE = re.compile(r"(?i)^fig\.?\s*\d+(?:\.\d+)*\s*[:. ]+")
_FIG_ONLY_CAPTION_RE = re.compile(r"(?i)^\s*fig\.?\s*\d+(?:\.\d+)*\s*\.?\s*$")
_CAPTION_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
    "and", "or", "to", "for", "with", "by", "from", "this", "that", "it",
    "its", "as", "be", "been", "being", "have", "has", "had", "not", "no",
})


def classify_image_type(caption: str, snippet: str = "") -> str:
    """Classify a figure into a pedagogic image-type category."""
    text = f"{caption or ''} {snippet or ''}".lower()
    for pattern, img_type in _IMAGE_TYPE_PATTERNS:
        if re.search(pattern, text, re.I):
            return img_type
    return "unknown"


def normalize_caption(caption: str) -> str:
    """Lowercase, strip Fig-number prefix and leading punctuation."""
    if not caption:
        return ""
    raw = caption.strip()
    if _FIG_ONLY_CAPTION_RE.match(raw):
        return ""
    text = _CAPTION_PREFIX_RE.sub("", raw)
    text = re.sub(r"\s+", " ", text.lower()).strip(" .:;-")
    if re.fullmatch(r"[\d.]+", text.replace(" ", "")):
        return ""
    return text[:512]


def compute_educational_salience(caption: str) -> float:
    """Heuristic 0–1 score for how educationally significant a figure is."""
    cap = caption or ""
    if not cap.strip():
        return 0.1
    # Longer descriptive captions are more pedagogically rich
    length_score = min(0.5, len(cap) / 200.0)
    # Has a proper Fig label → printed in textbook deliberately
    label_bonus = 0.2 if re.search(r"\bfig\.?\s*\d", cap, re.I) else 0.0
    # Contains important/observe-type cues common in NCERT
    exam_keywords = {"important", "key", "major", "main", "significant", "note", "observe", "notice", "shows", "depicts"}
    words = set(re.findall(r"\w+", cap.lower()))
    keyword_bonus = 0.15 if words & exam_keywords else 0.0
    return min(1.0, length_score + label_bonus + keyword_bonus)


def classify_educational_role(image_type: str, caption: str) -> str:
    """
    Query-independent educational role classification.

    primary_concept  — this image IS the concept being taught
    supporting_example — context/application image
    sidebar_example  — tangentially related
    decorative       — no pedagogical value (no caption, too short)
    activity         — student activity / experiment
    """
    cap = (caption or "").strip()
    if not cap or len(cap) < 8:
        return "decorative"
    role = _EDUCATIONAL_ROLE_BY_TYPE.get(image_type, "sidebar_example")
    # Activity images always get activity role regardless of type
    if re.search(r"\b(students?|children|activity|experiment|practis)\b", cap, re.I):
        return "activity"
    return role


def detect_section_titles(page_text: str) -> tuple[str | None, str | None]:
    """
    Detect section and subsection headings from PDF page text.

    Returns (section_title, subsection_title) as plain strings, or None.
    Heuristics:
      1. Numbered headings: "2.3 Automated Weather Station"
      2. ALL-CAPS lines: "AUTOMATED WEATHER STATION"
      3. Title-case short lines (≥ 70 % capitalized words, no terminal period)
    """
    section: str | None = None
    subsection: str | None = None

    for raw_line in (page_text or "").splitlines():
        line = raw_line.strip()
        if not line or len(line) > 80 or line.endswith("."):
            continue

        # Numbered heading: "1.2 Title" or "1.2.3 Title"
        m = _NUMBERED_HEADING_RE.match(line)
        if m:
            level = m.group(1).count(".")  # 0=chapter, 1=section, 2=subsection
            title = m.group(2).strip()
            if level == 1 and section is None:
                section = title
            elif level == 2 and subsection is None:
                subsection = title
            continue

        # ALL CAPS heading
        if _CAPS_HEADING_RE.match(line) and len(line) >= 6:
            if section is None:
                section = line.title()
            continue

        # Title-case: ≥ 70 % words start with uppercase, 2-8 words
        words = line.split()
        if 2 <= len(words) <= 8:
            cap_count = sum(1 for w in words if w and w[0].isupper())
            if cap_count / len(words) >= 0.70 and section is None:
                section = line

    return section, subsection


def _upload_image_dir(upload_id: uuid.UUID) -> str:
    return os.path.join(IMAGE_ROOT, str(upload_id))


def _asset_subdir(content_kind: str) -> str:
    return _CONTENT_KIND_SUBDIR.get(content_kind, FIGURES_SUBDIR)


def _upload_asset_dir(upload_id: uuid.UUID, content_kind: str) -> str:
    return os.path.join(_upload_image_dir(upload_id), _asset_subdir(content_kind))


def image_disk_path(upload_id: uuid.UUID, file_name: str) -> str:
    """
    Resolve on-disk path for a stored asset.

    Supports new layout ``<upload_id>/<kind>/file.jpg`` and legacy flat files.
    """
    root = _upload_image_dir(upload_id)
    normalized = file_name.replace("\\", "/").lstrip("/")
    direct = os.path.join(root, normalized)
    if os.path.isfile(direct):
        return direct
    base = os.path.basename(normalized)
    for sub in (FIGURES_SUBDIR, TABLES_SUBDIR, FORMULAS_SUBDIR):
        candidate = os.path.join(root, sub, base)
        if os.path.isfile(candidate):
            return candidate
    return direct


def purge_textbook_images_disk_and_rows(db: Session, upload_id: uuid.UUID) -> None:
    """Remove DB rows and on-disk assets for one upload."""
    upload = db.get(TextbookUpload, upload_id)
    if upload is not None:
        try:
            from app.services.image_service.multimodal_image_index import purge_multimodal_index_for_upload

            purge_multimodal_index_for_upload(db, upload)
        except Exception as exc:
            logger.debug("purge_multimodal_index_for_upload: %s", exc)
    db.execute(delete(TextbookImage).where(TextbookImage.textbook_upload_id == upload_id))
    d = _upload_image_dir(upload_id)
    if os.path.isdir(d):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception as exc:
            logger.warning("Could not remove image dir %s: %s", d, exc)


def _normalize_page_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()


def _format_figure_caption(label: str, number: str, title: str) -> str:
    label = label.strip()
    number = number.strip()
    title = re.sub(r"\s+", " ", (title or "").strip(" .:;"))
    prefix = f"{label} {number}".replace("  ", " ")
    if not title or len(title) < 2:
        return prefix[:500]
    sep = ". " if not prefix.endswith(".") else " "
    return f"{prefix}{sep}{title}"[:500]


def _figure_title_from_match(normalized: str, m: re.Match, next_start: int) -> str:
    title = normalized[m.end():next_start].lstrip(" :.-")
    return _TITLE_STOP_RE.split(title, maxsplit=1)[0].strip(" .:;-")


def parse_figure_slots(page_text: str) -> list[dict[str, str]]:
    """
    Ordered figure slots on a page with caption, number, and surrounding explanation text.
    Subject-agnostic: uses generic Fig./Figure numbering only.
    """
    raw = page_text or ""
    if not raw.strip():
        return []

    markers = list(_FIG_MARKER_RE.finditer(raw))
    if not markers:
        return []

    normalized = _normalize_page_text(raw)
    norm_markers = list(_FIG_MARKER_RE.finditer(normalized))
    if len(norm_markers) != len(markers):
        norm_markers = markers

    seen: set[str] = set()
    slots: list[dict[str, str]] = []
    for i, m in enumerate(markers):
        key = f"{m.group(1).lower()}-{m.group(2)}"
        if key in seen:
            continue
        seen.add(key)

        nm = norm_markers[i]
        next_norm_end = norm_markers[i + 1].start() if i + 1 < len(norm_markers) else len(normalized)
        title = _figure_title_from_match(normalized, nm, next_norm_end)
        cap = _format_figure_caption(m.group(1), m.group(2), title)
        if not cap:
            continue

        seg_start = markers[i - 1].end() if i > 0 else 0
        seg_end = markers[i + 1].start() if i + 1 < len(markers) else len(raw)
        segment = raw[seg_start:seg_end]
        rel_start = m.start() - seg_start
        rel_end = m.end() - seg_start
        before = re.sub(r"\s+", " ", segment[:rel_start]).strip()[:1400]
        after = re.sub(r"\s+", " ", segment[rel_end:]).strip()[:1400]

        slots.append({
            "figure_number": m.group(2).strip(),
            "caption": cap,
            "nearby_text_before": before,
            "nearby_text_after": after,
        })
    return slots


def parse_figure_captions(page_text: str) -> list[str]:
    """Return ordered figure labels + titles found in page or document text."""
    return [s["caption"] for s in parse_figure_slots(page_text)]


def extract_figure_number(caption: str) -> str | None:
    m = _FIG_MARKER_RE.search(caption or "")
    return m.group(2).strip() if m else None


def build_figure_context(
    *,
    caption: str | None,
    nearby_before: str = "",
    nearby_after: str = "",
    section_title: str | None = None,
    subsection_title: str | None = None,
    chapter_title: str | None = None,
    page_snippet: str | None = None,
) -> str:
    """Composite pedagogical context used for BGE figure_context matching."""
    parts: list[str] = []
    if chapter_title:
        parts.append(f"Chapter: {chapter_title.strip()}")
    if section_title:
        parts.append(f"Section: {section_title.strip()}")
    if subsection_title and subsection_title != section_title:
        parts.append(f"Subsection: {subsection_title.strip()}")
    if nearby_before:
        parts.append(nearby_before.strip())
    if caption:
        parts.append(caption.strip())
    if nearby_after:
        parts.append(nearby_after.strip())
    if page_snippet and page_snippet not in " ".join(parts):
        parts.append(page_snippet.strip()[:600])
    return "\n".join(p for p in parts if p)[:4000]


def _image_blob_area(blob: bytes) -> int:
    w, h = _image_blob_dimensions(blob)
    return w * h


def _image_blob_dimensions(blob: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(BytesIO(blob)) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return 0, 0


# Page-space rejection (PDF points). Subject-agnostic layout heuristics.
_BACKGROUND_AREA_FRAC = 0.80
_PAGE_ASPECT_AREA_FRAC = 0.55
_PAGE_ASPECT_TOLERANCE = 0.12
_MARGIN_TOUCH_PT = 48.0
_LARGE_PLATE_AREA_FRAC = 0.35
_LARGE_PLATE_WIDTH_FRAC = 0.72
_LARGE_PLATE_HEIGHT_FRAC = 0.45


def reject_figure_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page_width: float,
    page_height: float,
    *,
    recurring_template: bool = False,
) -> tuple[bool, str | None]:
    """
    Return (rejected, reason) for a PDF image rectangle in page coordinates.

    Rejects full-page / near-full-page background plates and large repeated
  content-area templates — not cropped teaching figures.
    """
    if page_width <= 0 or page_height <= 0:
        return True, "invalid_page"

    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    if w < 1 or h < 1:
        return True, "degenerate_rect"

    page_area = page_width * page_height
    img_area = w * h
    area_frac = img_area / page_area

    if area_frac >= _BACKGROUND_AREA_FRAC:
        return True, "area_gt_70pct"

    page_ratio = page_width / page_height
    img_ratio = w / h if h > 0 else 0.0
    if area_frac >= _PAGE_ASPECT_AREA_FRAC and abs(img_ratio - page_ratio) < _PAGE_ASPECT_TOLERANCE:
        return True, "page_aspect_plate"

    m = _MARGIN_TOUCH_PT
    touches_left = x0 <= m
    touches_top = y0 <= m
    touches_right = x1 >= page_width - m
    touches_bottom = y1 >= page_height - m
    if touches_left and touches_top and touches_right and touches_bottom:
        return True, "touches_all_margins"

    # Large content-area plate — must ALSO touch at least one page edge.
    # Centered teaching diagrams (e.g. rain gauge, cross-section) can be wide/tall
    # yet have clear margins on every side and must not be rejected as backgrounds.
    touches_any = touches_left or touches_top or touches_right or touches_bottom
    if (
        area_frac >= _LARGE_PLATE_AREA_FRAC
        and w >= _LARGE_PLATE_WIDTH_FRAC * page_width
        and h >= _LARGE_PLATE_HEIGHT_FRAC * page_height
        and touches_any
    ):
        return True, "large_content_plate"

    if recurring_template:
        return True, "recurring_page_template"

    return False, None


def log_figure_extract(
    *,
    page: int,
    figure: str | None,
    caption_bbox: list[float] | None,
    image_bbox: list[float] | None,
    distance: float | None,
    selected: bool,
    rejected_reason: str | None = None,
) -> None:
    """Structured debug line for layout extraction."""
    logger.info(
        "[FIGURE-EXTRACT] page=%s figure=%s caption_bbox=%s image_bbox=%s "
        "distance=%s selected=%s rejected_reason=%s",
        page,
        figure or "",
        caption_bbox,
        image_bbox,
        None if distance is None else round(distance, 2),
        selected,
        rejected_reason or "",
    )


def _is_page_background_plate(
    width: int,
    height: int,
    *,
    page_width_pt: float | None = None,
    page_height_pt: float | None = None,
) -> bool:
    """
    Drop full-page InDesign raster backgrounds (e.g. 2480×3508), not teaching figures.
    """
    if width < 1 or height < 1:
        return True
    area = width * height
    if area > 5_000_000 and (width >= 2200 or height >= 3200):
        return True
    if page_width_pt and page_height_pt and page_height_pt > 0:
        rejected, _ = reject_figure_rect(
            0.0, 0.0, float(width), float(height), page_width_pt, page_height_pt
        )
        if rejected and area > 2_800_000:
            return True
        page_ratio = page_width_pt / page_height_pt
        img_ratio = width / height
        if area > 2_800_000 and abs(img_ratio - page_ratio) < _PAGE_ASPECT_TOLERANCE:
            return True
    return False


def compute_content_hash(data: bytes) -> str:
    """SHA-256 hex digest for image deduplication."""
    return hashlib.sha256(data).hexdigest()


def compute_phash(image_bytes: bytes) -> int | None:
    """
    Compute a 64-bit perceptual hash of an image, stored as a signed BigInteger.

    Uses imagehash.phash (DCT-based). Returns a signed int64 suitable for a
    PostgreSQL bigint column, or None if imagehash is not installed or the image
    is corrupt. Hamming distance threshold is configurable via _PHASH_HAMMING_THRESHOLD.
    """
    try:
        import imagehash
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as im:
            h = imagehash.phash(im)
        raw = int(h)
        # Convert unsigned uint64 → signed int64 for PostgreSQL bigint
        if raw >= (1 << 63):
            raw -= (1 << 64)
        return raw
    except Exception:
        return None


def phash_hamming(h1: int, h2: int) -> int:
    """Hamming distance between two signed int64 pHash values."""
    # Mask to 64 bits to handle Python's arbitrary-precision negative ints
    xor = (h1 & 0xFFFFFFFFFFFFFFFF) ^ (h2 & 0xFFFFFFFFFFFFFFFF)
    return bin(xor).count("1")


def _enrich_image_row(
    row: TextbookImage,
    *,
    image_bytes: bytes,
    upload: TextbookUpload,
    nearby_before: str = "",
    nearby_after: str = "",
    page_markdown: str = "",
) -> None:
    """
    Populate production metadata fields on a freshly created TextbookImage row.

    Called for both PDF and DOCX extractions. Uses caption_generator for
    uncaptioned / minimal-label figures, and derives grade_level / subject from
    the parent upload to avoid joins at retrieval time.

    Tables and formulas use BGE-based ML topic tagging (structured_asset_tagger).
    """
    content_kind = getattr(row, "content_kind", None) or "figure"
    if content_kind in ("table", "formula"):
        from app.services.image_service.structured_asset_tagger import enrich_structured_asset_tags

        enrich_structured_asset_tags(
            row,
            image_bytes=image_bytes,
            upload=upload,
            page_markdown=page_markdown,
        )
        return

    from app.services.image_service.caption_generator import (
        generate_contextual_caption,
        extract_semantic_keywords,
        generate_educational_tags,
    )
    from app.services.image_service.figure_context_gates import is_minimal_figure_caption

    # Content hash (deduplication)
    row.content_hash = compute_content_hash(image_bytes)

    # Denormalise upload-level fields so retrieval doesn't need a join
    row.grade_level = str(getattr(upload, "class_level", "") or "")
    row.subject = str(getattr(upload, "subject_name", "") or "")

    # Generated caption for uncaptioned or bare-label figures
    has_real_caption = bool(row.caption) and not is_minimal_figure_caption(row.caption)
    if not has_real_caption:
        gen = generate_contextual_caption(
            figure_number=row.figure_number,
            image_type=row.image_type or "unknown",
            section_title=row.section_title,
            subsection_title=row.subsection_title,
            chapter_title=row.chapter_title,
            nearby_before=nearby_before,
            nearby_after=nearby_after,
        )
        if gen:
            row.generated_caption = gen
            # Use generated caption to improve educational role classification
            if row.educational_role in ("decorative", "unknown"):
                row.educational_role = classify_educational_role(row.image_type or "unknown", gen)
            # Recalculate salience using generated caption
            if row.educational_salience < 0.15:
                row.educational_salience = max(
                    row.educational_salience,
                    compute_educational_salience(gen) * 0.8,
                )

    # Semantic keywords from all available text
    effective_caption = row.caption or row.generated_caption or ""
    row.semantic_keywords = extract_semantic_keywords(
        effective_caption,
        row.generated_caption,
        nearby_before,
        nearby_after,
        row.section_title,
        row.chapter_title,
        row.image_type or "unknown",
    )

    # Educational tags (subject domain + type + role)
    row.educational_tags = generate_educational_tags(
        effective_caption,
        row.generated_caption,
        nearby_before,
        nearby_after,
        row.image_type or "unknown",
        row.educational_role or "unknown",
        row.section_title,
    )

    # Educational title, description, and concept tags (Stage 7)
    from app.services.image_service.caption_generator import generate_educational_title

    edu = generate_educational_title(
        figure_number=row.figure_number,
        image_type=row.image_type or "unknown",
        caption=row.caption,
        section_title=row.section_title,
        subsection_title=row.subsection_title,
        chapter_title=row.chapter_title,
        nearby_before=nearby_before,
        nearby_after=nearby_after,
    )
    if edu.get("short_title"):
        row.title = edu["short_title"]
    if edu.get("description"):
        row.educational_description = edu["description"]
    if edu.get("concept_tags"):
        row.concept_tags = "|".join(edu["concept_tags"])[:512]


def generate_extraction_audit_report(
    rows: list[dict[str, Any]],
    *,
    upload_id: uuid.UUID | None = None,
) -> str:
    """
    Human-readable audit table after extraction (requirement 7).

    Each row should include: figure_number, caption, file_name, image_bbox,
    caption_bbox, distance, extraction_confidence.
    """
    header = (
        "Figure Number | Caption | Image File | Image BBox | Caption BBox | "
        "Distance | Extraction Confidence"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        cap = (r.get("caption") or "")[:80]
        lines.append(
            f"{r.get('figure_number') or '-'} | {cap} | {r.get('file_name') or '-'} | "
            f"{r.get('image_bbox') or '-'} | {r.get('caption_bbox') or '-'} | "
            f"{r.get('distance', '-')} | {r.get('extraction_confidence', '-')}"
        )
    report = "\n".join(lines)
    prefix = f"[FIGURE-EXTRACT-AUDIT] upload={upload_id}\n" if upload_id else "[FIGURE-EXTRACT-AUDIT]\n"
    logger.info("%s%s", prefix, report)
    return report


def _pdf_image_raw_bytes(img) -> bytes | None:
    try:
        raw = getattr(img, "data", None)
        if raw is None and hasattr(img, "image"):
            imobj = img.image
            raw = getattr(imobj, "data", None)
        return raw if raw else None
    except Exception:
        return None


def _valid_pdf_image_blobs_pypdf(page) -> list[tuple[bytes, int]]:
    """
    Embedded figure blobs in PDF content-stream order (top-to-bottom layout proxy).

    Does NOT sort by area — largest blob was often a wrong full-page plate while the
    actual Fig. 2.2 diagram was the next embedded image.
    """
    from app.services.image_service.textbook_image_display import normalize_image_blob

    pw = float(page.mediabox.width) if getattr(page, "mediabox", None) else 612.0
    ph = float(page.mediabox.height) if getattr(page, "mediabox", None) else 792.0
    ranked: list[tuple[bytes, int]] = []
    for img in getattr(page, "images", None) or []:
        raw = _pdf_image_raw_bytes(img)
        if not raw:
            continue
        w, h = _image_blob_dimensions(raw)
        area = w * h
        if area < _MIN_PIXELS or area > _MAX_FIGURE_AREA:
            continue
        if _is_page_background_plate(w, h, page_width_pt=pw, page_height_pt=ph):
            continue
        if not normalize_image_blob(raw):
            continue
        ranked.append((raw, area))
    return ranked


def _rects_overlap_ratio(a: Any, b: Any) -> float:
    """Intersection-over-min-area for PyMuPDF Rect-like objects."""
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    min_area = min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0))
    if min_area <= 0:
        return 0.0
    return inter / min_area


def _valid_pdf_image_blobs_fitz(doc: Any, page_index: int) -> list[tuple[bytes, int]]:
    """
    PyMuPDF: rasterize each figure's on-page bounding box (reading order).

    NCERT PDFs often store a CMYK+SMask shell as a separate XObject; the visible
    diagram only appears when the page is rendered at the figure's rect.
    """
    from app.services.image_service.textbook_image_display import normalize_image_blob

    import fitz

    page = doc[page_index]
    pw, ph = page.rect.width, page.rect.height
    render_matrix = fitz.Matrix(2, 2)
    raw_candidates: list[tuple[Any, bytes, int]] = []

    for info in page.get_images(full=True):
        xref = int(info[0])
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for rect in rects:
            if rect.width < 24 or rect.height < 24:
                continue
            try:
                pix = page.get_pixmap(matrix=render_matrix, clip=rect, alpha=False)
                blob = pix.tobytes("jpeg")
                w, h = pix.width, pix.height
            except Exception:
                continue
            if _is_page_background_plate(w, h, page_width_pt=pw, page_height_pt=ph):
                continue
            area = w * h
            if area < _MIN_PIXELS or area > _MAX_FIGURE_AREA:
                continue
            if not normalize_image_blob(blob):
                continue
            raw_candidates.append((rect, blob, area))

    # When NCERT stacks a large diagram + small duplicate XObject, keep the larger rect.
    raw_candidates.sort(key=lambda c: -(c[0].width * c[0].height))
    selected: list[tuple[Any, bytes, int]] = []
    for rect, blob, area in raw_candidates:
        if any(
            _rects_overlap_ratio(rect, kept[0]) > 0.72
            for kept in selected
        ):
            continue
        selected.append((rect, blob, area))

    selected.sort(key=lambda c: (float(c[0].y0), float(c[0].x0)))
    return [(blob, area) for _, blob, area in selected]


def _valid_pdf_image_blobs(path: str, page_index: int, page: Any) -> list[tuple[bytes, int]]:
    """Prefer PyMuPDF positional extract; fall back to pypdf stream order."""
    try:
        import fitz

        doc = fitz.open(path)
        try:
            blobs = _valid_pdf_image_blobs_fitz(doc, page_index)
            if blobs:
                return blobs
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("PyMuPDF image extract unavailable, using pypdf: %s", exc)
    return _valid_pdf_image_blobs_pypdf(page)


def _page_snippet(page_text: str, figure_captions: list[str]) -> str | None:
    base = _normalize_page_text(page_text)[:880]
    if figure_captions:
        fig_part = " | ".join(figure_captions[:6])[:400]
        combined = f"{fig_part} — {base}" if base else fig_part
        return combined[:900] or None
    return base[:880] or None


def _save_blob(
    upload_id: uuid.UUID,
    page_index: int,
    seq: int,
    blob: bytes,
    *,
    content_kind: str = "figure",
    figure_number: str | None = None,
    preferred_name: str | None = None,
) -> str | None:
    from app.services.image_service.textbook_image_display import normalize_image_blob

    jpeg = normalize_image_blob(blob)
    if not jpeg:
        return None
    if preferred_name:
        name = os.path.basename(preferred_name)
    elif figure_number:
        from app.services.image_service.pdf_extraction_types import figure_number_to_filename

        name = figure_number_to_filename(figure_number, page_index, seq)
    else:
        name = f"p{page_index}_{seq}.jpg"
    subdir = _asset_subdir(content_kind)
    out_dir = _upload_asset_dir(upload_id, content_kind)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(jpeg)
    return f"{subdir}/{name}"


def _ml_asset_filename(asset: Any, page_index: int, seq: int) -> str:
    kind = getattr(asset, "content_kind", "figure")
    fig_num = getattr(asset, "figure_number", None)
    if kind == "table":
        if fig_num:
            return f"table_{fig_num.replace('.', '_')}_p{page_index}.jpg"
        return f"table_p{page_index}_{seq}.jpg"
    if kind == "formula":
        return f"formula_p{page_index}_{seq}.jpg"
    if fig_num:
        from app.services.image_service.pdf_extraction_types import figure_number_to_filename

        return figure_number_to_filename(fig_num, page_index, seq)
    return f"p{page_index}_{seq}.jpg"


def _persist_ml_asset_row(
    db: Session,
    upload: TextbookUpload,
    asset: Any,
    *,
    chapter_title: str | None,
    seen_phashes: dict[int, str],
) -> bool:
    """Persist a table or formula ML asset as a TextbookImage row."""
    from app.services.image_service.pdf_extraction_types import BBox, bbox_to_json

    ph = compute_phash(asset.image_bytes)
    if ph is not None:
        for existing_ph, existing_fname in seen_phashes.items():
            if phash_hamming(ph, existing_ph) <= _PHASH_HAMMING_THRESHOLD:
                logger.debug(
                    "[PHASH-DUP] ML %s skipped (matches %s)",
                    asset.content_kind,
                    existing_fname,
                )
                return False

    fname = _save_blob(
        upload.id,
        asset.page_index,
        asset.sequence,
        asset.image_bytes,
        content_kind=asset.content_kind,
        figure_number=asset.figure_number,
        preferred_name=_ml_asset_filename(asset, asset.page_index, asset.sequence),
    )
    if not fname:
        return False

    if ph is not None:
        seen_phashes[ph] = fname

    cap = asset.caption or ""
    structured = asset.structured_content or ""
    page_md = getattr(asset, "page_markdown", "") or ""
    img_type = asset.content_kind
    section_title = None
    if page_md:
        from app.services.image_service.structured_asset_tagger import detect_section_from_page_text

        section_title = detect_section_from_page_text(page_md)
    fig_ctx = "\n".join(p for p in (chapter_title, cap, structured, page_md[:1500]) if p and str(p).strip())

    bbox_json = None
    if asset.image_bbox:
        bbox_json = bbox_to_json(
            BBox(
                asset.image_bbox[0],
                asset.image_bbox[1],
                asset.image_bbox[2],
                asset.image_bbox[3],
            )
        )

    row = TextbookImage(
        id=uuid.uuid4(),
        textbook_upload_id=upload.id,
        page_index=asset.page_index,
        sequence=asset.sequence,
        file_name=fname,
        caption=cap or None,
        page_text_snippet=_page_snippet(fig_ctx, [cap] if cap else []),
        image_type=img_type,
        caption_normalized=normalize_caption(cap),
        is_decorative=False,
        educational_salience=compute_educational_salience(cap or structured),
        figure_number=asset.figure_number,
        chapter_title=chapter_title,
        section_title=section_title,
        has_caption=bool(cap.strip()),
        figure_context=fig_ctx or None,
        image_bbox=bbox_json,
        source_type=asset.source_type,
        caption_source="detected" if cap else "none",
        phash=ph,
        content_kind=asset.content_kind,
        structured_content=structured or None,
    )
    _enrich_image_row(
        row,
        image_bytes=asset.image_bytes,
        upload=upload,
        page_markdown=page_md,
    )
    db.add(row)
    return True


def extract_pdf_images(db: Session, upload: TextbookUpload) -> int:
    """
    Layout-aware PDF extraction with optional native ML pipeline (tables/formulas).
    """
    from app.services.image_service.figure_context_gates import is_minimal_figure_caption
    from app.services.image_service.pdf_extraction_types import (
        bbox_to_json,
        extract_document_layout,
        validation_report_dict,
    )
    path = upload.file_path or ""
    if not path or not os.path.isfile(path):
        return 0

    chapter_title = (upload.chapter or upload.content_label or "").strip() or None
    result = extract_document_layout(
        path,
        max_figures=_MAX_IMAGES_PER_UPLOAD,
        chapter_title=chapter_title,
    )
    ml_assets: list[Any] = list(getattr(result, "ml_assets", None) or [])

    report = validation_report_dict(result)
    logger.info(
        "[EXTRACT] upload=%s layout figures=%d pages_logged=%d",
        upload.id,
        report["total_figures"],
        len(report["pages"]),
    )
    for pl in report["pages"]:
        if pl.get("orphan_captions") or pl.get("suspicious_pairs"):
            logger.debug(
                "[EXTRACT] page %s orphans_cap=%s suspicious=%s",
                pl["page"],
                pl.get("orphan_captions"),
                pl.get("suspicious_pairs"),
            )

    created = 0
    audit_rows: list[dict[str, Any]] = []
    seen_phashes: dict[int, str] = {}  # phash_int → first filename for dedup logging

    for pf in result.figures:
        # Stage 9: document-level pHash deduplication
        ph = compute_phash(pf.image_bytes)
        if ph is not None:
            is_dup = False
            for existing_ph, existing_fname in seen_phashes.items():
                if phash_hamming(ph, existing_ph) <= _PHASH_HAMMING_THRESHOLD:
                    logger.debug(
                        "[PHASH-DUP] p%d seq=%d matches %s (hamming≤%d) — skipped",
                        pf.page_number, pf.sequence, existing_fname, _PHASH_HAMMING_THRESHOLD,
                    )
                    is_dup = True
                    break
            if is_dup:
                continue

        fname = _save_blob(
            upload.id,
            pf.page_index,
            pf.sequence,
            pf.image_bytes,
            content_kind="figure",
            figure_number=pf.figure_number,
            preferred_name=pf.preferred_file_name,
        )
        if not fname:
            continue

        if ph is not None:
            seen_phashes[ph] = fname

        cap = pf.caption or ""
        fig_ctx = pf.figure_context or ""
        snippet = _page_snippet(fig_ctx, [cap] if cap else [])
        img_type = classify_image_type(cap, fig_ctx)
        has_cap = not is_minimal_figure_caption(cap)
        cap_dist = round(pf.pairing_distance, 2) if pf.pairing_distance >= 0 else None
        conf = round(pf.pairing_confidence, 4) if pf.pairing_confidence >= 0 else None

        log_figure_extract(
            page=pf.page_number,
            figure=pf.figure_number,
            caption_bbox=pf.caption_bbox.to_list() if pf.caption_bbox else None,
            image_bbox=pf.image_bbox.to_list(),
            distance=cap_dist,
            selected=True,
        )

        row = TextbookImage(
            id=uuid.uuid4(),
            textbook_upload_id=upload.id,
            page_index=pf.page_index,
            sequence=pf.sequence,
            file_name=fname,
            caption=cap or None,
            page_text_snippet=snippet,
            image_type=img_type,
            caption_normalized=normalize_caption(cap),
            is_decorative=False,
            educational_salience=compute_educational_salience(cap),
            section_title=pf.section_title,
            subsection_title=pf.subsection_title,
            educational_role=classify_educational_role(img_type, cap),
            figure_number=pf.figure_number,
            chapter_title=chapter_title,
            has_caption=has_cap,
            nearby_text_before_figure=pf.nearby_before or None,
            nearby_text_after_figure=pf.nearby_after or None,
            figure_context=fig_ctx or None,
            image_bbox=bbox_to_json(pf.image_bbox),
            caption_bbox=bbox_to_json(pf.caption_bbox),
            caption_distance=cap_dist,
            pairing_confidence=conf,
            context_bge_indexed=False,
            # Production extraction metadata (Stage 2–3, migration 0016)
            source_type=pf.source_type,
            caption_source=pf.caption_source,
            page_coverage=pf.page_coverage if pf.page_coverage else None,
            phash=ph,
            content_kind="figure",
        )
        _enrich_image_row(
            row,
            image_bytes=pf.image_bytes,
            upload=upload,
            nearby_before=pf.nearby_before or "",
            nearby_after=pf.nearby_after or "",
        )
        db.add(row)
        created += 1
        audit_rows.append({
            "figure_number": pf.figure_number,
            "caption": cap,
            "file_name": fname,
            "image_bbox": pf.image_bbox.to_list(),
            "caption_bbox": pf.caption_bbox.to_list() if pf.caption_bbox else None,
            "distance": cap_dist,
            "extraction_confidence": conf,
        })

    for asset in ml_assets:
        if asset.content_kind not in ("table", "formula"):
            continue
        if created >= _MAX_IMAGES_PER_UPLOAD:
            break
        if _persist_ml_asset_row(
            db,
            upload,
            asset,
            chapter_title=chapter_title,
            seen_phashes=seen_phashes,
        ):
            created += 1

    if audit_rows:
        generate_extraction_audit_report(audit_rows, upload_id=upload.id)

    return created


def extract_docx_images(db: Session, upload: TextbookUpload) -> int:
    from docx import Document
    from docx.oxml.ns import qn

    path = upload.file_path or ""
    if not path or not os.path.isfile(path):
        return 0

    doc = Document(path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    figure_slots = parse_figure_slots(full_text)
    if not figure_slots:
        return 0

    captions = [s["caption"] for s in figure_slots]
    snippet = _page_snippet(full_text, captions)
    chapter_title = (upload.chapter or upload.content_label or "").strip() or None
    ranked_blobs: list[tuple[bytes, int]] = []
    try:
        shapes = list(doc.inline_shapes)
    except Exception:
        shapes = []

    for shape in shapes:
        try:
            blip = shape._inline.graphic.graphicData.pic.blipFill.blip  # type: ignore[attr-defined]
            r_id = blip.get(qn("r:embed"))
            if not r_id:
                continue
            part = doc.part.related_parts.get(r_id)
            if part is None:
                continue
            blob = part.blob
        except Exception:
            continue
        from app.services.image_service.textbook_image_display import normalize_image_blob

        if not normalize_image_blob(blob):
            continue
        ranked_blobs.append((blob, _image_blob_area(blob)))

    ranked_blobs.sort(key=lambda x: x[1], reverse=True)
    take = min(len(figure_slots), len(ranked_blobs), _MAX_IMAGES_PER_UPLOAD)
    sec_title, subsec_title = detect_section_titles(full_text)
    created = 0
    for seq in range(take):
        blob, _ = ranked_blobs[seq]
        fname = _save_blob(upload.id, 0, seq, blob, content_kind="figure")
        if not fname:
            continue
        slot = figure_slots[seq]
        cap = slot["caption"]
        before = slot.get("nearby_text_before", "")
        after = slot.get("nearby_text_after", "")
        fig_ctx = build_figure_context(
            caption=cap,
            nearby_before=before,
            nearby_after=after,
            section_title=sec_title,
            subsection_title=subsec_title,
            chapter_title=chapter_title,
            page_snippet=snippet,
        )
        img_type = classify_image_type(cap, fig_ctx or snippet or "")
        has_cap = len((cap or "").strip()) >= 8
        row = TextbookImage(
            id=uuid.uuid4(),
            textbook_upload_id=upload.id,
            page_index=0,
            sequence=seq,
            file_name=fname,
            caption=cap,
            page_text_snippet=snippet,
            image_type=img_type,
            caption_normalized=normalize_caption(cap),
            is_decorative=False,
            educational_salience=compute_educational_salience(cap),
            section_title=sec_title,
            subsection_title=subsec_title,
            educational_role=classify_educational_role(img_type, cap),
            figure_number=slot.get("figure_number"),
            chapter_title=chapter_title,
            has_caption=has_cap,
            nearby_text_before_figure=before or None,
            nearby_text_after_figure=after or None,
            figure_context=fig_ctx or None,
        )
        _enrich_image_row(
            row,
            image_bytes=blob,
            upload=upload,
            nearby_before=before,
            nearby_after=after,
        )
        db.add(row)
        created += 1

    return created


def image_count_for_upload(db: Session, upload_id: uuid.UUID) -> int:
    return int(
        db.scalar(
            select(func.count()).select_from(TextbookImage).where(TextbookImage.textbook_upload_id == upload_id)
        )
        or 0
    )


def upload_disk_assets_missing(db: Session, upload_id: uuid.UUID, *, sample_limit: int = 12) -> bool:
    """
    True when the upload has image rows in Postgres but none of the sampled
    assets are readable on disk (e.g. DB restored without ``_textbook_images``).
    """
    rows = list(
        db.scalars(
            select(TextbookImage)
            .where(TextbookImage.textbook_upload_id == upload_id)
            .limit(sample_limit)
        )
    )
    if not rows:
        return False

    from app.services.image_service.textbook_image_display import image_has_visible_content

    for im in rows:
        if image_has_visible_content(image_disk_path(im.textbook_upload_id, im.file_name)):
            return False
    return True


def ensure_textbook_images_extracted(db: Session, upload: TextbookUpload) -> int:
    """
    If the upload has no image rows yet, extract from disk file and commit.

    When rows exist but on-disk assets are missing, purge and re-extract once.

    Returns number of new rows created (0 if already extracted or failed).
    """
    if image_count_for_upload(db, upload.id) > 0:
        if upload_disk_assets_missing(db, upload.id):
            logger.warning(
                "Textbook image rows exist but disk assets are missing for upload %s — re-extracting",
                upload.id,
            )
            return reextract_textbook_images(db, upload)
        return 0
    if not upload.file_path or not os.path.isfile(upload.file_path):
        return 0

    ext = os.path.splitext(upload.file_path)[1].lower()
    try:
        if ext == ".pdf":
            n = extract_pdf_images(db, upload)
        elif ext in (".docx", ".doc"):
            n = extract_docx_images(db, upload)
        else:
            return 0
        if n:
            db.commit()
            _index_multimodal_after_extract(db, upload)
            ensure_figure_context_bge_indexed(db, upload)
            _run_ocr_if_needed(db, upload)
        return n
    except Exception as exc:
        logger.warning("Image extraction failed for upload %s: %s", upload.id, exc)
        db.rollback()
        return 0


def _run_ocr_if_needed(db: Session, upload: TextbookUpload) -> None:
    """Run OCR on scanned PDFs after image extraction (best-effort, non-blocking)."""
    try:
        from app.services.image_service.ocr_service import ocr_upload_if_needed

        images = list(
            db.scalars(
                select(TextbookImage).where(TextbookImage.textbook_upload_id == upload.id)
            ).all()
        )
        ocr_upload_if_needed(db, upload, images)
    except Exception as exc:
        logger.warning("OCR post-extract failed for %s: %s", upload.id, exc)


def _index_multimodal_after_extract(db: Session, upload: TextbookUpload) -> None:
    try:
        from app.services.image_service.multimodal_image_index import index_upload_images

        index_upload_images(db, upload)
    except Exception as exc:
        logger.warning("Multimodal image index after extract failed for %s: %s", upload.id, exc)


def ensure_figure_context_bge_indexed(db: Session, upload: TextbookUpload) -> int:
    """Index BGE embeddings for figure_context when missing (backward compat)."""
    try:
        from app.services.image_service.figure_context_bge import index_figure_context_embeddings

        rows = list(
            db.scalars(
                select(TextbookImage).where(TextbookImage.textbook_upload_id == upload.id)
            ).all()
        )
        if not rows:
            return 0
        if all(getattr(r, "context_bge_indexed", False) for r in rows):
            return 0
        return index_figure_context_embeddings(db, upload)
    except Exception as exc:
        logger.warning("figure_context BGE index failed for %s: %s", upload.id, exc)
        return 0


def reextract_textbook_images(db: Session, upload: TextbookUpload) -> int:
    """
    Force layout-aware re-extraction (purge + extract + index).
    Use for migration / repair; ignores existing rows.
    """
    purge_textbook_images_disk_and_rows(db, upload.id)
    db.commit()

    ext = os.path.splitext(upload.file_path or "")[1].lower()
    try:
        if ext == ".pdf":
            n = extract_pdf_images(db, upload)
        elif ext in (".docx", ".doc"):
            n = extract_docx_images(db, upload)
        else:
            return 0
        if n:
            db.commit()
            _index_multimodal_after_extract(db, upload)
            ensure_figure_context_bge_indexed(db, upload)
            _run_ocr_if_needed(db, upload)
        return n
    except Exception as exc:
        logger.warning("reextract failed for %s: %s", upload.id, exc)
        db.rollback()
        return 0


def replace_all_images_after_reprocess(db: Session, upload: TextbookUpload) -> int:
    """Purge prior assets then extract fresh (used after re-embedding)."""
    return reextract_textbook_images(db, upload)
