"""
ML-based topic tagging for textbook tables and formulas.

Uses BGE semantic embeddings (BAAI/bge-base-en-v1.5) to match asset text against
a curriculum concept taxonomy, with OCR and structured-content fallbacks.

Populates: concept_tags, educational_tags, semantic_keywords, title,
educational_description, generated_caption, figure_context, educational_role.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from app.services.image_service.caption_generator import (
    CURRICULUM_CONCEPT_TAGS,
    extract_semantic_keywords,
    generate_educational_tags,
)

if TYPE_CHECKING:
    from app.modules.catalog.models import TextbookImage, TextbookUpload

logger = logging.getLogger(__name__)

# Extra concepts tuned for tabular / formula assets
_STRUCTURED_CONCEPT_TAGS: dict[str, list[str]] = {
    **CURRICULUM_CONCEPT_TAGS,
    "data_table": ["table", "column", "row", "data", "comparison", "values", "units", "measurement"],
    "mathematical_equation": ["equation", "formula", "expression", "variable", "solve", "calculate", "derive"],
    "physics_formula": ["force", "energy", "velocity", "acceleration", "power", "work", "density", "pressure"],
    "chemical_equation": ["reaction", "reactant", "product", "balanced", "mole", "stoichiometry", "equation"],
    "statistical_table": ["mean", "median", "frequency", "percentage", "ratio", "distribution", "sample"],
}

_SUBJECT_CONCEPT_BOOST: dict[str, set[str]] = {
    "geography": {
        "weather_climate", "landforms", "water_cycle", "maps_cartography",
        "natural_disasters", "data_table",
    },
    "physics": {
        "motion_forces", "electricity", "magnetism", "heat_transfer", "light_optics",
        "waves_sound", "physics_formula", "mathematical_equation",
    },
    "chemistry": {
        "atomic_structure", "chemical_reactions", "states_of_matter",
        "chemical_equation", "mathematical_equation",
    },
    "biology": {
        "cell_biology", "photosynthesis", "human_body", "plant_biology",
        "ecology", "reproduction", "data_table",
    },
    "mathematics": {
        "geometry", "algebra", "statistics", "number_systems",
        "mathematical_equation", "statistical_table",
    },
    "math": {
        "geometry", "algebra", "statistics", "number_systems",
        "mathematical_equation", "statistical_table",
    },
    "history": {
        "ancient_civilizations", "colonial_period", "independence_movement", "data_table",
    },
}

_MIN_BGE_SIMILARITY = 42.0  # cosine * 100 floor for concept match
_MAX_CONCEPT_TAGS = 5
_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.M)
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,2})\s+([A-Z][A-Za-z0-9 ,\-/]{3,70})$", re.M)
_TABLE_CAPTION_RE = re.compile(r"(?i)\b(?:table|tbl\.?)\s*(\d+(?:\.\d+)*)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")

_prototype_cache: dict[str, Any] | None = None


def _strip_markup(text: str) -> str:
    t = _HTML_TAG_RE.sub(" ", text or "")
    return re.sub(r"\s{2,}", " ", t).strip()


def detect_section_from_page_text(page_text: str) -> str | None:
    """Pick the last numbered or markdown heading on a page."""
    if not page_text:
        return None
    headings: list[tuple[int, str]] = []
    for m in _NUMBERED_HEADING_RE.finditer(page_text):
        headings.append((m.start(), m.group(2).strip()))
    for m in _HEADING_RE.finditer(page_text):
        headings.append((m.start(), m.group(1).strip()))
    if not headings:
        return None
    headings.sort(key=lambda x: x[0])
    return headings[-1][1][:255]


def _concept_prototype_text(concept_id: str, keywords: list[str]) -> str:
    label = concept_id.replace("_", " ")
    kw = ", ".join(keywords[:12])
    return f"Textbook topic {label}: {kw}"


def _load_prototype_embeddings() -> dict[str, Any] | None:
    global _prototype_cache
    if _prototype_cache is not None:
        return _prototype_cache

    from app.services.image_service.figure_context_bge import embed_texts

    names = list(_STRUCTURED_CONCEPT_TAGS.keys())
    texts = [
        _concept_prototype_text(name, _STRUCTURED_CONCEPT_TAGS[name])
        for name in names
    ]
    vectors = embed_texts(texts)
    if not vectors or all(v is None for v in vectors):
        _prototype_cache = {}
        return _prototype_cache

    import numpy as np

    cache: dict[str, Any] = {}
    for name, vec in zip(names, vectors):
        if vec is not None:
            cache[name] = np.asarray(vec, dtype=np.float32)
    _prototype_cache = cache
    logger.info("[ML-TAG] Loaded %d concept prototype embeddings", len(cache))
    return _prototype_cache


def _subject_key(subject: str | None) -> str:
    return re.sub(r"[^a-z]", "", (subject or "").lower())


def _rank_concepts_by_bge(
    context_text: str,
    *,
    subject: str | None = None,
    content_kind: str = "table",
) -> list[tuple[str, float]]:
    from app.services.image_service.figure_context_bge import cosine_100, embed_query

    prototypes = _load_prototype_embeddings()
    if not prototypes or not context_text.strip():
        return []

    qvec = embed_query(context_text[:2500])
    if qvec is None:
        return []

    boost_set = _SUBJECT_CONCEPT_BOOST.get(_subject_key(subject), set())
    kind_boost = {"table": {"data_table", "statistical_table"}, "formula": {"mathematical_equation", "physics_formula", "chemical_equation"}}.get(
        content_kind, set()
    )

    scored: list[tuple[str, float]] = []
    for concept_id, pvec in prototypes.items():
        sim = cosine_100(qvec, pvec)
        if concept_id in boost_set:
            sim = min(100.0, sim * 1.08)
        if concept_id in kind_boost:
            sim = min(100.0, sim * 1.12)
        scored.append((concept_id, sim))

    scored.sort(key=lambda x: -x[1])
    return scored


def _regex_concept_hits(blob: str) -> list[str]:
    low = blob.lower()
    hits: list[str] = []
    for concept_id, keywords in _STRUCTURED_CONCEPT_TAGS.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", low) for kw in keywords):
            hits.append(concept_id)
            if len(hits) >= _MAX_CONCEPT_TAGS:
                break
    return hits


def _merge_concept_tags(
    bge_ranked: list[tuple[str, float]],
    regex_hits: list[str],
    *,
    min_sim: float = _MIN_BGE_SIMILARITY,
) -> list[str]:
    out: list[str] = []
    for concept_id, sim in bge_ranked:
        if sim >= min_sim and concept_id not in out:
            out.append(concept_id)
        if len(out) >= _MAX_CONCEPT_TAGS:
            return out
    for concept_id in regex_hits:
        if concept_id not in out:
            out.append(concept_id)
        if len(out) >= _MAX_CONCEPT_TAGS:
            break
    return out


def _build_context_blob(
    *,
    content_kind: str,
    caption: str | None,
    structured_content: str | None,
    chapter_title: str | None,
    section_title: str | None,
    subject: str | None,
    grade_level: str | None,
    page_markdown: str | None,
    ocr_text: str | None,
    figure_number: str | None,
) -> str:
    parts: list[str] = []
    if chapter_title:
        parts.append(f"Chapter: {chapter_title}")
    if section_title:
        parts.append(f"Section: {section_title}")
    if subject:
        parts.append(f"Subject: {subject}")
    if grade_level:
        parts.append(f"Class: {grade_level}")
    if figure_number:
        label = "Table" if content_kind == "table" else "Formula"
        parts.append(f"{label} {figure_number}")
    if caption:
        parts.append(caption)
    if structured_content:
        parts.append(_strip_markup(structured_content)[:1800])
    if page_markdown:
        parts.append(_strip_markup(page_markdown)[:1200])
    if ocr_text:
        parts.append(ocr_text[:800])
    parts.append(f"Content type: {content_kind}")
    return "\n".join(p for p in parts if p.strip())


def _generated_description(
    *,
    content_kind: str,
    concept_tags: list[str],
    caption: str | None,
    structured_excerpt: str,
    section_title: str | None,
    chapter_title: str | None,
) -> str:
    kind_label = "Table" if content_kind == "table" else "Formula"
    topics = ", ".join(t.replace("_", " ") for t in concept_tags[:3])
    loc = section_title or chapter_title or ""
    lead = f"{kind_label}"
    if caption:
        lead = caption.strip()[:120]
    elif topics:
        lead = f"{kind_label} on {topics}"
    desc_parts = []
    if chapter_title:
        desc_parts.append(f"Chapter: {chapter_title.strip()}")
    if section_title:
        desc_parts.append(f"Section: {section_title.strip()}")
    desc_parts.append(lead)
    if structured_excerpt:
        desc_parts.append(structured_excerpt[:220])
    elif topics:
        desc_parts.append(f"Topics: {topics}")
    return ". ".join(desc_parts)[:400]


def _short_title(
    *,
    content_kind: str,
    figure_number: str | None,
    concept_tags: list[str],
    section_title: str | None,
    caption: str | None,
) -> str:
    kind = "Table" if content_kind == "table" else "Formula"
    num = figure_number or ""
    topic = concept_tags[0].replace("_", " ") if concept_tags else ""
    if caption and len(caption.split()) >= 3:
        base = caption.strip()[:60]
    elif num and topic:
        base = f"{kind} {num} — {topic}"
    elif num:
        base = f"{kind} {num}"
    elif topic:
        base = f"{kind} — {topic}"
    else:
        base = kind
    if section_title and section_title.lower() not in base.lower():
        return f"{section_title}: {base}"[:80]
    return base[:80]


def tag_structured_asset_ml(
    *,
    content_kind: str,
    caption: str | None = None,
    structured_content: str | None = None,
    chapter_title: str | None = None,
    section_title: str | None = None,
    subject: str | None = None,
    grade_level: str | None = None,
    page_markdown: str | None = None,
    figure_number: str | None = None,
    image_bytes: bytes | None = None,
    ocr_text: str | None = None,
) -> dict[str, Any]:
    """
    Run BGE concept matching + keyword tagging for a table or formula asset.

    Returns dict with keys: concept_tags, educational_tags, semantic_keywords,
    title, educational_description, generated_caption, figure_context, ocr_text,
    educational_role, educational_salience.
    """
    effective_ocr = ocr_text
    if not effective_ocr and image_bytes and not (structured_content or "").strip():
        try:
            from app.services.image_service.ocr_service import ocr_image_bytes

            effective_ocr = ocr_image_bytes(image_bytes)
        except Exception as exc:
            logger.debug("[ML-TAG] OCR skipped: %s", exc)

    if not figure_number and caption:
        m = _TABLE_CAPTION_RE.search(caption)
        if m:
            figure_number = m.group(1)

    if not section_title and page_markdown:
        section_title = detect_section_from_page_text(page_markdown)

    context_blob = _build_context_blob(
        content_kind=content_kind,
        caption=caption,
        structured_content=structured_content,
        chapter_title=chapter_title,
        section_title=section_title,
        subject=subject,
        grade_level=grade_level,
        page_markdown=page_markdown,
        ocr_text=effective_ocr,
        figure_number=figure_number,
    )

    bge_ranked = _rank_concepts_by_bge(
        context_blob,
        subject=subject,
        content_kind=content_kind,
    )
    regex_hits = _regex_concept_hits(context_blob)
    concept_tags = _merge_concept_tags(bge_ranked, regex_hits)

    structured_excerpt = _strip_markup(structured_content or "")[:300]
    if not structured_excerpt and effective_ocr:
        structured_excerpt = effective_ocr[:300]

    generated_caption = _generated_description(
        content_kind=content_kind,
        concept_tags=concept_tags,
        caption=caption,
        structured_excerpt=structured_excerpt,
        section_title=section_title,
        chapter_title=chapter_title,
    )

    effective_caption = caption or generated_caption
    semantic_keywords = extract_semantic_keywords(
        effective_caption,
        generated_caption,
        structured_excerpt,
        page_markdown or "",
        section_title,
        chapter_title,
        content_kind,
    )
    if concept_tags:
        extra = "|".join(t.replace("_", " ") for t in concept_tags[:3])
        semantic_keywords = f"{content_kind}|{extra}|{semantic_keywords}"[:512]

    educational_tags = generate_educational_tags(
        effective_caption,
        generated_caption,
        structured_excerpt,
        page_markdown or "",
        content_kind,
        "supporting_example" if concept_tags else "unknown",
        section_title,
    )
    if content_kind not in educational_tags:
        educational_tags = f"{content_kind}|{educational_tags}"[:256]
    for tag in concept_tags[:3]:
        if tag not in educational_tags:
            educational_tags = f"{educational_tags}|{tag}"[:256]

    top_sim = bge_ranked[0][1] if bge_ranked else 0.0
    salience = min(1.0, 0.35 + (top_sim / 100.0) * 0.45 + min(0.2, len(concept_tags) * 0.04))
    role = "primary_concept" if top_sim >= 58.0 and concept_tags else "supporting_example"

    return {
        "concept_tags": concept_tags,
        "educational_tags": educational_tags,
        "semantic_keywords": semantic_keywords,
        "title": _short_title(
            content_kind=content_kind,
            figure_number=figure_number,
            concept_tags=concept_tags,
            section_title=section_title,
            caption=caption,
        ),
        "educational_description": generated_caption,
        "generated_caption": generated_caption if not caption else None,
        "figure_context": context_blob[:4000],
        "ocr_text": effective_ocr,
        "educational_role": role,
        "educational_salience": round(salience, 3),
        "section_title": section_title,
        "bge_top_score": round(top_sim, 1),
    }


def enrich_structured_asset_tags(
    row: TextbookImage,
    *,
    image_bytes: bytes,
    upload: TextbookUpload,
    page_markdown: str = "",
) -> None:
    """Populate ML topic tags on a table/formula TextbookImage row."""
    from app.services.image_service.textbook_image_extraction import compute_content_hash

    row.content_hash = compute_content_hash(image_bytes)
    row.grade_level = str(getattr(upload, "class_level", "") or "")
    row.subject = str(getattr(upload, "subject_name", "") or "")

    tags = tag_structured_asset_ml(
        content_kind=getattr(row, "content_kind", None) or row.image_type or "table",
        caption=row.caption,
        structured_content=getattr(row, "structured_content", None),
        chapter_title=row.chapter_title,
        section_title=row.section_title,
        subject=row.subject,
        grade_level=row.grade_level,
        page_markdown=page_markdown,
        figure_number=row.figure_number,
        image_bytes=image_bytes,
        ocr_text=getattr(row, "ocr_text", None),
    )

    if tags.get("section_title") and not row.section_title:
        row.section_title = tags["section_title"]
    if tags.get("concept_tags"):
        row.concept_tags = "|".join(tags["concept_tags"])[:512]
    if tags.get("educational_tags"):
        row.educational_tags = tags["educational_tags"]
    if tags.get("semantic_keywords"):
        row.semantic_keywords = tags["semantic_keywords"]
    if tags.get("title"):
        row.title = tags["title"]
    if tags.get("educational_description"):
        row.educational_description = tags["educational_description"]
    if tags.get("generated_caption"):
        row.generated_caption = tags["generated_caption"]
    if tags.get("figure_context"):
        row.figure_context = tags["figure_context"]
    if tags.get("ocr_text") and not row.ocr_text:
        row.ocr_text = tags["ocr_text"]
    if tags.get("educational_role"):
        row.educational_role = tags["educational_role"]
    if tags.get("educational_salience") is not None:
        row.educational_salience = float(tags["educational_salience"])

    logger.debug(
        "[ML-TAG] %s %s concepts=%s bge=%.1f",
        row.content_kind,
        row.file_name,
        tags.get("concept_tags"),
        tags.get("bge_top_score", 0),
    )
