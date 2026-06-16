"""
Content-kind routing for textbook asset retrieval (figures, tables, formulas).

Default queries search figures only. Table/formula queries route to the
matching asset pool and score structured_content + ML concept_tags.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.modules.catalog.models import TextbookImage
    from app.services.image_service.image_intent_extractor import ImageIntent

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def get_content_kind(im: TextbookImage) -> str:
    kind = getattr(im, "content_kind", None) or ""
    if kind in ("figure", "table", "formula"):
        return kind
    img_type = getattr(im, "image_type", None) or ""
    if img_type in ("table", "formula"):
        return img_type
    return "figure"


def _strip_markup(text: str) -> str:
    t = _HTML_TAG_RE.sub(" ", text or "")
    return re.sub(r"\s{2,}", " ", t).strip()


def asset_retrieval_text(im: TextbookImage) -> str:
    """Aggregate searchable text for tables/formulas and figures."""
    parts: list[str] = []
    for field in (
        im.caption,
        getattr(im, "generated_caption", None),
        getattr(im, "title", None),
        getattr(im, "figure_context", None),
        getattr(im, "page_text_snippet", None),
        getattr(im, "structured_content", None),
        getattr(im, "ocr_text", None),
        getattr(im, "concept_tags", None),
        getattr(im, "semantic_keywords", None),
        getattr(im, "section_title", None),
        im.file_name,
    ):
        if field and str(field).strip():
            parts.append(_strip_markup(str(field)))
    return " ".join(parts).lower()


def resolve_content_kind_pool(
    images: list[TextbookImage],
    intent: ImageIntent,
) -> list[TextbookImage]:
    """
    Restrict candidate pool by query intent.

    Default: figures only (tables/formulas do not compete with diagrams).
    Table/formula requests search those kinds; comparison/data queries may
    include tables alongside figures.
    """
    kinds: list[str] = list(getattr(intent, "preferred_content_kinds", None) or ["figure"])
    if not kinds:
        kinds = ["figure"]
    allowed = set(kinds)
    pool = [im for im in images if get_content_kind(im) in allowed]
    if pool:
        return pool
    if "figure" in allowed:
        return [im for im in images if get_content_kind(im) == "figure"]
    return images


def concept_tag_overlap_score(intent: ImageIntent, im: TextbookImage) -> float:
    """0–100 boost when ML concept_tags overlap query concept tokens."""
    raw = getattr(im, "concept_tags", None) or ""
    if not raw.strip():
        return 0.0
    tag_text = raw.replace("|", " ").replace("_", " ").lower()
    tag_tokens = set(re.findall(r"\b[a-z][a-z]+\b", tag_text))
    if not tag_tokens:
        return 0.0
    query_tokens = getattr(intent, "concept_tokens", frozenset()) or frozenset()
    if not query_tokens:
        query_tokens = set(re.findall(r"\b[a-z][a-z]+\b", (intent.core_concept or intent.query).lower()))
    overlap = query_tokens & tag_tokens
    if not overlap:
        return 0.0
    return min(100.0, 35.0 + len(overlap) * 18.0)


def referenced_asset_matches(intent: ImageIntent, im: TextbookImage) -> bool:
    """True when query cites Table N / Equation N and asset number matches."""
    ref_kind = getattr(intent, "referenced_asset_kind", None)
    ref_num = getattr(intent, "referenced_asset_number", None)
    if not ref_kind or not ref_num:
        return False
    if get_content_kind(im) != ref_kind:
        return False
    asset_num = (getattr(im, "figure_number", None) or "").strip()
    if not asset_num and im.caption:
        if ref_kind == "table":
            m = re.search(r"(?i)\b(?:table|tbl\.?)\s*(\d+(?:\.\d+)*)", im.caption)
            asset_num = m.group(1) if m else ""
        elif ref_kind == "formula":
            m = re.search(r"(?i)\b(?:formula|equation|eq\.?)\s*(\d+(?:\.\d+)*)", im.caption)
            asset_num = m.group(1) if m else ""
    return asset_num == ref_num
