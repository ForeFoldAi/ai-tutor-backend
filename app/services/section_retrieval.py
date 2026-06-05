"""
Heading-aware RAG retrieval for chapter-aware tutor Q&A.

Main section question → all chunks under that heading (all subtopics).
Subsection question → chunks for that subtopic only.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from app.config import CONTEXT_CHAR_BUDGET, RETRIEVAL_K
from app.services.section_heading import (
    HeadingScope,
    SubtopicInfo,
    enrich_chunks_with_section_metadata,
    figure_number_matches,
    is_sidebar_figure_caption,
    normalize_title,
    resolve_heading_scope,
    scope_instruction_for_prompt,
    select_chunks_for_scope,
    subtopics_detail_for_main_section,
)
from app.services.vector_service import fetch_chapter_chunks, retrieve_from_collection

logger = logging.getLogger(__name__)

SECTION_WIDE_K = max(RETRIEVAL_K, 12)
MAIN_SECTION_MAX_CHUNKS = 40
SUBSECTION_MAX_CHUNKS = 12


def retrieve_for_tutor_query(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None,
    k: int | None = None,
) -> tuple[list[Any], HeadingScope, str]:
    """
    Return (docs, scope, extra_prompt_instruction).

    Uses semantic search plus full-chapter scan when chapter_ids are available.
    """
    semantic_k = k if k is not None else RETRIEVAL_K
    wide_k = max(semantic_k, SECTION_WIDE_K)

    semantic = retrieve_from_collection(
        query,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        k=wide_k,
    )

    if not chapter_ids:
        scope = resolve_heading_scope(query, semantic)
        max_c = MAIN_SECTION_MAX_CHUNKS if scope.is_main_section else SUBSECTION_MAX_CHUNKS
        docs = select_chunks_for_scope([], scope, semantic_ranked=semantic, max_chunks=max_c)
        return docs, scope, scope_instruction_for_prompt(scope, chunks=docs)

    chapter_chunks = fetch_chapter_chunks(collection_name, chapter_ids)
    if chapter_chunks:
        enrich_chunks_with_section_metadata(chapter_chunks)
    catalog = chapter_chunks if chapter_chunks else semantic
    scope = resolve_heading_scope(query, catalog)

    max_chunks = MAIN_SECTION_MAX_CHUNKS if scope.is_main_section else (
        SUBSECTION_MAX_CHUNKS if scope.is_subsection else semantic_k
    )
    docs = select_chunks_for_scope(
        chapter_chunks or semantic,
        scope,
        semantic_ranked=semantic,
        max_chunks=max_chunks,
    )

    if scope.matched:
        logger.info(
            "[SECTION] scope=%s matched=%r children=%d docs=%d",
            scope.kind,
            scope.matched.title,
            len(scope.child_headings or []),
            len(docs),
        )

    return docs, scope, scope_instruction_for_prompt(scope, chunks=docs)


_SUBTOPIC_IMAGE_QUERY_HINTS: dict[str, str] = {
    "precipitation": "rain gauge rainfall",
    "temperature": "thermometer temperature scale",
    "atmospheric pressure": "barometer pressure",
    "wind": "anemometer wind vane wind sock",
    "humidity": "hygrometer relative humidity",
}


def _score_image_for_subtopic(im: Any, sub: SubtopicInfo) -> float:
    """Rank one textbook figure for a lettered subtopic (higher = better)."""
    cap = " ".join(
        x for x in (getattr(im, "caption", None), getattr(im, "page_text_snippet", None)) if x
    )
    if is_sidebar_figure_caption(cap):
        return -1.0

    score = 0.0
    page = int(getattr(im, "page_index", 0) or 0) + 1
    fig = getattr(im, "figure_number", None)

    if sub.figure_numbers and figure_number_matches(fig, sub.figure_numbers):
        score += 120.0
    for fn in sub.figure_numbers:
        if fn and fn in (cap or ""):
            score += 90.0

    if sub.pages:
        dist = min(abs(page - p) for p in sub.pages)
        score += max(0.0, 35.0 - dist * 8.0)

    sub_key = sub.normalized_title
    for field in ("subsection_title", "section_title"):
        val = getattr(im, field, None) or ""
        if sub_key and sub_key in normalize_title(val):
            score += 45.0

    hint = _SUBTOPIC_IMAGE_QUERY_HINTS.get(sub_key, "")
    blob = (cap or "").lower()
    if hint:
        for token in hint.split():
            if token in blob:
                score += 12.0
    if sub_key == "humidity":
        if any(w in blob for w in ("wind vane", "anemometer", "tarmac", "wind sock")):
            return -1.0
        if not any(
            w in blob for w in ("hygrometer", "humidity", "humid", "moisture", "vapour", "vapor")
        ):
            return -1.0
    if sub_key == "wind" and "hygrometer" in blob and "anemometer" not in blob:
        score -= 60.0

    if getattr(im, "is_decorative", False):
        score -= 40.0
    score += float(getattr(im, "educational_salience", 0.5) or 0.5) * 10.0
    return score


def _pick_subtopic_image(
    images: list[Any],
    sub: SubtopicInfo,
    *,
    used_figures: set[str] | None = None,
) -> dict | None:
    from app.services.image_service.textbook_image_retrieval import _payload_row

    used = used_figures or set()
    ranked = [(_score_image_for_subtopic(im, sub), im) for im in images]

    def _allowed(s: float, im: Any) -> bool:
        if s < 25.0:
            return False
        fn = (getattr(im, "figure_number", None) or "").strip()
        url = getattr(im, "file_name", None) or ""
        if fn and fn in used:
            return False
        return True

    ranked = [(s, im) for s, im in ranked if _allowed(s, im)]
    if not ranked:
        return None
    ranked.sort(key=lambda x: -x[0])
    score, im = ranked[0]
    row = {**_payload_row(im, score, None), "subtopic": sub.title}
    if getattr(im, "figure_number", None):
        row["figure_number"] = im.figure_number
    return row


def related_images_for_heading_scope(
    scope: HeadingScope,
    *,
    collection_name: str,
    chapter_ids: list[str],
    chapter_names: list[str],
    chapter_single: str,
    query: str,
    retrieved_docs: list[Any],
    conversation_history: list[dict] | None = None,
    fallback_top_n: int = 3,
) -> list[dict]:
    """
    For main-section questions: one figure per textbook subtopic (Fig number + page aware).
    """
    from app.core.database import SessionLocal
    from app.modules.catalog.models import TextbookImage
    from app.services.image_service.textbook_image_retrieval import (
        _list_images,
        _load_uploads,
        related_images_for_query,
    )

    if not scope.is_main_section or not chapter_ids:
        return related_images_for_query(
            collection_name,
            chapter_ids,
            chapter_names,
            chapter_single,
            query,
            retrieved_docs,
            top_n=fallback_top_n,
            conversation_history=conversation_history,
        )

    subtopics = subtopics_detail_for_main_section(scope, retrieved_docs)
    if not subtopics:
        return related_images_for_query(
            collection_name,
            chapter_ids,
            chapter_names,
            chapter_single,
            query,
            retrieved_docs,
            top_n=fallback_top_n,
            conversation_history=conversation_history,
        )

    pool: list[TextbookImage] = []
    try:
        with SessionLocal() as db:
            uploads = _load_uploads(db, chapter_ids)
            if uploads:
                pool = _list_images(db, list(uploads.keys()))
    except Exception as exc:
        logger.debug("Could not load textbook images for section scope: %s", exc)

    seen_figures: set[str] = set()
    seen_urls: set[str] = set()
    merged: list[dict] = []
    parent = scope.matched.title if scope.matched else ""

    for sub in subtopics[:12]:
        row: dict | None = None
        if pool:
            row = _pick_subtopic_image(pool, sub, used_figures=seen_figures)
        if not row:
            hint = _SUBTOPIC_IMAGE_QUERY_HINTS.get(sub.normalized_title, "")
            sub_q = f"{parent} {sub.title} {hint}".strip()
            try:
                hits = related_images_for_query(
                    collection_name,
                    chapter_ids,
                    chapter_names,
                    chapter_single,
                    sub_q,
                    retrieved_docs,
                    top_n=3,
                    conversation_history=conversation_history,
                )
            except Exception as exc:
                logger.debug("Per-subtopic image fetch failed for %r: %s", sub.title, exc)
                hits = []
            for candidate in hits:
                cap = candidate.get("caption") or ""
                if is_sidebar_figure_caption(cap):
                    continue
                fn = (candidate.get("figure_number") or "").strip()
                url = (candidate.get("url") or "").strip()
                if fn and fn in seen_figures:
                    continue
                if url and url in seen_urls:
                    continue
                row = {**candidate, "subtopic": sub.title}
                break
        if not row:
            continue
        url = row.get("url") or ""
        fn = (row.get("figure_number") or "").strip()
        if url and url in seen_urls:
            continue
        if fn and fn in seen_figures:
            continue
        if fn:
            seen_figures.add(fn)
        if url:
            seen_urls.add(url)
        merged.append(row)

    if merged:
        logger.info(
            "[SECTION-IMAGES] main_section subtopics=%d images=%d",
            len(subtopics),
            len(merged),
        )
        return merged

    return related_images_for_query(
        collection_name,
        chapter_ids,
        chapter_names,
        chapter_single,
        query,
        retrieved_docs,
        top_n=fallback_top_n,
        conversation_history=conversation_history,
    )
