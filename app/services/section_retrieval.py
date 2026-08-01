"""
Heading-aware RAG retrieval for chapter-aware tutor Q&A.

Main section question → all chunks under that heading (all subtopics).
Subsection question → chunks for that subtopic only.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import RETRIEVAL_K
from app.services.section_heading import (
    HeadingScope,
    SubtopicInfo,
    enrich_chunks_with_section_metadata,
    extract_headings_from_chunks,
    is_sidebar_figure_caption,
    resolve_heading_scope,
    scope_instruction_for_prompt,
    select_chunks_for_scope,
    subtopics_detail_for_main_section,
)
from app.services.subtopic_figure_match import (
    is_panel_subfigure,
    primary_figure_numbers,
    score_image_for_subtopic,
)
from app.services.vector_service import fetch_chapter_chunks, retrieve_from_collection

logger = logging.getLogger(__name__)

SECTION_WIDE_K = max(RETRIEVAL_K, 12)
MAIN_SECTION_MAX_CHUNKS = 40
SUBSECTION_MAX_CHUNKS = 12
_MIN_SUBTOPIC_IMAGE_SCORE = 25.0


def _catalog_for_heading_scope(
    chapter_ids: list[str] | None,
    chapter_chunks: list[Any],
    semantic: list[Any],
) -> list[Any]:
    """
    Prefer PDF text-layer chunks when embedded Chroma bodies are too sparse
    for heading / subtopic detection (common with ML-only PDF extraction).
    """
    base = chapter_chunks if chapter_chunks else semantic
    if not chapter_ids:
        return base

    if extract_headings_from_chunks(base):
        return base

    from app.services.pdf_text_layer import load_pdf_text_chunks_for_uploads

    pdf_chunks = load_pdf_text_chunks_for_uploads(chapter_ids)
    if not pdf_chunks:
        return base

    enrich_chunks_with_section_metadata(pdf_chunks)
    if extract_headings_from_chunks(pdf_chunks):
        logger.info(
            "[SECTION] using PDF text-layer catalog (%d pages) — Chroma chunks lack headings",
            len(pdf_chunks),
        )
        return pdf_chunks

    return base


def retrieve_for_tutor_query(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None,
    chapter_names: list[str] | None = None,
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
        chapter_names=chapter_names,
        k=wide_k,
    )

    if not chapter_ids:
        scope = resolve_heading_scope(query, semantic)
        max_c = MAIN_SECTION_MAX_CHUNKS if scope.is_main_section else SUBSECTION_MAX_CHUNKS
        docs = select_chunks_for_scope([], scope, semantic_ranked=semantic, max_chunks=max_c)
        return docs, scope, scope_instruction_for_prompt(scope, chunks=docs)

    chapter_chunks = fetch_chapter_chunks(
        collection_name, chapter_ids, chapter_names=chapter_names
    )
    if chapter_chunks:
        enrich_chunks_with_section_metadata(chapter_chunks)
    catalog = _catalog_for_heading_scope(chapter_ids, chapter_chunks, semantic)
    scope = resolve_heading_scope(query, catalog)

    max_chunks = MAIN_SECTION_MAX_CHUNKS if scope.is_main_section else (
        SUBSECTION_MAX_CHUNKS if scope.is_subsection else semantic_k
    )
    docs = select_chunks_for_scope(
        catalog or semantic,
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


def _pool_image_for_candidate(pool: list[Any], candidate: dict) -> Any | None:
    fn = (candidate.get("figure_number") or "").strip()
    url = (candidate.get("url") or "").strip()
    for im in pool:
        if fn and (getattr(im, "figure_number", None) or "").strip() == fn:
            return im
        fname = getattr(im, "file_name", None) or ""
        if fname and url.endswith(fname):
            return im
    return None


def _subtopic_image_allowed(im: Any, sub: SubtopicInfo) -> bool:
    return score_image_for_subtopic(im, sub) >= _MIN_SUBTOPIC_IMAGE_SCORE


def _pick_subtopic_image(
    images: list[Any],
    sub: SubtopicInfo,
    *,
    used_figures: set[str] | None = None,
) -> dict | None:
    from app.services.image_service.textbook_image_retrieval import _payload_row

    used = used_figures or set()

    def _row_for_image(im: Any, score: float) -> dict:
        row = {**_payload_row(im, score, None, subtopic=sub.title), "subtopic": sub.title}
        if getattr(im, "figure_number", None):
            row["figure_number"] = im.figure_number
        return row

    # Textbook subtopic blocks list figures in reading order — prefer the first
    # unused primary figure so e.g. Wind keeps 2.9 and Humidity can use 2.10.
    ordered_figs = primary_figure_numbers(sub) or list(sub.figure_numbers or [])
    for fn in ordered_figs:
        if fn in used:
            continue
        im = next(
            (x for x in images if (getattr(x, "figure_number", None) or "").strip() == fn),
            None,
        )
        if not im:
            continue
        score = score_image_for_subtopic(im, sub)
        if score >= _MIN_SUBTOPIC_IMAGE_SCORE:
            return _row_for_image(im, score)

    ranked = [(score_image_for_subtopic(im, sub), im) for im in images]

    def _allowed(s: float, im: Any) -> bool:
        if s < _MIN_SUBTOPIC_IMAGE_SCORE:
            return False
        fn = (getattr(im, "figure_number", None) or "").strip()
        if fn and fn in used:
            return False
        return True

    ranked = [(s, im) for s, im in ranked if _allowed(s, im)]
    if not ranked:
        return None
    ranked.sort(key=lambda x: -x[0])
    score, im = ranked[0]
    return _row_for_image(im, score)


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
        _attach_upload_refs,
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
                _attach_upload_refs(pool, uploads)
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
            sub_q = f"{parent} {sub.title}".strip()
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
                if is_panel_subfigure(fn) and primary_figure_numbers(sub):
                    continue
                pool_im = _pool_image_for_candidate(pool, candidate)
                if pool_im and not _subtopic_image_allowed(pool_im, sub):
                    continue
                url = (candidate.get("url") or "").strip()
                if fn and fn in seen_figures:
                    continue
                if url and url in seen_urls:
                    continue
                row = {**candidate, "subtopic": sub.title}
                if pool_im:
                    from app.services.image_service.pdf_figure_context import resolve_display_caption

                    row["caption"] = resolve_display_caption(pool_im, subtopic=sub.title)
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
