"""
Heading-aware RAG retrieval for chapter-aware tutor Q&A.

Main section question → all chunks under that heading (all subtopics).
Subsection question → chunks for that subtopic only.
"""

from __future__ import annotations

import logging
import re
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
# Qdrant Cosine score is similarity (higher = better). Relative keep avoids
# discarding useful near-neighbors; floor only drops clearly irrelevant tops.
_RELATIVE_KEEP = 0.50
_WEAK_TOP_FLOOR = 0.28
_BROAD_CONTEXT_CHAR_BUDGET = 11000

_BROAD_ASK_RE = re.compile(
    r"\b("
    r"tell\s+me\s+about|explain(?:\s+(?:this|the|about))?|describe|"
    r"what(?:'s|\s+is)\s+this\s+(?:chapter|topic|unit|lesson)\s+about|"
    r"overview|whole\s+(?:chapter|topic|lesson|unit)|"
    r"entire\s+(?:chapter|topic|lesson|unit)"
    r")\b",
    re.I,
)
_NARROW_FACT_RE = re.compile(
    r"\b("
    r"who\s+(?:was|is|were|are)|when\s+(?:did|was|is)|where\s+(?:is|was|did)|"
    r"which|first\s+battle|what\s+was|what\s+were|define|definition|"
    r"named|invented|date|year"
    r")\b",
    re.I,
)


def query_breadth(query: str) -> str:
    """'narrow' | 'broad' | 'chapter' — retrieval width, not spoken length."""
    q = (query or "").strip()
    if not q:
        return "narrow"
    from app.services.chapter_scope import is_current_lesson_query

    if is_current_lesson_query(q):
        return "chapter"
    if re.search(r"\b(whole|entire|full)\s+(chapter|topic|lesson|unit)\b", q, re.I):
        return "chapter"
    if _BROAD_ASK_RE.search(q) and not (_NARROW_FACT_RE.search(q) and len(q.split()) <= 12):
        return "broad"
    return "narrow"


def filter_weak_semantic_hits(docs: list[Any]) -> list[Any]:
    """Drop clearly irrelevant semantic hits. Docs without scores are kept.

    ponytail: relative-to-top, not a hard BGE cutoff — score distributions
    vary by chapter; raise _WEAK_TOP_FLOOR only if logs show junk still passing.
    """
    scored: list[tuple[Any, float]] = []
    unscored: list[Any] = []
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        if "_retrieval_score" in meta:
            try:
                scored.append((d, float(meta["_retrieval_score"])))
            except (TypeError, ValueError):
                unscored.append(d)
        else:
            unscored.append(d)
    if not scored:
        return docs
    top = max(s for _d, s in scored)
    if top < _WEAK_TOP_FLOOR:
        return []
    keep = [d for d, s in scored if s >= top * _RELATIVE_KEEP]
    return keep + unscored


def _spread_chapter_chunks(chunks: list[Any], n: int) -> list[Any]:
    if not chunks or n <= 0:
        return []
    if len(chunks) <= n:
        return list(chunks)
    step = (len(chunks) - 1) / max(n - 1, 1)
    idxs = sorted({min(len(chunks) - 1, int(round(i * step))) for i in range(n)})
    return [chunks[i] for i in idxs]


def _merge_unique(primary: list[Any], extra: list[Any], cap: int) -> list[Any]:
    seen: set[int] = set()
    out: list[Any] = []
    for doc in primary + extra:
        oid = id(doc)
        if oid in seen:
            continue
        seen.add(oid)
        out.append(doc)
        if len(out) >= cap:
            break
    return out


def voice_context_budget(base: int, query: str) -> int:
    if query_breadth(query) in ("broad", "chapter"):
        return max(base, _BROAD_CONTEXT_CHAR_BUDGET)
    return base


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
    breadth = query_breadth(query)
    wide_k = max(semantic_k, SECTION_WIDE_K)
    if breadth in ("broad", "chapter"):
        wide_k = max(wide_k, 16)

    semantic = retrieve_from_collection(
        query,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        k=wide_k,
    )
    semantic = filter_weak_semantic_hits(semantic)

    if not chapter_ids:
        scope = resolve_heading_scope(query, semantic)
        if breadth in ("broad", "chapter") and scope.kind == "subsection":
            scope = HeadingScope(kind="general")
        max_c = MAIN_SECTION_MAX_CHUNKS if (
            scope.is_main_section or breadth in ("broad", "chapter")
        ) else (SUBSECTION_MAX_CHUNKS if scope.is_subsection else semantic_k)
        docs = select_chunks_for_scope([], scope, semantic_ranked=semantic, max_chunks=max_c)
        logger.info(
            "[RAG] breadth=%s scope=%s docs=%d scores=%s q=%r",
            breadth,
            scope.kind,
            len(docs),
            _score_preview(semantic),
            (query or "")[:80],
        )
        return docs, scope, scope_instruction_for_prompt(scope, chunks=docs)

    chapter_chunks = fetch_chapter_chunks(
        collection_name, chapter_ids, chapter_names=chapter_names
    )
    if chapter_chunks:
        enrich_chunks_with_section_metadata(chapter_chunks)
    catalog = _catalog_for_heading_scope(chapter_ids, chapter_chunks, semantic)
    scope = resolve_heading_scope(query, catalog)
    if breadth in ("broad", "chapter") and scope.kind == "subsection":
        scope = HeadingScope(kind="general")

    if breadth == "chapter" or (breadth == "broad" and scope.is_main_section):
        max_chunks = MAIN_SECTION_MAX_CHUNKS
    elif breadth == "broad":
        max_chunks = max(semantic_k, 12)
    else:
        max_chunks = MAIN_SECTION_MAX_CHUNKS if scope.is_main_section else (
            SUBSECTION_MAX_CHUNKS if scope.is_subsection else semantic_k
        )
    docs = select_chunks_for_scope(
        catalog or semantic,
        scope,
        semantic_ranked=semantic,
        max_chunks=max_chunks,
    )
    if breadth == "chapter" and catalog:
        docs = _merge_unique(
            _spread_chapter_chunks(catalog, min(24, MAIN_SECTION_MAX_CHUNKS)),
            docs,
            MAIN_SECTION_MAX_CHUNKS,
        )

    logger.info(
        "[RAG] breadth=%s scope=%s matched=%r docs=%d scores=%s q=%r",
        breadth,
        scope.kind,
        getattr(scope.matched, "title", None),
        len(docs),
        _score_preview(semantic),
        (query or "")[:80],
    )

    return docs, scope, scope_instruction_for_prompt(scope, chunks=docs)


def _score_preview(docs: list[Any], n: int = 4) -> str:
    out: list[str] = []
    for d in docs[:n]:
        meta = getattr(d, "metadata", None) or {}
        if "_retrieval_score" in meta:
            try:
                out.append(f"{float(meta['_retrieval_score']):.3f}")
            except (TypeError, ValueError):
                continue
    return ",".join(out) if out else "-"


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
