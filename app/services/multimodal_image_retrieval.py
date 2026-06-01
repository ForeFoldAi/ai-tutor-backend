"""
Multimodal image retrieval — symbolic-first, CLIP-second.

Pipeline:
  1. All chapter images from Postgres
  2. Symbolic hard filters (required terms, geography, role, purity)
  3. CLIP cosine rerank ONLY within filtered pool (not global top-k)
  4. Pedagogy composite score (BGE + section + concept + low-weight CLIP)
  5. MMR diversity → strict top-2 selection

CLIP no longer defines the candidate pool.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.config import (
    EXTRA_IMAGE_SCORE_RATIO,
    MAX_PRIMARY_IMAGES,
    MIN_FINAL_SCORE_MULTIMODAL,
    TOP_RELATED_IMAGES,
)
from app.core.database import SessionLocal
from app.services.image_intent_extractor import ImageIntent, extract_image_intent
from app.services.image_vector_store import clip_similarities_for_images, image_collection_name
from app.services.multimodal_encoder import clip_model_available, encode_text
from app.services.multimodal_image_index import ensure_multimodal_indexed
from app.services.textbook_image_extraction import (
    ensure_figure_context_bge_indexed,
    ensure_textbook_images_extracted,
)
from app.services.textbook_image_retrieval import (
    _chapter_label_match_score,
    _context_excerpt_from_docs,
    _list_images,
    _load_uploads,
    _page_proximity_score,
    _pages_by_upload_from_docs,
    _pedagogy_rank,
    filter_chapter_candidates,
    finalize_related_images,
)

logger = logging.getLogger(__name__)


def related_images_multimodal(
    text_collection_name: str,
    chapter_ids: list[str] | None,
    chapter_names: list[str],
    chapter_single: str,
    query: str,
    retrieved_docs: list[Any],
    *,
    answer_text: str = "",
    top_n: int = TOP_RELATED_IMAGES,
    conversation_context: Any | None = None,
) -> list[dict]:
    if not chapter_ids:
        return []

    hints = [h for h in (chapter_names or []) if h and h.strip()]
    if chapter_single and chapter_single.strip():
        hints.append(chapter_single.strip())

    intent: ImageIntent = extract_image_intent(
        query, retrieved_docs, conversation_context=conversation_context
    )
    context_excerpt = _context_excerpt_from_docs(retrieved_docs)
    pages_by_upload = _pages_by_upload_from_docs(retrieved_docs)
    img_coll = image_collection_name(text_collection_name)
    min_score = MIN_FINAL_SCORE_MULTIMODAL

    with SessionLocal() as db:
        uploads = _load_uploads(db, chapter_ids)
        if not uploads:
            return []

        for u in uploads.values():
            try:
                ensure_textbook_images_extracted(db, u)
                ensure_figure_context_bge_indexed(db, u)
                if clip_model_available():
                    ensure_multimodal_indexed(db, u)
            except Exception as exc:
                logger.debug("ensure images/index: %s", exc)

        all_images = _list_images(db, list(uploads.keys()))
        if not all_images:
            logger.warning("[IMAGES] no textbook_images rows for chapter_ids=%s", chapter_ids)
            return []

        # ── Step 1–5: symbolic filtering on ALL chapter images ─────────────────
        filtered = filter_chapter_candidates(intent, all_images)
        if not filtered:
            logger.info(
                "[MULTIMODAL] 0 symbolic survivors (query=%r)",
                query[:60],
            )
            from app.config import DISABLE_PAGE_PROXIMITY_FALLBACK
            if DISABLE_PAGE_PROXIMITY_FALLBACK:
                return []
            return finalize_related_images(
                [],
                query=query,
                answer_text="",
                max_n=top_n,
                context_excerpt=context_excerpt,
                pages_by_upload=pages_by_upload,
                pool_images=all_images,
                intent=intent,
            )

        cand_images = [im for im, _ in filtered]

        # ── Step 6–7: CLIP rerank inside filtered pool only ───────────────────
        clip_sims: dict[str, float] = {}
        if clip_model_available():
            qvec = encode_text(intent.intent_text)
            if qvec is not None:
                image_ids = [str(im.id) for im in cand_images]
                clip_sims = clip_similarities_for_images(img_coll, qvec, image_ids)
                logger.debug(
                    "[MULTIMODAL] CLIP rerank on %d filtered images (not global top-k)",
                    len(cand_images),
                )

        # ── Step 8–10: pedagogy rank + MMR + strict select ────────────────────
        # Reuse _pedagogy_rank with precomputed clip_sims
        from app.services.textbook_image_retrieval import _pedagogy_rank as rank_fn

        results = rank_fn(
            intent,
            all_images,
            uploads,
            hints,
            pages_by_upload,
            max_n=top_n,
            min_ped_score=min_score,
            clip_sims=clip_sims,
            rag_docs=retrieved_docs,
        )

        if results:
            logger.info(
                "[MULTIMODAL] symbolic-first kept %d (pool=%d filtered=%d, query=%r)",
                len(results), len(all_images), len(filtered), query[:60],
            )
            return results

        # Last resort: legacy fallback with keyword gate (still no LLM answer)
        from app.services.textbook_image_retrieval import _topic_keyword_score

        scored_fb = []
        for im, _ in filtered[:top_n * 2]:
            up = uploads.get(im.textbook_upload_id)
            if not up:
                continue
            topic = _topic_keyword_score(query, "", im)
            prox = _page_proximity_score(
                str(im.textbook_upload_id), im.page_index, pages_by_upload
            )
            ch = _chapter_label_match_score(up.chapter, hints)
            scored_fb.append((prox + topic * 2.0 + ch * 0.2, im, topic, -1.0))

        return finalize_related_images(
            scored_fb,
            query=query,
            answer_text="",
            max_n=top_n,
            context_excerpt=context_excerpt,
            pages_by_upload=pages_by_upload,
            pool_images=all_images,
            intent=intent,
        )
