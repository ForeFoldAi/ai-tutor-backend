from __future__ import annotations

import hashlib
import logging
from typing import Any

from cachetools import TTLCache

from app.config import CHROMA_PATH
from app.services.lesson_planner.algorithms.rrf import reciprocal_rank_fusion
from app.services.lesson_planner.retrieval.hybrid import _bm25_rank, _collection_name
from app.services.vector_service import retrieve_from_collection

logger = logging.getLogger(__name__)

# Named Chroma collections for lesson planner RAG
COLLECTION_TEXTBOOK_CHUNKS = "textbook_chunks"
COLLECTION_TEXTBOOK_IMAGES = "textbook_images"
COLLECTION_TEXTBOOK_EXPERIMENTS = "textbook_experiments"
COLLECTION_QUESTION_BANK = "question_bank"

_search_cache: TTLCache = TTLCache(maxsize=256, ttl=300)


def _cache_key(collection: str, query: str, chapter_id: str | None, k: int) -> str:
    raw = f"{collection}|{chapter_id or ''}|{k}|{query.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def resolve_textbook_collection(board: str | None, grade: str, subject: str) -> str:
    """Subject-scoped chunk collection (existing catalog naming)."""
    return _collection_name(board, grade, subject)


def hybrid_search(
    query: str,
    *,
    collection: str,
    chapter_id: str | None = None,
    k: int = 8,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid dense + BM25 search with TTL cache."""
    cache_key = _cache_key(collection, query, chapter_id, k)
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    chapter_ids = [chapter_id] if chapter_id else None
    docs = retrieve_from_collection(
        query,
        collection_name=collection,
        chapter_ids=chapter_ids,
        k=max(k, 12),
    )
    if not docs:
        _search_cache[cache_key] = []
        return []

    dense_ranking = [str(i) for i in range(min(k, len(docs)))]
    sparse_ranking = _bm25_rank(query, docs, top_k=k)
    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])

    results: list[dict[str, Any]] = []
    for doc_id, score in fused:
        try:
            idx = int(doc_id)
        except ValueError:
            continue
        if idx < 0 or idx >= len(docs):
            continue
        doc = docs[idx]
        meta = dict(getattr(doc, "metadata", {}) or {})
        if metadata_filter:
            if not all(meta.get(k) == v for k, v in metadata_filter.items()):
                continue
        text = (doc.page_content or "").strip()
        if text:
            results.append({"text": text, "metadata": meta, "score": score})
        if len(results) >= k:
            break

    _search_cache[cache_key] = results
    return results


def search_question_bank(
    query: str,
    *,
    board: str | None,
    grade: str,
    subject: str,
    chapter_id: str | None = None,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Search dedicated question_bank collection, fallback to subject chunks."""
    bank_collection = f"{COLLECTION_QUESTION_BANK}_{resolve_textbook_collection(board, grade, subject)}"
    results = hybrid_search(query, collection=bank_collection, chapter_id=chapter_id, k=k)
    if results:
        return results
    return hybrid_search(
        query,
        collection=resolve_textbook_collection(board, grade, subject),
        chapter_id=chapter_id,
        k=k,
    )


def chroma_path() -> str:
    return CHROMA_PATH
