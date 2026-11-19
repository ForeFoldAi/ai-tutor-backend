from __future__ import annotations

import logging
import re
from typing import Any

from app.services.lesson_planner.algorithms.rrf import reciprocal_rank_fusion
from app.services.vector_service import retrieve_from_collection

logger = logging.getLogger(__name__)


def _collection_name(board: str | None, grade: str, subject: str) -> str:
    board_part = (board or "CBSE").replace(" ", "_")
    return f"{board_part}_{grade}_{subject}".replace(" ", "_")


def _bm25_rank(query: str, docs: list[Any], top_k: int = 10) -> list[str]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return [str(i) for i in range(min(top_k, len(docs)))]

    tokenized = [re.findall(r"\w+", (d.page_content or "").lower()) for d in docs]
    if not tokenized:
        return []
    bm25 = BM25Okapi(tokenized)
    q_tokens = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [str(i) for i in ranked[:top_k]]


def hybrid_retrieve(
    query: str,
    *,
    board: str | None,
    grade: str,
    subject: str,
    chapter_id: str | None = None,
    k: int = 8,
) -> tuple[str, list[dict[str, Any]]]:
    """Dense + BM25 hybrid retrieval with RRF fusion."""
    collection = _collection_name(board, grade, subject)
    chapter_ids = [chapter_id] if chapter_id else None
    docs = retrieve_from_collection(
        query,
        collection_name=collection,
        chapter_ids=chapter_ids,
        k=max(k, 12),
    )
    if not docs:
        return "", []

    dense_ranking = [str(i) for i in range(min(k, len(docs)))]
    sparse_ranking = _bm25_rank(query, docs, top_k=k)
    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
    selected_indices: list[int] = []
    for doc_id, _score in fused:
        try:
            idx = int(doc_id)
        except ValueError:
            continue
        if 0 <= idx < len(docs) and idx not in selected_indices:
            selected_indices.append(idx)
        if len(selected_indices) >= k:
            break

    chunks: list[dict[str, Any]] = []
    texts: list[str] = []
    for idx in selected_indices:
        doc = docs[idx]
        text = (doc.page_content or "").strip()
        if not text:
            continue
        meta = getattr(doc, "metadata", {}) or {}
        chunks.append({"text": text, "metadata": meta})
        texts.append(text)

    context = "\n\n---\n\n".join(texts)
    return context, chunks
