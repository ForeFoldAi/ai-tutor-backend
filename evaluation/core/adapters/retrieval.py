"""Embedding + retrieval adapters."""

from __future__ import annotations

from typing import Any


def embed_and_store(chunks: list[Any], *, collection_name: str) -> int:
    from app.services.vector_service import add_documents_to_store

    return int(add_documents_to_store(chunks, collection_name=collection_name))


def retrieve(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None = None,
    k: int | None = None,
) -> tuple[list[Any], Any, str]:
    from app.services.section_retrieval import retrieve_for_tutor_query

    kwargs: dict[str, Any] = {
        "collection_name": collection_name,
        "chapter_ids": chapter_ids,
    }
    if k is not None:
        # retrieve_for_tutor_query may not accept k — try and fall back
        try:
            return retrieve_for_tutor_query(query, **kwargs, k=k)  # type: ignore[call-arg]
        except TypeError:
            pass
    return retrieve_for_tutor_query(query, **kwargs)


def docs_to_serializable(docs: list[Any]) -> list[dict[str, Any]]:
    out = []
    for i, d in enumerate(docs or []):
        meta = dict(getattr(d, "metadata", None) or {})
        text = getattr(d, "page_content", "") or ""
        out.append(
            {
                "rank": i + 1,
                "page": meta.get("page"),
                "section_hint": meta.get("section_hint"),
                "textbook_upload_id": meta.get("textbook_upload_id"),
                "preview": text[:400],
                "score": meta.get("score") or meta.get("relevance_score"),
            }
        )
    return out
