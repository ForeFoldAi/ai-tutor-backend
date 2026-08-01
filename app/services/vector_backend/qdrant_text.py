"""Qdrant text RAG adapter (BGE 768-d). Call via vector_service when VECTOR_BACKEND=qdrant."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from app.services.vector_backend.qdrant_common import (
    TEXT_VECTOR_SIZE,
    deterministic_text_point_id,
    ensure_collection,
    get_qdrant_client,
    textbook_upload_filter,
)

logger = logging.getLogger(__name__)


def _embed_docs(docs: list[Document]) -> list[list[float]] | None:
    from app.services.vector_service import _get_embedding_model

    model = _get_embedding_model()
    if model is None:
        return None
    texts = [d.page_content or "" for d in docs]
    return model.embed_documents(texts)


def _embed_query(query: str) -> list[float] | None:
    from app.services.vector_service import _get_embedding_model

    model = _get_embedding_model()
    if model is None:
        return None
    return model.embed_query(query)


def add_documents(docs: list[Document], *, collection_name: str) -> int:
    if not docs:
        return 0
    from qdrant_client.http import models as qm

    embeddings = _embed_docs(docs)
    if embeddings is None:
        logger.error("Embedding model unavailable — cannot add documents to Qdrant")
        return 0

    ensure_collection(collection_name, vector_size=TEXT_VECTOR_SIZE, distance="Cosine")
    client = get_qdrant_client()

    upload_id = str((docs[0].metadata or {}).get("textbook_upload_id") or "")
    if upload_id:
        try:
            client.delete(
                collection_name=collection_name,
                points_selector=qm.FilterSelector(filter=textbook_upload_filter([upload_id])),
            )
            logger.info(
                "Dedup: removed existing Qdrant vectors for upload_id=%s in '%s'",
                upload_id,
                collection_name,
            )
        except Exception:
            pass

    points = []
    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        meta = dict(doc.metadata or {})
        uid = str(meta.get("textbook_upload_id") or upload_id or "0")
        # Always store sequential BIGINT string ids in payload
        meta["textbook_upload_id"] = uid
        page = meta.get("page", 0)
        pid = deterministic_text_point_id(uid, page, i, doc.page_content or "")
        payload = {**meta, "page_content": doc.page_content or ""}
        points.append(qm.PointStruct(id=pid, vector=list(emb), payload=payload))

    client.upsert(collection_name=collection_name, points=points)
    logger.info("Added %d documents to Qdrant collection '%s'", len(points), collection_name)
    return len(points)


def resolve_upload_ids(
    chapter_ids: list[str] | None,
    chapter_names: list[str] | None,
    *,
    collection_name: str,
) -> list[str] | None:
    """Map sequential ids ↔ legacy UUID payloads via content_label scroll (like Chroma sqlite remap)."""
    if not chapter_ids:
        return chapter_ids
    label_to_uid = _label_to_upload_id(collection_name)
    if not label_to_uid:
        return chapter_ids

    known = set(label_to_uid.values())
    names = [str(n).strip() for n in (chapter_names or []) if n and str(n).strip()]
    resolved: list[str] = []
    for i, raw in enumerate(chapter_ids):
        cid = str(raw).strip()
        if not cid:
            continue
        if cid in known:
            resolved.append(cid)
            continue
        name = names[i] if i < len(names) else ""
        mapped = label_to_uid.get(name) if name else None
        if mapped:
            logger.info(
                "[RETRIEVE] mapped upload id %s -> %s via chapter name %r (qdrant)",
                cid,
                mapped,
                name,
            )
            resolved.append(mapped)
        else:
            resolved.append(cid)
    return resolved


def _label_to_upload_id(collection_name: str) -> dict[str, str]:
    client = get_qdrant_client()
    out: dict[str, str] = {}
    try:
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=["content_label", "textbook_upload_id"],
                with_vectors=False,
            )
            for rec in records:
                payload = rec.payload or {}
                label = payload.get("content_label")
                uid = payload.get("textbook_upload_id")
                if label and uid:
                    out[str(label)] = str(uid)
            if offset is None:
                break
    except Exception as exc:
        logger.debug("Qdrant label lookup failed for %r: %s", collection_name, exc)
    return out


def similarity_search(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None = None,
    chapter_names: list[str] | None = None,
    k: int = 5,
) -> list[Document]:
    filter_ids = resolve_upload_ids(chapter_ids, chapter_names, collection_name=collection_name)
    vec = _embed_query(query)
    if vec is None:
        return []
    client = get_qdrant_client()
    try:
        ensure_collection(collection_name, vector_size=TEXT_VECTOR_SIZE)
        qfilter = textbook_upload_filter(filter_ids) if filter_ids else None
        hits = client.search(
            collection_name=collection_name,
            query_vector=vec,
            query_filter=qfilter,
            limit=k,
            with_payload=True,
        )
    except Exception as exc:
        logger.exception("Qdrant search failed for %s: %s", collection_name, exc)
        return []

    docs: list[Document] = []
    for hit in hits:
        payload = dict(hit.payload or {})
        text = payload.pop("page_content", "") or ""
        docs.append(Document(page_content=text, metadata=payload))
    return docs


def fetch_chapter_chunks(
    collection_name: str,
    chapter_ids: list[str],
    *,
    chapter_names: list[str] | None = None,
    limit: int = 500,
) -> list[Document]:
    if not chapter_ids:
        return []
    filter_ids = resolve_upload_ids(chapter_ids, chapter_names, collection_name=collection_name)
    if not filter_ids:
        return []
    client = get_qdrant_client()
    qfilter = textbook_upload_filter(filter_ids)
    out: list[Document] = []
    try:
        offset = None
        while len(out) < limit:
            records, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=qfilter,
                limit=min(256, limit - len(out)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = dict(rec.payload or {})
                text = payload.pop("page_content", "") or ""
                if text:
                    out.append(Document(page_content=text, metadata=payload))
            if offset is None:
                break
    except Exception as exc:
        logger.warning("Qdrant fetch_chapter_chunks failed: %s", exc)
        return []
    out.sort(
        key=lambda d: (
            int((d.metadata or {}).get("page", 0) or 0),
            (d.metadata or {}).get("section_number") or "",
        )
    )
    return out[:limit]


def delete_docs(collection_name: str, *, where_filter: dict | None = None) -> None:
    from qdrant_client.http import models as qm

    client = get_qdrant_client()
    try:
        if where_filter and "textbook_upload_id" in where_filter:
            raw = where_filter["textbook_upload_id"]
            if isinstance(raw, dict) and "$in" in raw:
                ids = [str(x) for x in raw["$in"]]
            else:
                ids = [str(raw)]
            flt = textbook_upload_filter(ids)
            if flt is None:
                return
            client.delete(
                collection_name=collection_name,
                points_selector=qm.FilterSelector(filter=flt),
            )
        else:
            # Full wipe: drop + recreate is safest across Qdrant versions.
            try:
                from app.config import IMAGE_COLLECTION_SUFFIX
                from app.services.vector_backend.qdrant_common import IMAGE_VECTOR_SIZE

                suffix = IMAGE_COLLECTION_SUFFIX or "_images"
                size = IMAGE_VECTOR_SIZE if collection_name.endswith(suffix) else TEXT_VECTOR_SIZE
                client.delete_collection(collection_name)
                ensure_collection(collection_name, vector_size=size)
            except Exception:
                try:
                    client.delete_collection(collection_name)
                except Exception:
                    pass
    except Exception:
        logger.exception("Failed to delete Qdrant docs from '%s'", collection_name)


def collection_stats() -> dict[str, int]:
    client = get_qdrant_client()
    stats: dict[str, int] = {}
    try:
        for coll in client.get_collections().collections:
            info = client.get_collection(coll.name)
            stats[coll.name] = int(info.points_count or 0)
    except Exception:
        logger.exception("Failed to get Qdrant collection stats")
    return stats
