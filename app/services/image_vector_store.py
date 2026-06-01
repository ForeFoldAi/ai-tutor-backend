"""ChromaDB storage for precomputed CLIP image embeddings (separate from text RAG)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from app.config import CHROMA_PATH, IMAGE_COLLECTION_SUFFIX
from app.services.vector_service import _sanitize_chroma_env

logger = logging.getLogger(__name__)


def image_collection_name(text_collection: str) -> str:
    """Derive image vector collection from the subject text collection name."""
    base = (text_collection or "textbooks").strip()
    suffix = IMAGE_COLLECTION_SUFFIX or "_images"
    if base.endswith(suffix):
        return base
    return f"{base}{suffix}"


def subject_collection_from_upload(board: str, class_level: str, subject_name: str) -> str:
    return f"{board}_{class_level}_{subject_name}".replace(" ", "_")


def _get_client():
    _sanitize_chroma_env()
    import chromadb

    return chromadb.PersistentClient(path=CHROMA_PATH)


def _get_or_create_collection(client, name: str):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def delete_image_vectors_for_upload(collection_name: str, upload_id: str) -> None:
    _sanitize_chroma_env()
    try:
        client = _get_client()
        coll = client.get_collection(collection_name)
        coll.delete(where={"textbook_upload_id": upload_id})
        logger.debug("Deleted image vectors for upload %s in %s", upload_id, collection_name)
    except Exception:
        pass


def upsert_image_vectors(
    collection_name: str,
    *,
    upload_id: str,
    items: list[dict[str, Any]],
) -> int:
    """
    Insert or replace vectors for one upload.

    Each item: ``image_id`` (uuid str), ``embedding`` (np.ndarray), metadata keys.
    """
    if not items:
        return 0
    _sanitize_chroma_env()
    delete_image_vectors_for_upload(collection_name, upload_id)

    client = _get_client()
    coll = _get_or_create_collection(client, collection_name)

    ids: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []

    for it in items:
        emb = it.get("embedding")
        if emb is None:
            continue
        image_id = str(it["image_id"])
        ids.append(image_id)
        embeddings.append(np.asarray(emb, dtype=np.float32).tolist())
        metadatas.append(
            {
                "textbook_upload_id": upload_id,
                "image_id": image_id,
                "page_index": int(it.get("page_index", 0)),
                "file_name": str(it.get("file_name", ""))[:200],
            }
        )

    if not ids:
        return 0

    coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=ids)
    logger.info("Indexed %d image vectors in collection '%s'", len(ids), collection_name)
    return len(ids)


def query_image_vectors(
    collection_name: str,
    query_embedding: np.ndarray,
    *,
    chapter_ids: list[str],
    k: int = 24,
) -> list[dict[str, Any]]:
    """
    Return hits: ``image_id``, ``textbook_upload_id``, ``page_index``, ``file_name``, ``similarity``.
    """
    if not chapter_ids or query_embedding is None:
        return []

    _sanitize_chroma_env()
    try:
        client = _get_client()
        coll = client.get_collection(collection_name)
    except Exception:
        return []

    where: dict[str, Any]
    if len(chapter_ids) == 1:
        where = {"textbook_upload_id": chapter_ids[0]}
    else:
        where = {"textbook_upload_id": {"$in": chapter_ids}}

    try:
        res = coll.query(
            query_embeddings=[np.asarray(query_embedding, dtype=np.float32).tolist()],
            n_results=min(k, 100),
            where=where,
            include=["metadatas", "distances"],
        )
    except Exception as exc:
        logger.warning("Image vector query failed: %s", exc)
        return []

    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out: list[dict[str, Any]] = []
    for meta, dist in zip(metas, dists):
        if not meta:
            continue
        # cosine distance in Chroma: 0 = identical; similarity ≈ 1 - dist
        sim = max(0.0, 1.0 - float(dist))
        out.append(
            {
                "image_id": meta.get("image_id"),
                "textbook_upload_id": meta.get("textbook_upload_id"),
                "page_index": int(meta.get("page_index", 0)),
                "file_name": meta.get("file_name", ""),
                "similarity": sim,
            }
        )
    return out


def get_embeddings_for_images(
    collection_name: str,
    image_ids: list[str],
) -> dict[str, np.ndarray]:
    """
    Fetch stored CLIP embeddings for specific image IDs (for rerank-within-filtered-pool).
    Returns ``{image_id: normalized_vector}``.
    """
    if not image_ids:
        return {}

    _sanitize_chroma_env()
    try:
        client = _get_client()
        coll = client.get_collection(collection_name)
    except Exception:
        return {}

    try:
        res = coll.get(ids=image_ids, include=["embeddings"])
    except Exception as exc:
        logger.debug("get_embeddings_for_images failed: %s", exc)
        return {}

    ids_out = res.get("ids") or []
    embs = res.get("embeddings") or []
    out: dict[str, np.ndarray] = {}
    for iid, emb in zip(ids_out, embs):
        if emb is None:
            continue
        vec = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        out[str(iid)] = vec
    return out


def clip_similarities_for_images(
    collection_name: str,
    query_embedding: np.ndarray,
    image_ids: list[str],
) -> dict[str, float]:
    """Cosine similarity (0-1) between query CLIP vector and each image embedding."""
    if query_embedding is None or not image_ids:
        return {}
    q = np.asarray(query_embedding, dtype=np.float32)
    qn = np.linalg.norm(q)
    if qn < 1e-9:
        return {}
    q = q / qn

    stored = get_embeddings_for_images(collection_name, image_ids)
    sims: dict[str, float] = {}
    for iid, vec in stored.items():
        sims[iid] = float(max(0.0, np.dot(q, vec)))
    return sims
