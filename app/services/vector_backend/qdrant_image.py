"""Qdrant CLIP image vector adapter (512-d cosine)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.services.vector_backend.qdrant_common import (
    IMAGE_VECTOR_SIZE,
    ensure_collection,
    get_qdrant_client,
    image_point_id,
    textbook_upload_filter,
)

logger = logging.getLogger(__name__)


def delete_for_upload(collection_name: str, upload_id: str) -> None:
    from qdrant_client.http import models as qm

    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=collection_name,
            points_selector=qm.FilterSelector(
                filter=textbook_upload_filter([str(upload_id)])
            ),
        )
    except Exception:
        pass


def upsert(
    collection_name: str,
    *,
    upload_id: str,
    items: list[dict[str, Any]],
) -> int:
    if not items:
        return 0
    from qdrant_client.http import models as qm

    ensure_collection(collection_name, vector_size=IMAGE_VECTOR_SIZE, distance="Cosine")
    delete_for_upload(collection_name, upload_id)
    client = get_qdrant_client()
    points = []
    for it in items:
        emb = it.get("embedding")
        if emb is None:
            continue
        image_id = str(it["image_id"])
        payload = {
            "textbook_upload_id": str(upload_id),
            "image_id": image_id,
            "page_index": int(it.get("page_index", 0)),
            "file_name": str(it.get("file_name", ""))[:200],
        }
        points.append(
            qm.PointStruct(
                id=image_point_id(image_id),
                vector=np.asarray(emb, dtype=np.float32).tolist(),
                payload=payload,
            )
        )
    if not points:
        return 0
    client.upsert(collection_name=collection_name, points=points)
    logger.info("Indexed %d image vectors in Qdrant '%s'", len(points), collection_name)
    return len(points)


def query(
    collection_name: str,
    query_embedding: np.ndarray,
    *,
    chapter_ids: list[str],
    k: int = 24,
) -> list[dict[str, Any]]:
    if not chapter_ids or query_embedding is None:
        return []
    client = get_qdrant_client()
    try:
        ensure_collection(collection_name, vector_size=IMAGE_VECTOR_SIZE)
        hits = client.search(
            collection_name=collection_name,
            query_vector=np.asarray(query_embedding, dtype=np.float32).tolist(),
            query_filter=textbook_upload_filter([str(x) for x in chapter_ids]),
            limit=min(k, 100),
            with_payload=True,
        )
    except Exception as exc:
        logger.warning("Qdrant image query failed: %s", exc)
        return []

    out: list[dict[str, Any]] = []
    for hit in hits:
        meta = hit.payload or {}
        # Cosine similarity score from Qdrant is already similarity-like for Cosine distance
        sim = float(hit.score or 0.0)
        out.append(
            {
                "image_id": meta.get("image_id"),
                "textbook_upload_id": meta.get("textbook_upload_id"),
                "page_index": int(meta.get("page_index", 0)),
                "file_name": meta.get("file_name", ""),
                "similarity": max(0.0, sim),
            }
        )
    return out


def get_embeddings(collection_name: str, image_ids: list[str]) -> dict[str, np.ndarray]:
    if not image_ids:
        return {}
    client = get_qdrant_client()
    point_ids = [image_point_id(str(i)) for i in image_ids]
    try:
        records = client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_vectors=True,
            with_payload=True,
        )
    except Exception as exc:
        logger.debug("Qdrant get_embeddings failed: %s", exc)
        return {}

    out: dict[str, np.ndarray] = {}
    for rec in records:
        payload = rec.payload or {}
        iid = str(payload.get("image_id") or rec.id)
        vec_raw = rec.vector
        if isinstance(vec_raw, dict):
            vec_raw = next(iter(vec_raw.values()), None)
        if vec_raw is None:
            continue
        vec = np.asarray(vec_raw, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        out[iid] = vec
    return out
