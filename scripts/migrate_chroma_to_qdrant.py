#!/usr/bin/env python3
"""
Migrate Chroma collections → Qdrant (copy vectors + payload).

Remaps legacy UUID ``textbook_upload_id`` values to sequential BIGINT ids via
Postgres ``textbook_uploads`` (content_label / id), matching resolve_chroma_upload_ids.

Usage (from ai-tutor-backend, with Chroma volume + Qdrant reachable):

  VECTOR_BACKEND=chroma PYTHONPATH=. python scripts/migrate_chroma_to_qdrant.py
  # then flip: VECTOR_BACKEND=qdrant

Does not delete Chroma data.
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("migrate_chroma_to_qdrant")


def _uuid_to_sequential_map() -> dict[str, str]:
    """Build legacy UUID / any stored id → sequential str(id) from Postgres."""
    from sqlalchemy import select

    from app.core.database import SessionLocal
    from app.modules.catalog.models import TextbookUpload

    mapping: dict[str, str] = {}
    with SessionLocal() as db:
        rows = db.scalars(select(TextbookUpload)).all()
        for row in rows:
            sid = str(row.id)
            mapping[sid] = sid
            # content_label used when Chroma still has UUID payloads
            label = (row.content_label or row.chapter or "").strip()
            if label:
                mapping[f"label:{label}"] = sid
    return mapping


def _remap_upload_id(raw: str | None, by_label: dict[str, str], known: dict[str, str]) -> str:
    if not raw:
        return "0"
    s = str(raw).strip()
    if s in known and known[s] == s:
        return s
    if s.isdigit():
        return s
    # Legacy UUID or unknown — try leave as-is; caller may fix via label
    return s


def migrate() -> int:
    from app.config import CHROMA_PATH, IMAGE_COLLECTION_SUFFIX
    from app.services.vector_service import _sanitize_chroma_env
    from app.services.vector_backend.qdrant_common import (
        IMAGE_VECTOR_SIZE,
        TEXT_VECTOR_SIZE,
        deterministic_text_point_id,
        ensure_collection,
        get_qdrant_client,
        image_point_id,
    )

    _sanitize_chroma_env()
    import chromadb
    from qdrant_client.http import models as qm

    if not os.path.isdir(CHROMA_PATH):
        logger.error("CHROMA_PATH %s missing", CHROMA_PATH)
        return 1

    client_c = chromadb.PersistentClient(path=CHROMA_PATH)
    client_q = get_qdrant_client()
    id_map = _uuid_to_sequential_map()
    label_map = {k[6:]: v for k, v in id_map.items() if k.startswith("label:")}

    suffix = IMAGE_COLLECTION_SUFFIX or "_images"
    total_points = 0

    for coll in client_c.list_collections():
        name = coll.name
        is_image = name.endswith(suffix)
        size = IMAGE_VECTOR_SIZE if is_image else TEXT_VECTOR_SIZE
        ensure_collection(name, vector_size=size, distance="Cosine")

        offset = 0
        batch = 100
        while True:
            raw = coll.get(
                include=["documents", "metadatas", "embeddings"],
                limit=batch,
                offset=offset,
            )
            ids = raw.get("ids") or []
            if not ids:
                break
            docs = raw.get("documents") or []
            metas = raw.get("metadatas") or []
            embs = raw.get("embeddings") or []

            points = []
            for i, pid in enumerate(ids):
                meta = dict(metas[i] or {})
                emb = embs[i]
                if emb is None:
                    continue
                uid_raw = str(meta.get("textbook_upload_id") or "")
                label = str(meta.get("content_label") or "").strip()
                if uid_raw.isdigit():
                    uid = uid_raw
                elif label and label in label_map:
                    uid = label_map[label]
                    logger.info("Remap %s -> %s via label %r in %s", uid_raw, uid, label, name)
                else:
                    uid = _remap_upload_id(uid_raw, label_map, id_map)
                meta["textbook_upload_id"] = uid

                if is_image:
                    image_id = str(meta.get("image_id") or pid)
                    meta["image_id"] = image_id
                    if not str(image_id).isdigit():
                        logger.warning("Skipping non-sequential image_id %s in %s", image_id, name)
                        continue
                    points.append(
                        qm.PointStruct(
                            id=image_point_id(image_id),
                            vector=list(emb),
                            payload=meta,
                        )
                    )
                else:
                    text = docs[i] if i < len(docs) else ""
                    page = meta.get("page", 0)
                    qid = deterministic_text_point_id(uid, page, offset + i, text or "")
                    payload = {**meta, "page_content": text or ""}
                    points.append(qm.PointStruct(id=qid, vector=list(emb), payload=payload))

            if points:
                client_q.upsert(collection_name=name, points=points)
                total_points += len(points)
                logger.info("%s: upserted %d (offset %d)", name, len(points), offset)

            offset += len(ids)
            if len(ids) < batch:
                break

    logger.info("Done. Migrated %d points into Qdrant.", total_points)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(migrate())
