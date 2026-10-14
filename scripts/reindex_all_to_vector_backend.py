#!/usr/bin/env python3
"""
Full re-embed into the active VECTOR_BACKEND from Postgres textbook uploads.

Cleaner than migrate_chroma_to_qdrant when you can afford re-embedding.
Uses sequential BIGINT textbook_upload_id in all new payloads.

  PYTHONPATH=. python scripts/reindex_all_to_vector_backend.py
  VECTOR_BACKEND=qdrant PYTHONPATH=. python scripts/reindex_all_to_vector_backend.py
"""

from __future__ import annotations

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("reindex_all_to_vector_backend")


def main() -> int:
    from sqlalchemy import select

    from app.core.database import SessionLocal
    from app.modules.catalog.models import TextbookUpload
    from app.services.document_service import process_document
    from app.services.image_service.storage_backend import materialize_textbook_file
    from app.services.vector_service import add_documents_to_store, delete_collection_docs

    with SessionLocal() as db:
        rows = list(db.scalars(select(TextbookUpload)).all())

    logger.info("Reindexing %d uploads into VECTOR_BACKEND=%s", len(rows), os.environ.get("VECTOR_BACKEND", "chroma"))
    for row in rows:
        path = materialize_textbook_file(row.file_path or "")
        if not path or not os.path.isfile(path):
            logger.warning("Skip upload %s — missing file %s", row.id, row.file_path)
            continue
        collection = f"{row.board}_{row.class_level}_{row.subject_name}".replace(" ", "_")
        try:
            delete_collection_docs(
                collection,
                where_filter={"textbook_upload_id": str(row.id)},
            )
        except Exception:
            pass
        extra_meta = {
            "board": str(row.board),
            "class_level": str(row.class_level),
            "subject_name": row.subject_name,
            "textbook_upload_id": str(row.id),  # sequential BIGINT
            "content_type": row.content_type or "",
            "content_label": row.content_label or "",
        }
        try:
            chunks = process_document(path, extra_metadata=extra_meta)
            n = add_documents_to_store(chunks, collection_name=collection)
            logger.info("upload %s -> %s (%d chunks)", row.id, collection, n)
        except Exception:
            logger.exception("Failed upload %s", row.id)

        try:
            from app.services.image_service.multimodal_image_index import index_upload_images

            with SessionLocal() as db:
                fresh = db.get(TextbookUpload, row.id)
                if fresh:
                    index_upload_images(db, fresh)
        except Exception as exc:
            logger.debug("image reindex skip %s: %s", row.id, exc)

    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
