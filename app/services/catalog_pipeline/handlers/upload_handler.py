"""Resilient upload handler for standalone PDF/Word documents."""

from __future__ import annotations

import logging
import os
from typing import Any

from app.core.database import SessionLocal
from app.modules.catalog.models import ProcessingStatusEnum, TextbookUpload
from app.services.catalog_pipeline.constants import EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


def _local_path_for_upload(file_path: str | None) -> str:
    from app.services.image_service.storage_backend import materialize_textbook_file

    return materialize_textbook_file(file_path or "")


def handle_upload_processing(upload_id: int) -> dict[str, Any]:
    """
    Process standalone upload: text extraction, chunking, batched vector indexing, and image extraction.
    Guarantees atomic status updates and clean database connection lifecycle.
    """
    db = SessionLocal()
    try:
        row = db.get(TextbookUpload, upload_id)
        if not row or not row.file_path:
            logger.warning("Upload %s or file_path missing; skipping processing", upload_id)
            return {"status": "skipped", "reason": "not_found"}

        row.chunk_status = ProcessingStatusEnum.PROCESSING
        row.embedding_status = ProcessingStatusEnum.PROCESSING
        db.commit()

        from app.services.document_service import process_document
        from app.services.vector_service import add_documents_to_store, delete_collection_docs

        local_path = _local_path_for_upload(row.file_path)
        if not local_path or not os.path.isfile(local_path):
            raise FileNotFoundError(f"Source file missing for upload {upload_id}: {row.file_path}")

        extra_meta = {
            "board": str(row.board),
            "class_level": str(row.class_level),
            "subject_name": row.subject_name,
            "textbook_upload_id": str(row.id),
            "content_type": row.content_type or "",
            "content_label": row.content_label or "",
        }

        # Step 1: Extract & Chunk
        chunks = process_document(local_path, extra_metadata=extra_meta)
        total_chunks = len(chunks)
        row.chunk_count = total_chunks
        row.chunk_status = ProcessingStatusEnum.EMBEDDED
        db.commit()

        # Step 2: Vector Store Indexing with Batches and Idempotent Cleanup
        collection = f"{row.board}_{row.class_level}_{row.subject_name}".replace(" ", "_")
        try:
            # Delete any previous chunks for this upload_id to prevent duplicates on retry
            delete_collection_docs(collection_name=collection, where_filter={"textbook_upload_id": str(row.id)})
        except Exception as del_exc:
            logger.warning("Could not delete prior docs for upload %s (non-fatal): %s", upload_id, del_exc)

        total_added = 0
        if total_chunks > 0:
            for idx in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
                batch = chunks[idx : idx + EMBEDDING_BATCH_SIZE]
                added = add_documents_to_store(batch, collection_name=collection)
                total_added += added
                logger.info(
                    "Upload %s: Indexed batch %d-%d (%d added) into %s",
                    upload_id,
                    idx + 1,
                    min(idx + EMBEDDING_BATCH_SIZE, total_chunks),
                    added,
                    collection,
                )

        if total_added > 0 or total_chunks == 0:
            row.embedding_status = ProcessingStatusEnum.EMBEDDED
        else:
            row.embedding_status = ProcessingStatusEnum.FAILED
        row.ocr_status = ProcessingStatusEnum.EMBEDDED
        db.commit()

        # Step 3: Image extraction
        try:
            from app.services.image_service.textbook_image_extraction import replace_all_images_after_reprocess

            replace_all_images_after_reprocess(db, row)
        except Exception as img_exc:
            logger.warning("Image extraction after upload %s (non-fatal): %s", upload_id, img_exc)

        logger.info("Successfully processed upload %s: %d chunks, %d embedded", upload_id, total_chunks, total_added)
        return {
            "status": "completed",
            "upload_id": upload_id,
            "chunks": total_chunks,
            "embedded": total_added,
        }

    except Exception as exc:
        logger.exception("Failed to process upload %s: %s", upload_id, exc)
        try:
            row = db.get(TextbookUpload, upload_id)
            if row:
                row.chunk_status = ProcessingStatusEnum.FAILED
                row.embedding_status = ProcessingStatusEnum.FAILED
                db.commit()
        except Exception as db_err:
            logger.error("Failed to mark upload %s as FAILED: %s", upload_id, db_err)
        raise exc
    finally:
        db.close()
