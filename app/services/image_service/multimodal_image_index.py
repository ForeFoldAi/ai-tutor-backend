"""Index extracted textbook images with CLIP embeddings in ChromaDB."""

from __future__ import annotations

import logging
import os
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.catalog.models import TextbookImage, TextbookUpload
from app.services.image_service.image_vector_store import (
    delete_image_vectors_for_upload,
    image_collection_name,
    subject_collection_from_upload,
    upsert_image_vectors,
)
from app.services.image_service.multimodal_encoder import (
    clip_model_available,
    current_model_name,
    encode_image_file,
)
from app.services.image_service.textbook_image_extraction import IMAGE_ROOT

logger = logging.getLogger(__name__)


def _image_disk_path(upload_id: uuid.UUID, file_name: str) -> str:
    return os.path.join(IMAGE_ROOT, str(upload_id), file_name)


def index_upload_images(db: Session, upload: TextbookUpload) -> int:
    """
    Encode all on-disk images for *upload* and store vectors in the subject image collection.

    Returns number of images successfully indexed.
    """
    if not clip_model_available():
        logger.warning("CLIP unavailable — skip multimodal index for upload %s", upload.id)
        return 0

    images = list(
        db.scalars(
            select(TextbookImage)
            .where(TextbookImage.textbook_upload_id == upload.id)
            .order_by(TextbookImage.page_index, TextbookImage.sequence)
        )
    )
    if not images:
        return 0

    coll = image_collection_name(
        subject_collection_from_upload(
            str(upload.board),
            str(upload.class_level),
            upload.subject_name,
        )
    )
    upload_id = str(upload.id)
    delete_image_vectors_for_upload(coll, upload_id)

    items: list[dict] = []
    model_name = current_model_name()
    indexed = 0

    for im in images:
        path = _image_disk_path(upload.id, im.file_name)
        vec = encode_image_file(path)
        if vec is None:
            im.multimodal_indexed = False
            continue
        items.append(
            {
                "image_id": str(im.id),
                "embedding": vec,
                "page_index": im.page_index,
                "file_name": im.file_name,
            }
        )
        im.multimodal_indexed = True
        im.embedding_model = model_name
        indexed += 1

    if items:
        upsert_image_vectors(coll, upload_id=upload_id, items=items)
    db.commit()
    logger.info("Multimodal index: upload %s → %d/%d images", upload.id, indexed, len(images))
    return indexed


def ensure_multimodal_indexed(db: Session, upload: TextbookUpload) -> int:
    """Index any images for *upload* that are not yet marked ``multimodal_indexed``."""
    pending = list(
        db.scalars(
            select(TextbookImage).where(
                TextbookImage.textbook_upload_id == upload.id,
                TextbookImage.multimodal_indexed.is_(False),
            )
        )
    )
    if not pending:
        return 0
    return index_upload_images(db, upload)


def purge_multimodal_index_for_upload(db: Session, upload: TextbookUpload) -> None:
    coll = image_collection_name(
        subject_collection_from_upload(
            str(upload.board),
            str(upload.class_level),
            upload.subject_name,
        )
    )
    delete_image_vectors_for_upload(coll, str(upload.id))
    for im in db.scalars(select(TextbookImage).where(TextbookImage.textbook_upload_id == upload.id)):
        im.multimodal_indexed = False
        im.embedding_model = None
