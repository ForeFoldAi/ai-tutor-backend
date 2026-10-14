"""Celery task definitions for catalog PDF chunking, embedding, and segmentation."""

from __future__ import annotations

import logging
from typing import Any

from app.core.celery_app import celery_app
from app.services.catalog_pipeline.constants import (
    DEFAULT_RETRY_BACKOFF_SECONDS,
    MAX_INGESTION_RETRIES,
    MAX_RETRY_BACKOFF_SECONDS,
    QUEUE_CATALOG,
    TASK_ANALYZE_TEXTBOOK,
    TASK_PROCESS_CHAPTERS,
    TASK_PROCESS_UPLOAD,
)
from app.services.catalog_pipeline.handlers.chapter_handler import handle_textbook_chapters
from app.services.catalog_pipeline.handlers.structure_handler import handle_textbook_analysis
from app.services.catalog_pipeline.handlers.upload_handler import handle_upload_processing

logger = logging.getLogger(__name__)


@celery_app.task(
    name=TASK_PROCESS_UPLOAD,
    queue=QUEUE_CATALOG,
    acks_late=True,
    bind=True,
    max_retries=MAX_INGESTION_RETRIES,
    default_retry_delay=DEFAULT_RETRY_BACKOFF_SECONDS,
)
def process_upload_task(self, upload_id: int) -> dict[str, Any]:
    """Celery task: Extract, chunk, embed, and index a standalone textbook upload."""
    logger.info("Celery [task_id=%s]: Starting process_upload_task for upload_id=%s", self.request.id, upload_id)
    try:
        return handle_upload_processing(upload_id)
    except Exception as exc:
        logger.exception("process_upload_task failed on attempt %d: %s", self.request.retries + 1, exc)
        if self.request.retries < self.max_retries:
            countdown = min(DEFAULT_RETRY_BACKOFF_SECONDS * (2 ** self.request.retries), MAX_RETRY_BACKOFF_SECONDS)
            raise self.retry(exc=exc, countdown=countdown)
        raise exc


@celery_app.task(
    name=TASK_ANALYZE_TEXTBOOK,
    queue=QUEUE_CATALOG,
    acks_late=True,
    bind=True,
    max_retries=MAX_INGESTION_RETRIES,
    default_retry_delay=DEFAULT_RETRY_BACKOFF_SECONDS,
)
def analyze_textbook_task(self, textbook_id: int) -> dict[str, Any]:
    """Celery task: Analyze textbook PDF structure and detect chapters."""
    logger.info("Celery [task_id=%s]: Starting analyze_textbook_task for textbook_id=%s", self.request.id, textbook_id)
    try:
        return handle_textbook_analysis(textbook_id)
    except Exception as exc:
        logger.exception("analyze_textbook_task failed on attempt %d: %s", self.request.retries + 1, exc)
        if self.request.retries < self.max_retries:
            countdown = min(DEFAULT_RETRY_BACKOFF_SECONDS * (2 ** self.request.retries), MAX_RETRY_BACKOFF_SECONDS)
            raise self.retry(exc=exc, countdown=countdown)
        raise exc


@celery_app.task(
    name=TASK_PROCESS_CHAPTERS,
    queue=QUEUE_CATALOG,
    acks_late=True,
    bind=True,
    max_retries=MAX_INGESTION_RETRIES,
    default_retry_delay=DEFAULT_RETRY_BACKOFF_SECONDS,
)
def process_chapters_task(self, textbook_id: int) -> dict[str, Any]:
    """Celery task: Process and embed all confirmed chapters for a textbook."""
    logger.info("Celery [task_id=%s]: Starting process_chapters_task for textbook_id=%s", self.request.id, textbook_id)
    try:
        return handle_textbook_chapters(textbook_id)
    except Exception as exc:
        logger.exception("process_chapters_task failed on attempt %d: %s", self.request.retries + 1, exc)
        if self.request.retries < self.max_retries:
            countdown = min(DEFAULT_RETRY_BACKOFF_SECONDS * (2 ** self.request.retries), MAX_RETRY_BACKOFF_SECONDS)
            raise self.retry(exc=exc, countdown=countdown)
        raise exc
