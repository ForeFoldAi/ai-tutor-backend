"""Catalog processing pipeline (Celery when available, background tasks otherwise)."""

from app.services.catalog_pipeline.dispatcher import (
    dispatch_textbook_analysis,
    dispatch_textbook_chapters,
    dispatch_upload_processing,
)

__all__ = [
    "dispatch_textbook_analysis",
    "dispatch_textbook_chapters",
    "dispatch_upload_processing",
]
