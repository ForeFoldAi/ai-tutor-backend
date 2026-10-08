"""Handlers for processing catalog and textbook pipeline tasks."""

from app.services.catalog_pipeline.handlers.upload_handler import handle_upload_processing
from app.services.catalog_pipeline.handlers.structure_handler import handle_textbook_analysis
from app.services.catalog_pipeline.handlers.chapter_handler import handle_textbook_chapters

__all__ = [
    "handle_upload_processing",
    "handle_textbook_analysis",
    "handle_textbook_chapters",
]
