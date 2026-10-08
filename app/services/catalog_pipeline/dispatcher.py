"""Safe dispatcher for catalog pipeline tasks with automatic Celery/BackgroundTasks fallback."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import BackgroundTasks

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _can_use_celery() -> bool:
    settings = get_settings()
    return bool(settings.redis_url and str(settings.redis_url).strip())


def dispatch_upload_processing(upload_id: int, background_tasks: Any = None) -> None:
    """Dispatches standalone upload processing to Celery if available, else BackgroundTasks."""
    if _can_use_celery():
        try:
            from app.services.catalog_pipeline.tasks import process_upload_task

            task = process_upload_task.delay(upload_id)
            logger.info("Dispatched upload %s to Celery [task_id=%s]", upload_id, task.id)
            return
        except Exception as exc:
            logger.warning("Failed to dispatch upload %s to Celery (%s); falling back to background_tasks", upload_id, exc)

    from app.services.catalog_pipeline.handlers.upload_handler import handle_upload_processing

    if background_tasks is not None:
        background_tasks.add_task(handle_upload_processing, upload_id)
        logger.info("Dispatched upload %s to FastAPI BackgroundTasks", upload_id)
    else:
        threading.Thread(target=handle_upload_processing, args=(upload_id,), daemon=True).start()
        logger.info("Dispatched upload %s to local daemon thread", upload_id)


def dispatch_textbook_analysis(textbook_id: int, background_tasks: Any = None) -> None:
    """Dispatches full textbook structure detection to Celery if available, else BackgroundTasks."""
    if _can_use_celery():
        try:
            from app.services.catalog_pipeline.tasks import analyze_textbook_task

            task = analyze_textbook_task.delay(textbook_id)
            logger.info("Dispatched textbook analysis %s to Celery [task_id=%s]", textbook_id, task.id)
            return
        except Exception as exc:
            logger.warning("Failed to dispatch textbook %s to Celery (%s); falling back to background_tasks", textbook_id, exc)

    from app.services.catalog_pipeline.handlers.structure_handler import handle_textbook_analysis

    if background_tasks is not None:
        background_tasks.add_task(handle_textbook_analysis, textbook_id)
        logger.info("Dispatched textbook analysis %s to FastAPI BackgroundTasks", textbook_id)
    else:
        threading.Thread(target=handle_textbook_analysis, args=(textbook_id,), daemon=True).start()
        logger.info("Dispatched textbook analysis %s to local daemon thread", textbook_id)


def dispatch_textbook_chapters(textbook_id: int, background_tasks: Any = None) -> None:
    """Dispatches confirmed chapters processing to Celery if available, else BackgroundTasks."""
    if _can_use_celery():
        try:
            from app.services.catalog_pipeline.tasks import process_chapters_task

            task = process_chapters_task.delay(textbook_id)
            logger.info("Dispatched textbook chapters %s to Celery [task_id=%s]", textbook_id, task.id)
            return
        except Exception as exc:
            logger.warning("Failed to dispatch chapters for textbook %s to Celery (%s); falling back to background_tasks", textbook_id, exc)

    from app.services.catalog_pipeline.handlers.chapter_handler import handle_textbook_chapters

    if background_tasks is not None:
        background_tasks.add_task(handle_textbook_chapters, textbook_id)
        logger.info("Dispatched textbook chapters %s to FastAPI BackgroundTasks", textbook_id)
    else:
        threading.Thread(target=handle_textbook_chapters, args=(textbook_id,), daemon=True).start()
        logger.info("Dispatched textbook chapters %s to local daemon thread", textbook_id)
