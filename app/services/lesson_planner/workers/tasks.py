from __future__ import annotations

import logging
import os

from celery.exceptions import SoftTimeLimitExceeded

from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.modules.teacher.lesson_planner.constants import QUEUE_AUTOSAVE, QUEUE_EXPORT, QUEUE_GENERATE, QUEUE_REGENERATE
from app.modules.teacher.lesson_planner.models import LessonExport, LessonJob
from app.modules.teacher.lesson_planner.constants import ExportStatus
from app.services.lesson_planner.orchestrator import execute_generation_job
from app.services.lesson_planner.export.service import run_export

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="lesson_planner.generate",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    max_retries=int(os.environ.get("LESSON_PLANNER_MAX_RETRIES", "3")),
    queue=QUEUE_GENERATE,
)
def lesson_generation_worker(self, job_id: str, resume: bool = False) -> dict:
    db = SessionLocal()
    try:
        job = db.get(LessonJob, int(job_id))
        if job:
            job.celery_task_id = self.request.id
            db.commit()
    finally:
        db.close()

    try:
        return execute_generation_job(job_id, resume=resume)
    except SoftTimeLimitExceeded:
        logger.error("Generation job timed out: %s", job_id)
        raise


@celery_app.task(
    bind=True,
    name="lesson_planner.regenerate",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=int(os.environ.get("LESSON_PLANNER_MAX_RETRIES", "3")),
    queue=QUEUE_REGENERATE,
)
def regeneration_worker(self, job_id: str) -> dict:
    return lesson_generation_worker(job_id)


@celery_app.task(
    bind=True,
    name="lesson_planner.export",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=2,
    queue=QUEUE_EXPORT,
)
def export_worker(self, export_id: str, artifact_types: list[str] | None = None) -> dict:
    db = SessionLocal()
    try:
        export = db.get(LessonExport, int(export_id))
        if export:
            export.celery_task_id = self.request.id
            export.status = ExportStatus.PROCESSING
            db.commit()
    finally:
        db.close()
    return run_export(export_id, artifact_types=artifact_types)


@celery_app.task(name="lesson_planner.autosave", queue=QUEUE_AUTOSAVE)
def autosave_worker(job_id: str, snapshot: dict) -> dict:
    from app.services.lesson_planner.redis.job_state import publish_event

    publish_event(job_id, {"event": "autosave", "snapshot": snapshot})
    return {"ok": True}
