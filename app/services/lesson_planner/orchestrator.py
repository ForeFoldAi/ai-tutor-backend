from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.database import SessionLocal
from app.modules.teacher.lesson_planner.constants import (
    ArtifactStatus,
    ArtifactType,
    ExportStatus,
    JobStatus,
    LessonPlanStatus,
    WS_PROGRESS,
)
from app.modules.teacher.lesson_planner.models import LessonArtifact, LessonExport, LessonJob, LessonPlan, LessonVersion
from app.modules.teacher.lesson_planner.service import build_plan_snapshot
from app.services.lesson_planner.graph import run_planner_workflow
from app.services.lesson_planner.indexing.chroma_indexer import index_lesson_planner_collections
from app.services.lesson_planner.observability.metrics import ARTIFACT_GENERATION, GENERATION_JOBS, GENERATION_LATENCY
from app.services.lesson_planner.observability.tracing import setup_lesson_planner_otel, trace_span
from app.services.lesson_planner.redis.job_state import is_cancelled, publish_event
from app.services.lesson_planner.security import sanitize_payload
from app.services.lesson_planner.state import initial_state_from_payload

logger = logging.getLogger(__name__)

setup_lesson_planner_otel()


def _persist_outputs(
    db,
    *,
    job: LessonJob,
    plan: LessonPlan,
    outputs: dict[str, dict[str, Any]],
    user_id: int,
) -> None:
    for artifact_key, content in outputs.items():
        try:
            artifact_type = ArtifactType(artifact_key)
        except ValueError:
            continue
        row = db.scalar(
            select(LessonArtifact).where(
                LessonArtifact.lesson_plan_id == plan.id,
                LessonArtifact.artifact_type == artifact_type,
            )
        )
        if row is None:
            row = LessonArtifact(
                lesson_plan_id=plan.id,
                job_id=job.id,
                artifact_type=artifact_type,
                version_number=1,
            )
            db.add(row)
        row.content = content
        row.status = ArtifactStatus.COMPLETED
        row.job_id = job.id

    latest = db.scalar(
        select(LessonVersion.version_number)
        .where(LessonVersion.lesson_plan_id == plan.id)
        .order_by(LessonVersion.version_number.desc())
        .limit(1)
    )
    next_version = (latest or 0) + 1
    db.refresh(plan)
    snapshot = build_plan_snapshot(plan)
    version = LessonVersion(
        lesson_plan_id=plan.id,
        version_number=next_version,
        snapshot=snapshot,
        change_summary="Auto-saved after generation",
        created_by=user_id,
    )
    db.add(version)


def _maybe_index_chroma(payload: dict[str, Any]) -> None:
    chapter_id = payload.get("chapter_id")
    if not chapter_id:
        return
    try:
        index_lesson_planner_collections(
            board=payload.get("board"),
            grade=str(payload.get("grade", "")),
            subject=str(payload.get("subject", "")),
            chapter_id=str(chapter_id),
            chapter_name=str(payload.get("chapter_name", "")),
        )
    except Exception as exc:
        logger.warning("Chroma indexing skipped: %s", exc)


def execute_generation_job(job_id: str, *, resume: bool = False) -> dict[str, Any]:
    db = SessionLocal()
    try:
        job = db.get(LessonJob, int(job_id))
        if not job:
            return {"ok": False, "error": "job not found"}

        if is_cancelled(job_id):
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(UTC)
            GENERATION_JOBS.labels(status="cancelled").inc()
            db.commit()
            return {"ok": False, "error": "cancelled"}

        job.status = JobStatus.RUNNING
        job.started_at = job.started_at or datetime.now(UTC)
        job.progress = 5
        job.message = "Resuming generation" if resume else "Starting generation"
        db.commit()

        plan = db.get(LessonPlan, job.lesson_plan_id) if job.lesson_plan_id else None
        if not plan:
            job.status = JobStatus.FAILED
            job.error_message = "Lesson plan not found"
            job.completed_at = datetime.now(UTC)
            db.commit()
            return {"ok": False, "error": "plan not found"}

        payload = sanitize_payload(job.input_payload or {})
        if not resume:
            _maybe_index_chroma(payload)

        state = initial_state_from_payload(
            payload,
            job_id=job_id,
            user_id=str(job.user_id),
            lesson_plan_id=str(plan.id),
        )

        publish_event(job_id, {"event": WS_PROGRESS, "progress": 8, "message": "Loading workflow"})
        started = time.perf_counter()
        with trace_span("lesson_planner.execute_job", attributes={"job_id": job_id, "resume": resume}):
            final_state = run_planner_workflow(state, resume=resume)
        elapsed = time.perf_counter() - started
        GENERATION_LATENCY.observe(elapsed)

        if is_cancelled(job_id):
            job.status = JobStatus.CANCELLED
            plan.status = LessonPlanStatus.DRAFT
            job.completed_at = datetime.now(UTC)
            GENERATION_JOBS.labels(status="cancelled").inc()
            db.commit()
            return {"ok": False, "error": "cancelled"}

        outputs = final_state.get("outputs") or {}
        for artifact_key in outputs:
            ARTIFACT_GENERATION.labels(artifact_type=artifact_key, status="completed").inc()
        _persist_outputs(db, job=job, plan=plan, outputs=outputs, user_id=job.user_id)

        errors = final_state.get("errors") or []
        if errors and not outputs:
            job.status = JobStatus.FAILED
            job.error_message = "; ".join(errors)
            plan.status = LessonPlanStatus.FAILED
            GENERATION_JOBS.labels(status="failed").inc()
        else:
            job.status = JobStatus.COMPLETED
            plan.status = LessonPlanStatus.COMPLETED
            job.error_message = "; ".join(errors) if errors else None
            GENERATION_JOBS.labels(status="completed").inc()

        job.progress = 100
        job.message = "Generation complete"
        job.completed_at = datetime.now(UTC)
        db.commit()
        return {"ok": True, "outputs": outputs, "errors": errors}
    except Exception as exc:
        logger.exception("Generation job failed: %s", job_id)
        db.rollback()
        job = db.get(LessonJob, int(job_id))
        if job:
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            job.completed_at = datetime.now(UTC)
            if job.lesson_plan_id:
                plan = db.get(LessonPlan, job.lesson_plan_id)
                if plan:
                    plan.status = LessonPlanStatus.FAILED
            db.commit()
        GENERATION_JOBS.labels(status="failed").inc()
        publish_event(job_id, {"event": "error", "message": str(exc)})
        return {"ok": False, "error": str(exc)}
    finally:
        db.close()


def get_export_for_user(db, export_id: int, user_id: int) -> LessonExport:
    from app.modules.teacher.lesson_planner.service import LessonPlannerNotFoundError

    export = db.get(LessonExport, export_id)
    if not export:
        raise LessonPlannerNotFoundError("Export not found")
    plan = db.get(LessonPlan, export.lesson_plan_id)
    if not plan or plan.user_id != user_id or plan.deleted_at is not None:
        raise LessonPlannerNotFoundError("Export not found")
    return export
