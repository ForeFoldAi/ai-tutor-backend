from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import exists, not_, select
from sqlalchemy.orm import Session, selectinload

from app.modules.teacher.lesson_planner.constants import (
    ArtifactStatus,
    ArtifactType,
    ExportStatus,
    JobStatus,
    LessonPlanStatus,
)
from app.modules.teacher.lesson_planner.models import LessonArtifact, LessonExport, LessonJob, LessonPlan, LessonVersion
from app.modules.teacher.lesson_planner.schemas import (
    GenerateLessonPlanRequest,
    LessonPlanPatchRequest,
    SaveLessonPlanRequest,
)


class LessonPlannerNotFoundError(Exception):
    pass


class LessonPlannerConflictError(Exception):
    pass


def _active_plan_query(user_id: int):
    return select(LessonPlan).where(
        LessonPlan.user_id == user_id,
        LessonPlan.deleted_at.is_(None),
    )


def get_lesson_plan(db: Session, plan_id: int, user_id: int) -> LessonPlan:
    plan = db.scalar(
        _active_plan_query(user_id)
        .where(LessonPlan.id == plan_id)
        .options(selectinload(LessonPlan.artifacts))
    )
    if not plan:
        raise LessonPlannerNotFoundError("Lesson plan not found")
    return plan


def list_lesson_plans(db: Session, user_id: int, *, limit: int = 50, offset: int = 0) -> list[LessonPlan]:
    # Only plans the tutor explicitly saved (not generate/auto-save drafts).
    user_saved = exists(
        select(LessonVersion.id).where(
            LessonVersion.lesson_plan_id == LessonPlan.id,
            LessonVersion.change_summary.is_not(None),
            not_(LessonVersion.change_summary.ilike("auto-saved%")),
        )
    )
    return list(
        db.scalars(
            _active_plan_query(user_id)
            .where(user_saved)
            .order_by(LessonPlan.updated_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()
    )


def create_lesson_plan_from_request(
    db: Session,
    user_id: int,
    payload: GenerateLessonPlanRequest,
    *,
    status: LessonPlanStatus = LessonPlanStatus.DRAFT,
) -> LessonPlan:
    topic_suffix = f" — {', '.join(payload.topics)}" if payload.topics else ""
    title = payload.title or f"{payload.subject} — {payload.chapter_name}{topic_suffix}"
    plan = LessonPlan(
        user_id=user_id,
        title=title,
        grade=payload.grade,
        subject=payload.subject,
        board=payload.board,
        chapter_id=payload.chapter_id,
        chapter_name=payload.chapter_name,
        duration_minutes=payload.duration_minutes,
        learning_objectives=payload.learning_objectives,
        status=status,
        plan_metadata={
            "requested_artifacts": [a.value for a in payload.requested_artifacts],
            "topics": list(payload.topics),
            "sections": list(payload.sections),
            "ppt_template": payload.ppt_template,
            "ppt_slide_count": payload.ppt_slide_count,
        },
    )
    db.add(plan)
    db.flush()
    return plan


def ensure_artifact_rows(
    db: Session,
    plan: LessonPlan,
    artifact_types: list[ArtifactType],
    *,
    job_id: int | None = None,
) -> list[LessonArtifact]:
    existing = {a.artifact_type: a for a in plan.artifacts}
    rows: list[LessonArtifact] = []
    for artifact_type in artifact_types:
        row = existing.get(artifact_type)
        if row is None:
            row = LessonArtifact(
                lesson_plan_id=plan.id,
                job_id=job_id,
                artifact_type=artifact_type,
                status=ArtifactStatus.PENDING,
                version_number=1,
            )
            db.add(row)
            rows.append(row)
        else:
            row.job_id = job_id
            row.status = ArtifactStatus.PENDING
            rows.append(row)
    db.flush()
    return rows


def create_generation_job(
    db: Session,
    *,
    user_id: int,
    plan: LessonPlan,
    payload: GenerateLessonPlanRequest,
    idempotency_key: str | None = None,
) -> LessonJob:
    if idempotency_key:
        existing = db.scalar(
            select(LessonJob).where(LessonJob.idempotency_key == idempotency_key)
        )
        if existing:
            raise LessonPlannerConflictError("Duplicate idempotency key")

    job = LessonJob(
        lesson_plan_id=plan.id,
        user_id=user_id,
        status=JobStatus.QUEUED,
        progress=0,
        message="Queued for generation",
        requested_artifacts=[a.value for a in payload.requested_artifacts],
        input_payload=payload.model_dump(mode="json"),
        idempotency_key=idempotency_key,
    )
    db.add(job)
    plan.status = LessonPlanStatus.GENERATING
    db.flush()
    ensure_artifact_rows(db, plan, payload.requested_artifacts, job_id=job.id)
    return job


def get_job(db: Session, job_id: int, user_id: int) -> LessonJob:
    job = db.scalar(
        select(LessonJob).where(LessonJob.id == job_id, LessonJob.user_id == user_id)
    )
    if not job:
        raise LessonPlannerNotFoundError("Job not found")
    return job


def update_lesson_plan(db: Session, plan_id: int, user_id: int, patch: LessonPlanPatchRequest) -> LessonPlan:
    plan = get_lesson_plan(db, plan_id, user_id)
    for field, value in patch.model_dump(exclude_unset=True).items():
        setattr(plan, field, value)
    db.flush()
    return plan


def soft_delete_lesson_plan(db: Session, plan_id: int, user_id: int) -> None:
    plan = get_lesson_plan(db, plan_id, user_id)
    plan.deleted_at = datetime.now(UTC)
    plan.status = LessonPlanStatus.CANCELLED
    db.flush()


def save_lesson_plan_version(
    db: Session,
    user_id: int,
    payload: SaveLessonPlanRequest,
) -> LessonVersion:
    plan = get_lesson_plan(db, payload.lesson_plan_id, user_id)
    if payload.title:
        plan.title = payload.title

    latest = db.scalar(
        select(LessonVersion.version_number)
        .where(LessonVersion.lesson_plan_id == plan.id)
        .order_by(LessonVersion.version_number.desc())
        .limit(1)
    )
    next_version = (latest or 0) + 1

    for artifact_key, content in payload.artifacts.items():
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
                artifact_type=artifact_type,
                content=content,
                status=ArtifactStatus.COMPLETED,
                version_number=next_version,
            )
            db.add(row)
        else:
            row.content = content
            row.status = ArtifactStatus.COMPLETED
            row.version_number = next_version

    snapshot = build_plan_snapshot(plan)
    version = LessonVersion(
        lesson_plan_id=plan.id,
        version_number=next_version,
        snapshot=snapshot,
        change_summary=payload.change_summary,
        created_by=user_id,
    )
    db.add(version)
    meta = dict(plan.plan_metadata or {})
    meta["user_saved"] = True
    plan.plan_metadata = meta
    plan.status = LessonPlanStatus.COMPLETED
    db.flush()
    return version


def build_plan_snapshot(plan: LessonPlan) -> dict[str, Any]:
    return {
        "id": str(plan.id),
        "title": plan.title,
        "grade": plan.grade,
        "subject": plan.subject,
        "board": plan.board,
        "chapter_id": plan.chapter_id,
        "chapter_name": plan.chapter_name,
        "duration_minutes": plan.duration_minutes,
        "learning_objectives": plan.learning_objectives,
        "status": plan.status.value,
        "artifacts": {
            a.artifact_type.value: a.content
            for a in (plan.artifacts or [])
            if a.content is not None
        },
    }


def create_export_record(
    db: Session,
    *,
    user_id: int,
    lesson_plan_id: int,
    export_format: str,
    version_id: int | None = None,
) -> LessonExport:
    plan = get_lesson_plan(db, lesson_plan_id, user_id)
    export = LessonExport(
        lesson_plan_id=plan.id,
        version_id=version_id,
        export_format=export_format,
        status=ExportStatus.PENDING,
    )
    db.add(export)
    db.flush()
    return export


def mark_job_cancelled(db: Session, job_id: int) -> None:
    job = db.get(LessonJob, job_id)
    if not job:
        return
    job.status = JobStatus.CANCELLED
    job.message = "Cancelled by user"
    job.completed_at = datetime.now(UTC)
    if job.lesson_plan_id:
        plan = db.get(LessonPlan, job.lesson_plan_id)
        if plan and plan.status == LessonPlanStatus.GENERATING:
            plan.status = LessonPlanStatus.DRAFT
    db.flush()
