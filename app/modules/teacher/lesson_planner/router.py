from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.teacher.lesson_planner.constants import JobStatus, LessonPlanStatus
from app.modules.teacher.lesson_planner.schemas import (
    CancelJobRequest,
    ChapterTopicsResponse,
    ExportLessonPlanRequest,
    ExportResponse,
    GenerateLessonPlanRequest,
    GenerateResponse,
    JobResponse,
    LessonPlanPatchRequest,
    LessonPlanResponse,
    LessonPlanSummaryResponse,
    MessageResponse,
    PptThemesResponse,
    RegenerateRequest,
    ResumeJobRequest,
    SaveLessonPlanRequest,
)
from app.modules.teacher.lesson_planner.service import (
    LessonPlannerConflictError,
    LessonPlannerNotFoundError,
    create_export_record,
    create_generation_job,
    create_lesson_plan_from_request,
    get_job,
    get_lesson_plan,
    list_lesson_plans,
    mark_job_cancelled,
    save_lesson_plan_version,
    soft_delete_lesson_plan,
    update_lesson_plan,
)
from app.modules.users.models import User
from app.services.lesson_planner.orchestrator import get_export_for_user
from app.services.lesson_planner.redis.job_state import request_cancel
from app.services.lesson_planner.security import sanitize_payload
from app.services.lesson_planner.workers.tasks import export_worker, lesson_generation_worker, regeneration_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/lesson-planner", tags=["lesson-planner"])

TutorUser = Annotated[User, Depends(require_roles(Role.TUTOR, Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))]
DbSession = Annotated[Session, Depends(get_db)]


def _check_rate_limit(request: Request, user: User) -> None:
    from app.services.lesson_planner.rate_limit import check_rate_limit

    check_rate_limit(str(user.id), request.client.host if request.client else "unknown")


@router.get("/chapter-topics", response_model=ChapterTopicsResponse)
def chapter_topics(
    db: DbSession,
    current_user: TutorUser,
    chapter_id: str,
    subject: str = "",
    board: str = "",
    class_level: str = "",
    chapter_name: str = "",
) -> ChapterTopicsResponse:
    """Topic headings for a textbook chapter (for lesson topic multi-select)."""
    _ = db, current_user  # auth gate only
    from app.modules.student_learning.topic_progress import list_chapter_topics

    try:
        cid = int(chapter_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid chapter_id") from exc

    topics = list_chapter_topics(
        chapter_id=cid,
        board=board,
        class_level=class_level,
        subject_name=subject,
        chapter_name=chapter_name,
    )
    return ChapterTopicsResponse(chapter_id=str(cid), topics=topics)


@router.get("/ppt-themes", response_model=PptThemesResponse)
def ppt_themes(current_user: TutorUser) -> PptThemesResponse:
    """Available classroom PPT templates for lesson planner decks."""
    _ = current_user
    from app.services.lesson_planner.export.pptx_themes import DEFAULT_THEME_ID, list_themes

    return PptThemesResponse(themes=list_themes(), default_id=DEFAULT_THEME_ID)


@router.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_202_ACCEPTED)
def generate_lesson_plan(
    payload: GenerateLessonPlanRequest,
    request: Request,
    db: DbSession,
    current_user: TutorUser,
) -> GenerateResponse:
    _check_rate_limit(request, current_user)
    safe_payload = GenerateLessonPlanRequest.model_validate(sanitize_payload(payload.model_dump()))

    if safe_payload.lesson_plan_id:
        plan = get_lesson_plan(db, safe_payload.lesson_plan_id, current_user.id)
    else:
        plan = create_lesson_plan_from_request(db, current_user.id, safe_payload)

    try:
        job = create_generation_job(
            db,
            user_id=current_user.id,
            plan=plan,
            payload=safe_payload,
            idempotency_key=safe_payload.idempotency_key,
        )
    except LessonPlannerConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    db.commit()
    async_result = lesson_generation_worker.delay(str(job.id))
    job.celery_task_id = async_result.id
    db.commit()

    return GenerateResponse(
        job_id=job.id,
        lesson_plan_id=plan.id,
        status=JobStatus.QUEUED,
        websocket_url=f"/ws/lesson-planner/{job.id}",
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_generation_job(job_id: int, db: DbSession, current_user: TutorUser) -> JobResponse:
    try:
        job = get_job(db, job_id, current_user.id)
    except LessonPlannerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return JobResponse.model_validate(job)


@router.get("/{plan_id}", response_model=LessonPlanResponse)
def get_plan(plan_id: int, db: DbSession, current_user: TutorUser) -> LessonPlanResponse:
    try:
        plan = get_lesson_plan(db, plan_id, current_user.id)
    except LessonPlannerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return LessonPlanResponse.model_validate(plan)


@router.get("", response_model=list[LessonPlanSummaryResponse])
def list_plans(
    db: DbSession,
    current_user: TutorUser,
    limit: int = 50,
    offset: int = 0,
) -> list[LessonPlanSummaryResponse]:
    plans = list_lesson_plans(db, current_user.id, limit=limit, offset=offset)
    return [LessonPlanSummaryResponse.model_validate(p) for p in plans]


@router.patch("/{plan_id}", response_model=LessonPlanResponse)
def patch_plan(
    plan_id: int,
    payload: LessonPlanPatchRequest,
    db: DbSession,
    current_user: TutorUser,
) -> LessonPlanResponse:
    try:
        plan = update_lesson_plan(db, plan_id, current_user.id, payload)
        db.commit()
        plan = get_lesson_plan(db, plan_id, current_user.id)
    except LessonPlannerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return LessonPlanResponse.model_validate(plan)


@router.delete("/{plan_id}", response_model=MessageResponse)
def delete_plan(plan_id: int, db: DbSession, current_user: TutorUser) -> MessageResponse:
    try:
        soft_delete_lesson_plan(db, plan_id, current_user.id)
        db.commit()
    except LessonPlannerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return MessageResponse(message="Lesson plan deleted")


@router.post("/save", response_model=MessageResponse)
def save_plan(payload: SaveLessonPlanRequest, db: DbSession, current_user: TutorUser) -> MessageResponse:
    try:
        save_lesson_plan_version(db, current_user.id, payload)
        db.commit()
    except LessonPlannerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return MessageResponse(message="Lesson plan saved")


@router.post("/export", response_model=ExportResponse, status_code=status.HTTP_202_ACCEPTED)
def export_plan(payload: ExportLessonPlanRequest, db: DbSession, current_user: TutorUser) -> ExportResponse:
    try:
        export = create_export_record(
            db,
            user_id=current_user.id,
            lesson_plan_id=payload.lesson_plan_id,
            export_format=payload.export_format.value,
            version_id=payload.version_id,
        )
        db.commit()
    except LessonPlannerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    async_result = export_worker.delay(
        str(export.id),
        [a.value for a in payload.artifact_types] if payload.artifact_types else None,
    )
    export.celery_task_id = async_result.id
    db.commit()
    return ExportResponse(
        export_id=export.id,
        status=export.status.value,
        file_path=export.file_path,
        download_url=f"/api/lesson-planner/exports/{export.id}/download",
    )


@router.get("/exports/{export_id}", response_model=ExportResponse)
def get_export(export_id: int, db: DbSession, current_user: TutorUser) -> ExportResponse:
    try:
        export = get_export_for_user(db, export_id, current_user.id)
    except LessonPlannerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ExportResponse(
        export_id=export.id,
        status=export.status.value,
        file_path=export.file_path,
        download_url=f"/api/lesson-planner/exports/{export.id}/download",
    )


@router.get("/exports/{export_id}/download")
def download_export(export_id: int, db: DbSession, current_user: TutorUser) -> FileResponse:
    try:
        export = get_export_for_user(db, export_id, current_user.id)
    except LessonPlannerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if export.status.value != "completed" or not export.file_path:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Export not ready")
    path = Path(export.file_path)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Export file missing")
    media = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    return FileResponse(
        path,
        media_type=media.get(export.export_format.value, "application/octet-stream"),
        filename=path.name,
    )


@router.post("/cancel", response_model=MessageResponse)
def cancel_generation(payload: CancelJobRequest, db: DbSession, current_user: TutorUser) -> MessageResponse:
    try:
        job = get_job(db, payload.job_id, current_user.id)
    except LessonPlannerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    request_cancel(str(job.id))
    if job.celery_task_id:
        from app.core.celery_app import celery_app

        celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGTERM")
    mark_job_cancelled(db, job.id)
    db.commit()
    return MessageResponse(message="Generation cancelled")


@router.post("/resume", response_model=GenerateResponse, status_code=status.HTTP_202_ACCEPTED)
def resume_generation(
    payload: ResumeJobRequest,
    request: Request,
    db: DbSession,
    current_user: TutorUser,
) -> GenerateResponse:
    _check_rate_limit(request, current_user)
    try:
        job = get_job(db, payload.job_id, current_user.id)
    except LessonPlannerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if job.status not in {JobStatus.FAILED, JobStatus.CANCELLED}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job cannot be resumed")

    from app.services.lesson_planner.checkpoint.store import load_checkpoint

    if not load_checkpoint(str(job.id)):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No checkpoint found for job")

    job.status = JobStatus.QUEUED
    job.message = "Queued for resume"
    job.error_message = None
    if job.lesson_plan_id:
        plan = get_lesson_plan(db, job.lesson_plan_id, current_user.id)
        plan.status = LessonPlanStatus.GENERATING
    db.commit()

    if not job.lesson_plan_id:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job has no lesson plan")

    async_result = lesson_generation_worker.delay(str(job.id), resume=True)
    job.celery_task_id = async_result.id
    db.commit()

    return GenerateResponse(
        job_id=job.id,
        lesson_plan_id=job.lesson_plan_id,
        status=JobStatus.QUEUED,
        websocket_url=f"/ws/lesson-planner/{job.id}",
    )


@router.post("/regenerate", response_model=GenerateResponse, status_code=status.HTTP_202_ACCEPTED)
def regenerate_artifacts(
    payload: RegenerateRequest,
    request: Request,
    db: DbSession,
    current_user: TutorUser,
) -> GenerateResponse:
    _check_rate_limit(request, current_user)
    try:
        plan = get_lesson_plan(db, payload.lesson_plan_id, current_user.id)
    except LessonPlannerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    generate_payload = GenerateLessonPlanRequest(
        grade=plan.grade,
        subject=plan.subject,
        chapter_id=plan.chapter_id,
        chapter_name=plan.chapter_name,
        duration_minutes=plan.duration_minutes,
        learning_objectives=plan.learning_objectives,
        board=plan.board,
        title=plan.title,
        requested_artifacts=payload.artifact_types,
        lesson_plan_id=plan.id,
        idempotency_key=payload.idempotency_key,
    )
    try:
        job = create_generation_job(
            db,
            user_id=current_user.id,
            plan=plan,
            payload=generate_payload,
            idempotency_key=payload.idempotency_key,
        )
    except LessonPlannerConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    db.commit()
    async_result = regeneration_worker.delay(str(job.id))
    job.celery_task_id = async_result.id
    db.commit()
    return GenerateResponse(
        job_id=job.id,
        lesson_plan_id=plan.id,
        status=JobStatus.QUEUED,
        websocket_url=f"/ws/lesson-planner/{job.id}",
    )
