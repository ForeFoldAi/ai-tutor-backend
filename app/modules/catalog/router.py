import asyncio
import logging
import os
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, get_db
from app.core.student_messages import IMAGE_INVALID_PATH, IMAGE_NOT_AVAILABLE, PDF_ONLY
from app.core.text_clean import clean_display_label
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import get_current_user, get_current_user_bearer_or_query, require_roles
from app.modules.auth.schemas import MessageResponse
from app.modules.catalog.models import (
    BoardDefinition,
    BoardEnum,
    ClassEnum,
    ProcessingStatusEnum,
    SyllabusSubject,
    Textbook,
    TextbookChapter,
    TextbookImage,
    TextbookUpload,
)
from app.modules.catalog.publisher_books import MANUAL_UPLOAD, attach_upload_to_book
from app.modules.catalog.schemas import (
    BoardCreateRequest,
    BoardResponse,
    CatalogEnumsResponse,
    EmbeddingStatsResponse,
    ProcessResponse,
    StudentChapterResponse,
    StudentSubjectResponse,
    SyllabusBulkCreateRequest,
    SyllabusSubjectResponse,
    SyllabusUpdateRequest,
    TextbookUploadCreateRequest,
    TextbookUploadPatchStatusRequest,
    TextbookUploadResponse,
    TextbookUploadUpdateRequest,
    TextbookImageAdminResponse,
    TextbookImageUpdateRequest,
    ActiveJobInfo,
    QueueDepthInfo,
    RecentProcessInfo,
    WorkerNodeInfo,
    WorkerStatusResponse,
)
from app.modules.users.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/admin/catalog", tags=["master-admin-catalog"])


@router.get("/enums", response_model=CatalogEnumsResponse)
def list_catalog_enums(
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    return CatalogEnumsResponse(boards=[b for b in BoardEnum], classes=[c for c in ClassEnum])


@router.get("/boards", response_model=list[BoardResponse])
def list_boards(
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    rows = list(db.scalars(select(BoardDefinition).order_by(BoardDefinition.board)))
    return [BoardResponse.model_validate(r) for r in rows]


@router.post("/boards", response_model=BoardResponse, status_code=status.HTTP_201_CREATED)
def create_board(
    payload: BoardCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    row = BoardDefinition(board=payload.board, country=payload.country.strip())
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # If exists, return existing row for idempotent UX.
        existing = db.scalar(select(BoardDefinition).where(BoardDefinition.board == payload.board))
        if existing:
            return BoardResponse.model_validate(existing)
        raise
    db.refresh(row)
    return BoardResponse.model_validate(row)


@router.delete("/boards/{board_id}", response_model=MessageResponse)
def delete_board(
    board_id: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    row = db.get(BoardDefinition, board_id)
    if not row:
        return MessageResponse(message="Board not found.")
    db.delete(row)
    db.commit()
    return MessageResponse(message="Board deleted.")


@router.get("/syllabus", response_model=list[SyllabusSubjectResponse])
def list_syllabus(
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    rows = list(db.scalars(select(SyllabusSubject).order_by(SyllabusSubject.board, SyllabusSubject.class_level, SyllabusSubject.subject_name)))
    return [SyllabusSubjectResponse.model_validate(r) for r in rows]


@router.post("/syllabus", response_model=list[SyllabusSubjectResponse], status_code=status.HTTP_201_CREATED)
def create_syllabus_bulk(
    payload: SyllabusBulkCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    created: list[SyllabusSubject] = []
    for cls in payload.class_names:
        for subject in payload.subject_names:
            exists = db.scalar(
                select(SyllabusSubject).where(
                    SyllabusSubject.board == payload.board,
                    SyllabusSubject.class_level == cls,
                    SyllabusSubject.subject_name == subject,
                )
            )
            if exists:
                created.append(exists)
                continue
            row = SyllabusSubject(board=payload.board, class_level=cls, subject_name=subject)
            db.add(row)
            created.append(row)
    db.commit()
    for row in created:
        db.refresh(row)
    return [SyllabusSubjectResponse.model_validate(r) for r in created]


@router.patch("/syllabus/{syllabus_id}", response_model=SyllabusSubjectResponse)
def update_syllabus(
    syllabus_id: int,
    payload: SyllabusUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    row = db.get(SyllabusSubject, syllabus_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Syllabus entry not found.")
    for key, value in payload.model_dump(exclude_unset=True, exclude_none=True).items():
        setattr(row, key, value)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="That subject already exists for this board and class.",
        )
    db.refresh(row)
    return SyllabusSubjectResponse.model_validate(row)


@router.delete("/syllabus/{syllabus_id}", response_model=MessageResponse)
def delete_syllabus(
    syllabus_id: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    row = db.get(SyllabusSubject, syllabus_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Syllabus entry not found.")
    db.delete(row)
    db.commit()
    return MessageResponse(message="Syllabus entry deleted.")


@router.get("/textbook-uploads", response_model=list[TextbookUploadResponse])
def list_textbook_uploads(
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    rows = list(db.scalars(select(TextbookUpload).order_by(TextbookUpload.upload_date.desc())))
    book_of = dict(
        db.execute(
            select(TextbookChapter.textbook_upload_id, TextbookChapter.textbook_id).where(
                TextbookChapter.textbook_upload_id.is_not(None)
            )
        ).all()
    )
    return [
        TextbookUploadResponse.model_validate(r).model_copy(update={"textbook_id": book_of.get(r.id)})
        for r in rows
    ]


@router.post("/textbook-uploads", response_model=TextbookUploadResponse, status_code=status.HTTP_201_CREATED)
def create_textbook_upload(
    payload: TextbookUploadCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    row = TextbookUpload(
        file_name=payload.file_name,
        board=payload.board,
        class_level=payload.class_name,
        subject_name=payload.subject,
        chapter=clean_display_label(payload.chapter),
        content_type=payload.content_type,
        content_label=clean_display_label(payload.content_label),
        uploaded_by=current_user.id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return TextbookUploadResponse.model_validate(row)


# Uploads go through document storage backend (local or S3/MinIO).


async def _save_upload_file(file: UploadFile, board: str, class_level: str, subject: str) -> str:
    """
    Save an uploaded file without blocking the event loop.

    Returns a **relative** storage key (``board/class/subject/name``) suitable for
    both local disk and S3/MinIO. Legacy absolute paths still resolve via
    ``materialize_textbook_file``.
    """
    import hashlib

    content = await file.read()
    digest = hashlib.sha256(content).hexdigest()[:10]
    raw_name = file.filename or "document"
    # Keep basename only; avoid path injection.
    base = os.path.basename(raw_name).replace("..", "_") or "document"
    safe_name = f"{digest}_{base}"
    key = f"{board}/{class_level}/{subject}/{safe_name}"

    def _store() -> None:
        from app.services.image_service.storage_backend import get_document_storage_backend

        get_document_storage_backend().save(key, content)

    await asyncio.to_thread(_store)
    return key


def file_still_referenced(
    db: Session,
    file_path: str | None,
    *,
    exclude_upload_id: int | None = None,
    exclude_textbook_id: int | None = None,
) -> bool:
    """Storage keys are content-hashed, so re-uploading the same file shares one S3 object."""
    if not file_path:
        return False
    other_upload = select(TextbookUpload.id).where(
        TextbookUpload.file_path == file_path, TextbookUpload.id != (exclude_upload_id or -1)
    )
    other_book = select(Textbook.id).where(
        Textbook.file_path == file_path, Textbook.id != (exclude_textbook_id or -1)
    )
    return db.scalar(other_upload.limit(1)) is not None or db.scalar(other_book.limit(1)) is not None


def _local_path_for_upload(file_path: str | None) -> str:
    from app.services.image_service.storage_backend import materialize_textbook_file

    return materialize_textbook_file(file_path or "")


def _process_upload_background(upload_id: int) -> None:
    """Delegate to resilient upload handler."""
    from app.services.catalog_pipeline.handlers.upload_handler import handle_upload_processing

    handle_upload_processing(upload_id)


@router.post("/textbook-uploads/upload", response_model=list[TextbookUploadResponse], status_code=status.HTTP_201_CREATED)
async def upload_textbook_files(
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    files: list[UploadFile] = File(...),
    board: str = Form(...),
    class_name: str = Form(...),
    subject: str = Form(...),
    content_type: str = Form("CHAPTER"),
    content_labels: str = Form(""),
    textbook_id: int | None = Form(None),
):
    """Upload one or more PDF/Word files for a subject.

    ``content_labels`` is a comma-separated list aligned to ``files``.
    If fewer labels than files are provided the file name is used as the label.
    ``textbook_id`` attaches the files to that publisher's book (omit = default pool).
    """
    allowed_ext = {".pdf", ".doc", ".docx"}
    labels = [l.strip() for l in content_labels.split(",")]

    book: Textbook | None = None
    if textbook_id is not None:
        book = db.get(Textbook, textbook_id)
        if (
            not book
            or book.board != BoardEnum(board)
            or book.class_level != ClassEnum(class_name)
            or " ".join(book.subject_name.lower().split()) != " ".join(subject.lower().split())
        ):
            raise HTTPException(status_code=400, detail="Publisher does not belong to this board, class and subject.")

    from app.services.catalog_pipeline import dispatch_upload_processing

    results: list[TextbookUploadResponse] = []
    for idx, file in enumerate(files):
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in allowed_ext:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_ext)}",
            )

        file_path = await _save_upload_file(file, board, class_name, subject)
        label = clean_display_label(
            labels[idx] if idx < len(labels) and labels[idx] else (file.filename or f"File {idx + 1}")
        )

        row = TextbookUpload(
            file_name=file.filename or "document",
            board=BoardEnum(board),
            class_level=ClassEnum(class_name),
            subject_name=subject.strip(),
            chapter=label,
            content_type=content_type,
            content_label=label,
            file_path=file_path,
            uploaded_by=current_user.id,
        )
        db.add(row)
        db.flush()
        if book is not None:
            attach_upload_to_book(db, book, row)
        db.commit()
        db.refresh(row)
        results.append(
            TextbookUploadResponse.model_validate(row).model_copy(update={"textbook_id": book.id if book else None})
        )

        dispatch_upload_processing(row.id, background_tasks)

    return results


@router.post("/textbook-uploads/{upload_id}/process", response_model=ProcessResponse)
def process_textbook_upload(
    upload_id: int,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Manually trigger (re)processing for a single upload."""
    row = db.get(TextbookUpload, upload_id)
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found.")
    if not row.file_path or not os.path.isfile(_local_path_for_upload(row.file_path)):
        raise HTTPException(status_code=400, detail="Source file not found on disk.")

    from app.services.catalog_pipeline import dispatch_upload_processing

    row.chunk_status = ProcessingStatusEnum.QUEUED
    row.embedding_status = ProcessingStatusEnum.QUEUED
    row.ocr_status = ProcessingStatusEnum.QUEUED
    db.commit()
    db.refresh(row)

    dispatch_upload_processing(row.id, background_tasks)

    return ProcessResponse(
        id=row.id,
        chunk_count=row.chunk_count,
        chunk_status=row.chunk_status,
        embedding_status=row.embedding_status,
        message="Processing started.",
    )


@router.get("/embedding-stats", response_model=EmbeddingStatsResponse)
def get_embedding_stats(
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Aggregated embedding pipeline metrics."""
    total = db.scalar(select(func.count()).select_from(TextbookUpload)) or 0
    embedded = db.scalar(
        select(func.count()).select_from(TextbookUpload).where(TextbookUpload.embedding_status == ProcessingStatusEnum.EMBEDDED)
    ) or 0
    failed = db.scalar(
        select(func.count()).select_from(TextbookUpload).where(TextbookUpload.embedding_status == ProcessingStatusEnum.FAILED)
    ) or 0
    pending = total - embedded - failed
    total_chunks = db.scalar(select(func.sum(TextbookUpload.chunk_count)).select_from(TextbookUpload)) or 0

    return EmbeddingStatsResponse(
        total_documents=total,
        embedded_count=embedded,
        failed_count=failed,
        pending_count=pending,
        total_chunks=total_chunks,
        embedding_model="BAAI/bge-base-en-v1.5",
    )


@router.get("/worker-status", response_model=WorkerStatusResponse)
def get_catalog_worker_status(
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Real-time inspection of Celery workers, queue depth, active jobs, and database completion stats."""
    from app.core.config import get_settings
    settings = get_settings()

    workers_info: list[WorkerNodeInfo] = []
    queue_depths = QueueDepthInfo()
    active_jobs_list: list[ActiveJobInfo] = []
    redis_connected = False

    # 1. Redis connectivity & Queue lengths
    if settings.redis_url:
        try:
            import redis
            r = redis.from_url(settings.redis_url, socket_timeout=1.0)
            r.ping()
            redis_connected = True
            queue_depths.catalog_ingest = r.llen("catalog-ingest") or 0
            queue_depths.lesson_generate = r.llen("lesson-generate") or 0
            queue_depths.mail = r.llen("mail") or 0
        except Exception as r_err:
            logger.warning("Redis ping in worker-status failed: %s", r_err)

    # 2. Celery Worker Inspector
    overall_status = "OFFLINE"
    if redis_connected:
        try:
            from app.core.celery_app import celery_app
            inspector = celery_app.control.inspect(timeout=0.6)
            ping_res = inspector.ping() or {}
            active_tasks = inspector.active() or {}
            stats = inspector.stats() or {}
            active_queues = inspector.active_queues() or {}

            for worker_name, ping_data in ping_res.items():
                w_stats = stats.get(worker_name, {})
                w_active = active_tasks.get(worker_name, [])
                raw_q = active_queues.get(worker_name, [])
                q_names = [q.get("name", "") for q in raw_q if isinstance(q, dict)]

                total_processed = 0
                if isinstance(w_stats.get("total"), dict):
                    total_processed = sum(w_stats["total"].values())

                workers_info.append(
                    WorkerNodeInfo(
                        name=worker_name,
                        status="online" if ping_data.get("ok") == "pong" else "degraded",
                        concurrency=w_stats.get("pool", {}).get("max-concurrency", 1),
                        active_tasks_count=len(w_active),
                        processed_total=total_processed,
                        active_queues=q_names,
                    )
                )

                for t in w_active:
                    args_str = [str(a) for a in t.get("args", [])]
                    target_label = f"Upload #{args_str[0]}" if args_str else ""
                    active_jobs_list.append(
                        ActiveJobInfo(
                            task_id=str(t.get("id", "")),
                            task_name=str(t.get("name", "")),
                            args=args_str,
                            worker=worker_name,
                            time_start=t.get("time_start"),
                            target_label=target_label,
                        )
                    )

            if workers_info:
                overall_status = "HEALTHY"
            else:
                overall_status = "DEGRADED"
        except Exception as c_err:
            logger.warning("Celery inspect in worker-status failed: %s", c_err)
            overall_status = "DEGRADED"

    # 3. Database aggregation stats
    total = db.scalar(select(func.count()).select_from(TextbookUpload)) or 0
    embedded = db.scalar(
        select(func.count()).select_from(TextbookUpload).where(TextbookUpload.embedding_status == ProcessingStatusEnum.EMBEDDED)
    ) or 0
    failed = db.scalar(
        select(func.count()).select_from(TextbookUpload).where(TextbookUpload.embedding_status == ProcessingStatusEnum.FAILED)
    ) or 0
    processing = db.scalar(
        select(func.count()).select_from(TextbookUpload).where(
            (TextbookUpload.embedding_status == ProcessingStatusEnum.PROCESSING) | 
            (TextbookUpload.chunk_status == ProcessingStatusEnum.PROCESSING)
        )
    ) or 0
    queued = db.scalar(
        select(func.count()).select_from(TextbookUpload).where(
            (TextbookUpload.embedding_status == ProcessingStatusEnum.QUEUED) | 
            (TextbookUpload.chunk_status == ProcessingStatusEnum.QUEUED)
        )
    ) or 0
    total_chunks = db.scalar(select(func.sum(TextbookUpload.chunk_count)).select_from(TextbookUpload)) or 0

    # 4. Recent 10 processes
    recent_rows = list(
        db.scalars(
            select(TextbookUpload)
            .order_by(TextbookUpload.updated_at.desc())
            .limit(10)
        )
    )
    recent_processes = [
        RecentProcessInfo(
            id=r.id,
            file_name=r.file_name,
            board=str(r.board.value if hasattr(r.board, "value") else r.board),
            class_level=str(r.class_level.value if hasattr(r.class_level, "value") else r.class_level),
            subject_name=r.subject_name,
            chunk_count=r.chunk_count or 0,
            embedding_status=str(r.embedding_status.value if hasattr(r.embedding_status, "value") else r.embedding_status),
            updated_at=r.updated_at,
        )
        for r in recent_rows
    ]

    return WorkerStatusResponse(
        status=overall_status,
        redis_connected=redis_connected,
        workers=workers_info,
        queue_depths=queue_depths,
        active_jobs=active_jobs_list,
        total_documents=total,
        embedded_count=embedded,
        failed_count=failed,
        processing_count=processing,
        queued_count=queued,
        total_chunks=total_chunks,
        recent_processes=recent_processes,
        vector_backend=getattr(settings, "vector_backend", os.environ.get("VECTOR_BACKEND", "qdrant")),
        storage_backend=getattr(settings, "storage_backend", os.environ.get("STORAGE_BACKEND", "s3")),
    )


@router.post("/retry-all-failed", response_model=MessageResponse)
def retry_all_failed_uploads(
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Re-queues all failed textbook uploads for background processing."""
    failed_rows = list(
        db.scalars(
            select(TextbookUpload).where(
                (TextbookUpload.embedding_status == ProcessingStatusEnum.FAILED) |
                (TextbookUpload.chunk_status == ProcessingStatusEnum.FAILED)
            )
        )
    )
    if not failed_rows:
        return MessageResponse(message="No failed uploads found.")

    from app.services.catalog_pipeline import dispatch_upload_processing

    requeued_count = 0
    for r in failed_rows:
        r.chunk_status = ProcessingStatusEnum.QUEUED
        r.embedding_status = ProcessingStatusEnum.QUEUED
        r.ocr_status = ProcessingStatusEnum.QUEUED
        dispatch_upload_processing(r.id, background_tasks)
        requeued_count += 1
    db.commit()

    return MessageResponse(message=f"Successfully re-queued {requeued_count} failed upload(s) to worker.")


@router.patch("/textbook-uploads/{upload_id}/status", response_model=TextbookUploadResponse)
def patch_textbook_upload_status(
    upload_id: int,
    payload: TextbookUploadPatchStatusRequest,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    row = db.get(TextbookUpload, upload_id)
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found.")
    row.ocr_status = payload.ocr_status
    row.chunk_status = payload.chunk_status
    row.embedding_status = payload.embedding_status
    db.commit()
    db.refresh(row)
    return TextbookUploadResponse.model_validate(row)


@router.delete("/textbook-uploads/{upload_id}", response_model=MessageResponse)
def delete_textbook_upload(
    upload_id: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    row = db.get(TextbookUpload, upload_id)
    if not row:
        return MessageResponse(message="Upload not found.")

    from app.services.image_service.textbook_image_extraction import purge_textbook_images_disk_and_rows

    purge_textbook_images_disk_and_rows(db, upload_id)
    db.commit()

    # Purge vectors from ChromaDB BEFORE deleting the DB row so the chapter
    # is no longer searchable once it is removed from the catalog.
    from app.services.vector_service import delete_collection_docs
    collection = f"{row.board}_{row.class_level}_{row.subject_name}".replace(" ", "_")
    try:
        delete_collection_docs(
            collection,
            where_filter={"textbook_upload_id": str(row.id)},
        )
        logger.info("Deleted vectors for upload %s from collection '%s'", upload_id, collection)
    except Exception as exc:
        logger.warning(
            "Could not purge vectors for upload %s (collection=%r): %s",
            upload_id, collection, exc,
        )

    # Lesson planner keeps its own copies (question bank / figure captions / experiments) per chapter.
    from app.services.lesson_planner.retrieval.chroma_store import (
        COLLECTION_QUESTION_BANK,
        COLLECTION_TEXTBOOK_EXPERIMENTS,
        COLLECTION_TEXTBOOK_IMAGES,
    )

    for base in (COLLECTION_QUESTION_BANK, COLLECTION_TEXTBOOK_IMAGES, COLLECTION_TEXTBOOK_EXPERIMENTS):
        delete_collection_docs(f"{base}_{collection}", where_filter={"textbook_upload_id": str(row.id)})

    if not file_still_referenced(db, row.file_path, exclude_upload_id=row.id):
        try:
            from app.services.image_service.storage_backend import delete_textbook_file

            delete_textbook_file(row.file_path)
        except Exception as exc:
            logger.warning("Could not delete source file for upload %s: %s", upload_id, exc)

    db.execute(
        delete(TextbookChapter).where(
            TextbookChapter.textbook_upload_id == upload_id,
            TextbookChapter.detection_method == MANUAL_UPLOAD,
        )
    )
    db.delete(row)
    db.commit()
    return MessageResponse(message="Upload and associated vectors deleted.")


@router.put("/textbook-uploads/{upload_id}", response_model=TextbookUploadResponse)
def update_textbook_upload(
    upload_id: int,
    payload: TextbookUploadUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Update metadata for an uploaded chapter or document."""
    row = db.get(TextbookUpload, upload_id)
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found.")
    if payload.chapter is not None:
        row.chapter = clean_display_label(payload.chapter)
    if payload.content_label is not None:
        row.content_label = clean_display_label(payload.content_label)
    if payload.content_type is not None:
        row.content_type = payload.content_type
    db.commit()
    db.refresh(row)
    return TextbookUploadResponse.model_validate(row)


@router.get("/textbook-uploads/{upload_id}/file")
def get_textbook_upload_file(
    upload_id: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_user_bearer_or_query)],
):
    """Stream the raw PDF or doc file for inline browser previewing or download."""
    row = db.get(TextbookUpload, upload_id)
    if not row or not row.file_path:
        raise HTTPException(status_code=404, detail="Upload or file not found.")

    local_path = _local_path_for_upload(row.file_path)
    if not local_path or not os.path.isfile(local_path):
        raise HTTPException(status_code=404, detail="File could not be found on storage.")

    ext = os.path.splitext(local_path)[1].lower()
    media_type = "application/pdf"
    if ext in [".doc", ".docx"]:
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif ext == ".txt":
        media_type = "text/plain"

    return FileResponse(
        path=local_path,
        media_type=media_type,
        filename=os.path.basename(row.file_name or local_path),
        content_disposition_type="inline",
    )


def _serialize_textbook_image(im: TextbookImage) -> TextbookImageAdminResponse:
    url = f"/auth/catalog/textbook-images/{im.textbook_upload_id}/{im.file_name}"
    return TextbookImageAdminResponse(
        id=im.id,
        textbook_upload_id=im.textbook_upload_id,
        page_index=im.page_index,
        sequence=im.sequence,
        file_name=im.file_name,
        image_url=url,
        caption=im.caption,
        caption_normalized=im.caption_normalized,
        image_type=im.image_type or "unknown",
        figure_number=im.figure_number,
        title=im.title,
        educational_role=im.educational_role or "unknown",
        educational_description=im.educational_description,
        is_decorative=bool(im.is_decorative),
        image_bbox=im.image_bbox,
        structured_content=im.structured_content,
        content_kind=im.content_kind or "figure",
        created_at=im.created_at,
    )


@router.get("/textbook-uploads/{upload_id}/images", response_model=list[TextbookImageAdminResponse])
def list_upload_images(
    upload_id: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """List all extracted images and figures for a textbook upload."""
    upload = db.get(TextbookUpload, upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found.")

    images = list(
        db.scalars(
            select(TextbookImage)
            .where(TextbookImage.textbook_upload_id == upload_id)
            .order_by(TextbookImage.page_index, TextbookImage.sequence)
        )
    )
    return [_serialize_textbook_image(im) for im in images]


@router.get("/textbook-uploads/{upload_id}/pdf-pages-info")
def get_upload_pdf_pages_info(
    upload_id: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Return total pages and file name for direct PDF page cropping."""
    upload = db.get(TextbookUpload, upload_id)
    if not upload or not upload.file_path:
        raise HTTPException(status_code=404, detail="Upload or file not found.")

    local_path = _local_path_for_upload(upload.file_path)
    if not local_path or not os.path.isfile(local_path):
        raise HTTPException(status_code=404, detail="Source file not found on disk.")

    ext = os.path.splitext(local_path)[1].lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files support direct page rendering.")

    import fitz
    try:
        doc = fitz.open(local_path)
        total_pages = len(doc)
        doc.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF pages: {exc}")

    return {
        "upload_id": upload_id,
        "total_pages": total_pages,
        "file_name": upload.file_name,
    }


@router.get("/textbook-uploads/{upload_id}/pdf-page/{page_index}")
def get_upload_pdf_page_image(
    upload_id: int,
    page_index: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_user_bearer_or_query)],
    dpi: int = 150,
):
    """Render a specific PDF page as a crisp JPEG for direct canvas cropping."""
    upload = db.get(TextbookUpload, upload_id)
    if not upload or not upload.file_path:
        raise HTTPException(status_code=404, detail="Upload or file not found.")

    local_path = _local_path_for_upload(upload.file_path)
    if not local_path or not os.path.isfile(local_path):
        raise HTTPException(status_code=404, detail="Source file not found on disk.")

    ext = os.path.splitext(local_path)[1].lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files support direct page rendering.")

    import fitz
    try:
        doc = fitz.open(local_path)
        if page_index < 0 or page_index >= len(doc):
            doc.close()
            raise HTTPException(status_code=400, detail=f"Invalid page index {page_index}. Range: 0 to {len(doc) - 1}")
        page = doc[page_index]
        effective_dpi = max(72, min(240, dpi))
        pix = page.get_pixmap(dpi=effective_dpi)
        img_bytes = pix.tobytes("jpeg")
        doc.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render PDF page: {exc}")

    return Response(
        content=img_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.post("/textbook-uploads/{upload_id}/images", response_model=TextbookImageAdminResponse, status_code=status.HTTP_201_CREATED)
async def create_textbook_image(
    upload_id: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    file: UploadFile = File(...),
    caption: str | None = Form(None),
    figure_number: str | None = Form(None),
    page_index: int = Form(0),
    image_type: str = Form("diagram"),
    educational_role: str = Form("primary_concept"),
    title: str | None = Form(None),
):
    """Add a new figure manually cropped from the PDF or uploaded by the administrator."""
    import hashlib
    import time
    from io import BytesIO
    from PIL import Image
    from app.services.image_service.storage_backend import get_storage_backend
    from app.services.image_service.textbook_image_extraction import (
        image_disk_path,
        image_storage_key,
        normalize_caption,
        compute_educational_salience,
    )

    upload = db.get(TextbookUpload, upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found.")

    raw_bytes = await file.read()
    try:
        img = Image.open(BytesIO(raw_bytes))
        img.verify()
        img = Image.open(BytesIO(raw_bytes))
        out_io = BytesIO()
        if img.mode in ("RGBA", "P"):
            img.save(out_io, format="PNG")
            ext = ".png"
        else:
            img.save(out_io, format="JPEG", quality=95)
            ext = ".jpg"
        final_bytes = out_io.getvalue()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    ts = int(time.time() * 1000)
    file_name = f"figures/fig_manual_p{page_index + 1}_{ts}{ext}"

    # Save to storage backend
    backend = get_storage_backend()
    storage_key = image_storage_key(upload, file_name)
    backend.save(storage_key, final_bytes)

    # Save to disk cache
    local_p = image_disk_path(upload_id, file_name, upload=upload)
    os.makedirs(os.path.dirname(local_p), exist_ok=True)
    with open(local_p, "wb") as f:
        f.write(final_bytes)

    content_hash = hashlib.sha256(final_bytes).hexdigest()[:16]
    clean_cap = caption.strip() if caption else None

    # Get max sequence for page
    curr_seq = db.scalar(
        select(func.coalesce(func.max(TextbookImage.sequence), 0)).where(
            TextbookImage.textbook_upload_id == upload_id,
            TextbookImage.page_index == page_index,
        )
    ) or 0

    im = TextbookImage(
        textbook_upload_id=upload_id,
        page_index=page_index,
        sequence=curr_seq + 1,
        file_name=file_name,
        caption=clean_cap,
        caption_normalized=normalize_caption(clean_cap) if clean_cap else None,
        image_type=image_type.strip(),
        figure_number=figure_number.strip() if figure_number else None,
        title=title.strip() if title else None,
        educational_role=educational_role.strip(),
        educational_salience=compute_educational_salience(clean_cap) if clean_cap else 0.5,
        content_hash=content_hash,
        multimodal_indexed=False,
    )
    db.add(im)
    db.commit()
    db.refresh(im)

    # Re-index if CLIP available
    try:
        from app.services.image_service.image_vector_store import (
            image_collection_name,
            subject_collection_from_upload,
            upsert_image_vectors,
        )
        from app.services.image_service.multimodal_encoder import (
            clip_model_available,
            current_model_name,
            encode_image_file,
        )

        if clip_model_available() and upload:
            vec = encode_image_file(local_p)
            if vec is not None:
                coll = image_collection_name(
                    subject_collection_from_upload(
                        str(upload.board),
                        str(upload.class_level),
                        upload.subject_name,
                    )
                )
                upsert_image_vectors(
                    coll,
                    upload_id=str(upload.id),
                    items=[
                        {
                            "image_id": str(im.id),
                            "embedding": vec,
                            "page_index": im.page_index,
                            "file_name": im.file_name,
                        }
                    ],
                )
                im.multimodal_indexed = True
                im.embedding_model = current_model_name()
                db.commit()
                db.refresh(im)
    except Exception as exc:
        logger.warning("Could not index manual figure %s: %s", im.id, exc)

    return _serialize_textbook_image(im)


@router.patch("/textbook-images/{image_id}", response_model=TextbookImageAdminResponse)
def update_textbook_image(
    image_id: int,
    payload: TextbookImageUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Update metadata (caption, title, figure number, role, type) for an extracted figure."""
    from app.services.image_service.textbook_image_extraction import (
        compute_educational_salience,
        normalize_caption,
    )

    im = db.get(TextbookImage, image_id)
    if not im:
        raise HTTPException(status_code=404, detail="Image not found.")

    if payload.caption is not None:
        raw_cap = payload.caption.strip()
        im.caption = raw_cap if raw_cap else None
        im.caption_normalized = normalize_caption(raw_cap) if raw_cap else None
        im.educational_salience = compute_educational_salience(raw_cap)
        im.has_caption = bool(raw_cap)
        im.multimodal_indexed = False

    if payload.title is not None:
        raw_title = payload.title.strip()
        im.title = raw_title if raw_title else None

    if payload.figure_number is not None:
        raw_fn = payload.figure_number.strip()
        im.figure_number = raw_fn if raw_fn else None

    if payload.image_type is not None:
        im.image_type = payload.image_type.strip()

    if payload.educational_role is not None:
        im.educational_role = payload.educational_role.strip()

    if payload.is_decorative is not None:
        im.is_decorative = payload.is_decorative

    if payload.educational_description is not None:
        raw_desc = payload.educational_description.strip()
        im.educational_description = raw_desc if raw_desc else None

    db.commit()
    db.refresh(im)
    return _serialize_textbook_image(im)


@router.delete("/textbook-images/{image_id}", response_model=MessageResponse)
def delete_textbook_image(
    image_id: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Permanently delete an extracted figure from database, storage, and vector store."""
    from app.services.image_service.image_vector_store import (
        delete_single_image_vector,
        image_collection_name,
        subject_collection_from_upload,
    )
    from app.services.image_service.storage_backend import get_storage_backend
    from app.services.image_service.textbook_image_extraction import (
        image_disk_path,
        image_storage_key,
    )

    im = db.get(TextbookImage, image_id)
    if not im:
        raise HTTPException(status_code=404, detail="Image not found.")

    upload = db.get(TextbookUpload, im.textbook_upload_id)

    # 1. Purge vector from Chroma/Qdrant
    if upload:
        try:
            coll = image_collection_name(
                subject_collection_from_upload(
                    str(upload.board),
                    str(upload.class_level),
                    upload.subject_name,
                )
            )
            delete_single_image_vector(coll, str(im.id))
        except Exception as exc:
            logger.warning("Could not delete vector for image %s: %s", im.id, exc)

    # 2. Delete physical file from storage backend and disk
    try:
        backend = get_storage_backend()
        storage_key = image_storage_key(upload or im.textbook_upload_id, im.file_name)
        backend.delete(storage_key)
        disk_p = image_disk_path(im.textbook_upload_id, im.file_name, upload=upload)
        if os.path.isfile(disk_p):
            try:
                os.remove(disk_p)
            except OSError:
                pass
    except Exception as exc:
        logger.warning("Could not delete file for image %s: %s", im.id, exc)

    # 3. Delete database row
    db.delete(im)
    db.commit()
    return MessageResponse(message="Image deleted successfully.")


@router.post("/textbook-images/{image_id}/crop", response_model=TextbookImageAdminResponse)
async def crop_textbook_image(
    image_id: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    file: UploadFile | None = File(None),
    x: float | None = Form(None),
    y: float | None = Form(None),
    width: float | None = Form(None),
    height: float | None = Form(None),
    rotate: int = Form(0),
    page_index: int | None = Form(None),
):
    """Crop, resize, or reshape an existing textbook figure."""
    import hashlib
    from io import BytesIO
    from PIL import Image
    from app.services.image_service.storage_backend import get_storage_backend
    from app.services.image_service.textbook_image_extraction import (
        image_disk_path,
        image_storage_key,
    )

    im = db.get(TextbookImage, image_id)
    if not im:
        raise HTTPException(status_code=404, detail="Image not found.")

    upload = db.get(TextbookUpload, im.textbook_upload_id)
    final_bytes: bytes

    if file is not None:
        raw_bytes = await file.read()
        try:
            img = Image.open(BytesIO(raw_bytes))
            img.verify()
            img = Image.open(BytesIO(raw_bytes))
            out_io = BytesIO()
            is_png = im.file_name.lower().endswith(".png")
            fmt = "PNG" if is_png else "JPEG"
            if fmt == "JPEG" and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(out_io, format=fmt, quality=95)
            final_bytes = out_io.getvalue()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image uploaded: {exc}")
    elif width is not None and height is not None:
        local_p = image_disk_path(im.textbook_upload_id, im.file_name, upload=upload)
        if not os.path.isfile(local_p):
            raise HTTPException(status_code=404, detail="Original image not found on disk.")
        try:
            img = Image.open(local_p)
            if rotate % 360 != 0:
                img = img.rotate(-rotate, expand=True)
            w_img, h_img = img.size
            b_left = max(0, min(w_img - 1, int(x or 0)))
            b_top = max(0, min(h_img - 1, int(y or 0)))
            b_right = max(b_left + 1, min(w_img, int(b_left + width)))
            b_bottom = max(b_top + 1, min(h_img, int(b_top + height)))
            cropped = img.crop((b_left, b_top, b_right, b_bottom))
            out_io = BytesIO()
            is_png = im.file_name.lower().endswith(".png")
            fmt = "PNG" if is_png else "JPEG"
            if fmt == "JPEG" and cropped.mode in ("RGBA", "P"):
                cropped = cropped.convert("RGB")
            cropped.save(out_io, format=fmt, quality=95)
            final_bytes = out_io.getvalue()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to crop image: {exc}")
    else:
        raise HTTPException(status_code=400, detail="Provide either a cropped image file or crop coordinates.")

    # Save new image bytes
    backend = get_storage_backend()
    storage_key = image_storage_key(upload or im.textbook_upload_id, im.file_name)
    backend.save(storage_key, final_bytes)

    # Overwrite local disk cache
    local_p = image_disk_path(im.textbook_upload_id, im.file_name, upload=upload)
    os.makedirs(os.path.dirname(local_p), exist_ok=True)
    with open(local_p, "wb") as f:
        f.write(final_bytes)

    im.content_hash = hashlib.sha256(final_bytes).hexdigest()[:16]
    im.multimodal_indexed = False
    if page_index is not None and page_index >= 0:
        im.page_index = page_index

    # Re-index if CLIP available
    try:
        from app.services.image_service.image_vector_store import (
            image_collection_name,
            subject_collection_from_upload,
            upsert_image_vectors,
        )
        from app.services.image_service.multimodal_encoder import (
            clip_model_available,
            current_model_name,
            encode_image_file,
        )

        if clip_model_available() and upload:
            vec = encode_image_file(local_p)
            if vec is not None:
                coll = image_collection_name(
                    subject_collection_from_upload(
                        str(upload.board),
                        str(upload.class_level),
                        upload.subject_name,
                    )
                )
                upsert_image_vectors(
                    coll,
                    upload_id=str(upload.id),
                    items=[
                        {
                            "image_id": str(im.id),
                            "embedding": vec,
                            "page_index": im.page_index,
                            "file_name": im.file_name,
                        }
                    ],
                )
                im.multimodal_indexed = True
                im.embedding_model = current_model_name()
    except Exception as exc:
        logger.warning("Could not re-index cropped image %s: %s", im.id, exc)

    db.commit()
    db.refresh(im)
    return _serialize_textbook_image(im)


student_router = APIRouter(prefix="/auth/catalog", tags=["student-catalog"])


@student_router.get("/textbook-images/{upload_id}/{file_path:path}")
def get_textbook_image_file(
    upload_id: int,
    file_path: str,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_user_bearer_or_query)],
):
    """Serve an extracted textbook asset (figures, tables, or formulas)."""
    from app.services.image_service.textbook_image_extraction import (
        image_disk_path,
        image_storage_key,
    )

    safe = file_path.strip().replace("\\", "/").lstrip("/")
    if not safe or ".." in safe.split("/"):
        raise HTTPException(status_code=400, detail=IMAGE_INVALID_PATH)
    allowed_prefixes = ("figures/", "tables/", "formulas/")
    if "/" in safe and not safe.startswith(allowed_prefixes):
        raise HTTPException(status_code=400, detail=IMAGE_INVALID_PATH)

    basename = os.path.basename(safe)
    row = db.scalar(
        select(TextbookImage).where(
            TextbookImage.textbook_upload_id == upload_id,
            TextbookImage.file_name.in_([safe, basename]),
        )
    )
    if row is None:
        raise HTTPException(status_code=404, detail=IMAGE_NOT_AVAILABLE)

    upload = db.get(TextbookUpload, upload_id)

    # Prefer CDN redirect when configured (stable URLs; avoid short-lived presigns).
    try:
        from app.config import CDN_BASE_URL, STORAGE_BACKEND
        from fastapi.responses import RedirectResponse

        if upload is not None and (STORAGE_BACKEND or "").lower() == "s3" and (CDN_BASE_URL or "").strip():
            from app.services.image_service.storage_backend import get_storage_backend

            return RedirectResponse(
                get_storage_backend().get_url(image_storage_key(upload, row.file_name)),
                status_code=302,
            )
    except Exception:
        pass

    path = image_disk_path(upload_id, row.file_name, upload=upload)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=IMAGE_NOT_AVAILABLE)

    from app.services.image_service.textbook_image_display import can_serve_file_directly, encode_browser_jpeg

    if can_serve_file_directly(path):
        return FileResponse(path, media_type="image/jpeg", filename=basename)

    body = encode_browser_jpeg(path)
    if not body:
        raise HTTPException(status_code=404, detail=IMAGE_NOT_AVAILABLE)

    return Response(content=body, media_type="image/jpeg")


def _parse_class_level(raw: str | None) -> ClassEnum | None:
    if not raw or not str(raw).strip():
        return None
    grade_str = str(raw).strip()
    canon = f"CLASS_{grade_str}" if grade_str.isdigit() else grade_str
    try:
        return ClassEnum(canon)
    except ValueError:
        return None


def _user_may_access_class(user: User, class_level: ClassEnum) -> bool:
    if user.role in (Role.MASTER_ADMIN, Role.SCHOOL_ADMIN):
        return True
    grade_num = class_level.value.replace("CLASS_", "")
    for entry in user.teaching_classes or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("grade", "")).strip() == grade_num:
            return True
    return False


def _resolve_student_board_class(
    user: User,
    *,
    class_level_param: str | None = None,
    board_param: str | None = None,
) -> tuple[BoardEnum | None, ClassEnum | None]:
    board = None
    candidates = [
        str(board_param).strip() if board_param else "",
    ]
    # Prefer curriculum from teaching_classes over corrupted teaching_board (subject names).
    raw = user.teaching_classes if isinstance(user.teaching_classes, list) else []
    for item in raw:
        if isinstance(item, dict):
            cur = str(item.get("curriculum") or "").strip()
            if cur:
                candidates.append(cur)
                break
    if user.teaching_board and str(user.teaching_board).strip():
        candidates.append(str(user.teaching_board).strip())

    for label in candidates:
        if not label:
            continue
        try:
            board = BoardEnum(label)
            break
        except ValueError:
            continue

    class_level = _parse_class_level(class_level_param)
    if class_level is None:
        grades = user.teaching_classes or []
        if grades:
            grade_str = grades[0].get("grade", "") if isinstance(grades[0], dict) else ""
            class_level = _parse_class_level(grade_str)

    if class_level and not _user_may_access_class(user, class_level):
        return board, None

    return board, class_level


@student_router.get("/my-subjects", response_model=list[StudentSubjectResponse])
def list_my_subjects(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    class_level: str | None = None,
    board: str | None = None,
):
    """Students: class-mapped subjects. Tutors/admins: full syllabus for board+class."""
    board_enum, class_enum = _resolve_student_board_class(
        current_user,
        class_level_param=class_level,
        board_param=board,
    )
    if not board_enum or not class_enum:
        return []

    if current_user.role == Role.STUDENT:
        from app.modules.student_learning.enrollment import list_enrolled_subjects

        subjects, _scope = list_enrolled_subjects(
            db,
            current_user,
            board=board_enum,
            class_level=class_enum,
        )
        return subjects

    subjects = list(
        db.scalars(
            select(SyllabusSubject)
            .where(SyllabusSubject.board == board_enum, SyllabusSubject.class_level == class_enum)
            .order_by(SyllabusSubject.subject_name)
        )
    )

    textbooks = list(
        db.scalars(
            select(TextbookUpload)
            .where(TextbookUpload.board == board_enum, TextbookUpload.class_level == class_enum)
            .order_by(TextbookUpload.chapter)
        )
    )
    tb_by_subject: dict[str, list[TextbookUpload]] = {}
    for tb in textbooks:
        tb_by_subject.setdefault(tb.subject_name, []).append(tb)

    from app.modules.student_learning.enrollment import _chapter_sort_key

    for k in tb_by_subject:
        tb_by_subject[k].sort(key=_chapter_sort_key)

    result: list[StudentSubjectResponse] = []
    for subj in subjects:
        chapters = [
            StudentChapterResponse(id=tb.id, chapter=tb.chapter, file_name=tb.file_name)
            for tb in tb_by_subject.get(subj.subject_name, [])
        ]
        result.append(
            StudentSubjectResponse(
                id=subj.id,
                board=subj.board,
                class_level=subj.class_level,
                subject_name=subj.subject_name,
                chapters=chapters,
            )
        )

    return result
