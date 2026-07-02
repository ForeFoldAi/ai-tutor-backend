import asyncio
import logging
import os
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import UPLOADS_DIR
from app.core.database import SessionLocal, get_db
from app.core.student_messages import IMAGE_INVALID_PATH, IMAGE_NOT_AVAILABLE, PDF_ONLY
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import get_current_user, get_current_user_bearer_or_query, require_roles
from app.modules.auth.schemas import MessageResponse
from app.modules.catalog.models import (
    BoardDefinition,
    BoardEnum,
    ClassEnum,
    ProcessingStatusEnum,
    SyllabusSubject,
    TextbookImage,
    TextbookUpload,
)
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
    TextbookUploadCreateRequest,
    TextbookUploadPatchStatusRequest,
    TextbookUploadResponse,
)
from app.modules.users.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/admin/catalog", tags=["master-admin-catalog"])


@router.get("/enums", response_model=CatalogEnumsResponse)
def list_catalog_enums(
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN, Role.ORG_ADMIN))],
):
    return CatalogEnumsResponse(boards=[b for b in BoardEnum], classes=[c for c in ClassEnum])


@router.get("/boards", response_model=list[BoardResponse])
def list_boards(
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN, Role.ORG_ADMIN))],
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
    board_id: uuid.UUID,
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
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN, Role.ORG_ADMIN))],
):
    rows = list(db.scalars(select(SyllabusSubject).order_by(SyllabusSubject.board, SyllabusSubject.class_level, SyllabusSubject.subject_name)))
    return [SyllabusSubjectResponse.model_validate(r) for r in rows]


@router.post("/syllabus", response_model=list[SyllabusSubjectResponse], status_code=status.HTTP_201_CREATED)
def create_syllabus_bulk(
    payload: SyllabusBulkCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN, Role.ORG_ADMIN))],
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


@router.get("/textbook-uploads", response_model=list[TextbookUploadResponse])
def list_textbook_uploads(
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN, Role.ORG_ADMIN))],
):
    rows = list(db.scalars(select(TextbookUpload).order_by(TextbookUpload.upload_date.desc())))
    return [TextbookUploadResponse.model_validate(r) for r in rows]


@router.post("/textbook-uploads", response_model=TextbookUploadResponse, status_code=status.HTTP_201_CREATED)
def create_textbook_upload(
    payload: TextbookUploadCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN, Role.ORG_ADMIN))],
):
    row = TextbookUpload(
        file_name=payload.file_name,
        board=payload.board,
        class_level=payload.class_name,
        subject_name=payload.subject,
        chapter=payload.chapter,
        content_type=payload.content_type,
        content_label=payload.content_label,
        uploaded_by=current_user.id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return TextbookUploadResponse.model_validate(row)


def _write_file_sync(path: str, content: bytes) -> None:
    """Pure synchronous write — run inside asyncio.to_thread to avoid blocking."""
    with open(path, "wb") as f:
        f.write(content)


async def _save_upload_file(file: UploadFile, board: str, class_level: str, subject: str) -> str:
    """
    Save an uploaded file without blocking the event loop.

    Reads the upload fully into memory (safe for typical textbook PDFs < 50 MB),
    then offloads the disk write to a thread pool via asyncio.to_thread().
    """
    dest_dir = os.path.join(UPLOADS_DIR, board, class_level, subject)
    os.makedirs(dest_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename or 'document'}"
    dest_path = os.path.join(dest_dir, safe_name)
    content = await file.read()
    await asyncio.to_thread(_write_file_sync, dest_path, content)
    return dest_path


def _process_upload_background(upload_id: uuid.UUID) -> None:
    """Background task: chunk the document and create embeddings."""
    db = SessionLocal()
    try:
        row = db.get(TextbookUpload, upload_id)
        if not row or not row.file_path:
            return

        row.chunk_status = ProcessingStatusEnum.PROCESSING
        row.embedding_status = ProcessingStatusEnum.PROCESSING
        db.commit()

        from app.services.document_service import process_document
        from app.services.vector_service import add_documents_to_store

        extra_meta = {
            "board": str(row.board),
            "class_level": str(row.class_level),
            "subject_name": row.subject_name,
            "textbook_upload_id": str(row.id),
            "content_type": row.content_type or "",
            "content_label": row.content_label or "",
        }
        chunks = process_document(row.file_path, extra_metadata=extra_meta)
        row.chunk_count = len(chunks)
        row.chunk_status = ProcessingStatusEnum.EMBEDDED
        db.commit()

        collection = f"{row.board}_{row.class_level}_{row.subject_name}".replace(" ", "_")
        added = add_documents_to_store(chunks, collection_name=collection)

        if added > 0:
            row.embedding_status = ProcessingStatusEnum.EMBEDDED
        else:
            row.embedding_status = ProcessingStatusEnum.FAILED
        row.ocr_status = ProcessingStatusEnum.EMBEDDED
        db.commit()
        try:
            from app.services.image_service.textbook_image_extraction import replace_all_images_after_reprocess

            replace_all_images_after_reprocess(db, row)
        except Exception as img_exc:
            logger.warning("Image extraction after upload %s: %s", upload_id, img_exc)
        logger.info("Processed upload %s: %d chunks, %d embedded", upload_id, len(chunks), added)
    except Exception as exc:
        logger.exception("Failed to process upload %s: %s", upload_id, exc)
        try:
            row = db.get(TextbookUpload, upload_id)
            if row:
                row.chunk_status = ProcessingStatusEnum.FAILED
                row.embedding_status = ProcessingStatusEnum.FAILED
                db.commit()
        except Exception:
            pass
    finally:
        db.close()


@router.post("/textbook-uploads/upload", response_model=list[TextbookUploadResponse], status_code=status.HTTP_201_CREATED)
async def upload_textbook_files(
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN, Role.ORG_ADMIN))],
    files: list[UploadFile] = File(...),
    board: str = Form(...),
    class_name: str = Form(...),
    subject: str = Form(...),
    content_type: str = Form("CHAPTER"),
    content_labels: str = Form(""),
):
    """Upload one or more PDF/Word files for a subject.

    ``content_labels`` is a comma-separated list aligned to ``files``.
    If fewer labels than files are provided the file name is used as the label.
    """
    allowed_ext = {".pdf", ".doc", ".docx"}
    labels = [l.strip() for l in content_labels.split(",")]

    results: list[TextbookUploadResponse] = []
    for idx, file in enumerate(files):
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in allowed_ext:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_ext)}",
            )

        file_path = await _save_upload_file(file, board, class_name, subject)
        label = labels[idx] if idx < len(labels) and labels[idx] else (file.filename or f"File {idx + 1}")

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
        db.commit()
        db.refresh(row)
        results.append(TextbookUploadResponse.model_validate(row))

        background_tasks.add_task(_process_upload_background, row.id)

    return results


@router.post("/textbook-uploads/{upload_id}/process", response_model=ProcessResponse)
def process_textbook_upload(
    upload_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN, Role.ORG_ADMIN))],
):
    """Manually trigger (re)processing for a single upload."""
    row = db.get(TextbookUpload, upload_id)
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found.")
    if not row.file_path or not os.path.isfile(row.file_path):
        raise HTTPException(status_code=400, detail="Source file not found on disk.")

    row.chunk_status = ProcessingStatusEnum.QUEUED
    row.embedding_status = ProcessingStatusEnum.QUEUED
    row.ocr_status = ProcessingStatusEnum.QUEUED
    db.commit()
    db.refresh(row)

    background_tasks.add_task(_process_upload_background, row.id)

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
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN, Role.ORG_ADMIN))],
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


@router.patch("/textbook-uploads/{upload_id}/status", response_model=TextbookUploadResponse)
def patch_textbook_upload_status(
    upload_id: uuid.UUID,
    payload: TextbookUploadPatchStatusRequest,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN, Role.ORG_ADMIN))],
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
    upload_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN, Role.ORG_ADMIN))],
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

    db.delete(row)
    db.commit()
    return MessageResponse(message="Upload and associated vectors deleted.")


student_router = APIRouter(prefix="/auth/catalog", tags=["student-catalog"])


@student_router.get("/textbook-images/{upload_id}/{file_path:path}")
def get_textbook_image_file(
    upload_id: uuid.UUID,
    file_path: str,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_user_bearer_or_query)],
):
    """Serve an extracted textbook asset (figures, tables, or formulas)."""
    from app.services.image_service.textbook_image_extraction import image_disk_path

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

    path = image_disk_path(upload_id, row.file_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=IMAGE_NOT_AVAILABLE)

    from app.services.image_service.textbook_image_display import can_serve_file_directly, encode_browser_jpeg

    if can_serve_file_directly(path):
        return FileResponse(path, media_type="image/jpeg", filename=basename)

    body = encode_browser_jpeg(path)
    if not body:
        raise HTTPException(status_code=404, detail=IMAGE_NOT_AVAILABLE)

    return Response(content=body, media_type="image/jpeg")


def _resolve_student_board_class(user: User) -> tuple[BoardEnum | None, ClassEnum | None]:
    board = None
    if user.teaching_board:
        try:
            board = BoardEnum(user.teaching_board)
        except ValueError:
            pass

    class_level = None
    grades = user.teaching_classes or []
    if grades:
        grade_str = grades[0].get("grade", "") if isinstance(grades[0], dict) else ""
        canon = f"CLASS_{grade_str}" if grade_str.isdigit() else grade_str
        try:
            class_level = ClassEnum(canon)
        except ValueError:
            pass

    return board, class_level


@student_router.get("/my-subjects", response_model=list[StudentSubjectResponse])
def list_my_subjects(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    board, class_level = _resolve_student_board_class(current_user)
    if not board or not class_level:
        return []

    subjects = list(
        db.scalars(
            select(SyllabusSubject)
            .where(SyllabusSubject.board == board, SyllabusSubject.class_level == class_level)
            .order_by(SyllabusSubject.subject_name)
        )
    )

    textbooks = list(
        db.scalars(
            select(TextbookUpload)
            .where(TextbookUpload.board == board, TextbookUpload.class_level == class_level)
            .order_by(TextbookUpload.chapter)
        )
    )
    tb_by_subject: dict[str, list[TextbookUpload]] = {}
    for tb in textbooks:
        tb_by_subject.setdefault(tb.subject_name, []).append(tb)

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
