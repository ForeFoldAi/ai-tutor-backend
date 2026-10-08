"""Master Admin Textbook endpoints for full textbook upload, chapter detection, and review."""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.auth.schemas import MessageResponse
from app.modules.catalog.models import (
    BoardEnum,
    ChapterStatusEnum,
    ClassEnum,
    Textbook,
    TextbookChapter,
    TextbookStatusEnum,
)
from app.modules.catalog.router import _local_path_for_upload, _save_upload_file
from app.modules.catalog.publisher_books import MANUAL_UPLOAD
from app.modules.catalog.schemas import (
    PublisherCreateRequest,
    TextbookChapterItem,
    TextbookPublisherUpdateRequest,
    TextbookStructureResponse,
    TextbookStructureUpdateRequest,
    TextbookSummaryResponse,
)
from app.modules.users.models import User
from app.services.textbook_segmentation import (
    detect_textbook_structure,
    process_textbook_chapter_background,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/admin/catalog/textbooks", tags=["master-admin-textbooks"])


def _analyze_textbook_background(textbook_id: int) -> None:
    """Delegate to resilient textbook structure analysis handler."""
    from app.services.catalog_pipeline.handlers.structure_handler import handle_textbook_analysis

    handle_textbook_analysis(textbook_id)


def _process_textbook_all_chapters_background(textbook_id: int) -> None:
    """Delegate to resilient chapter processing handler."""
    from app.services.catalog_pipeline.handlers.chapter_handler import handle_textbook_chapters

    handle_textbook_chapters(textbook_id)


@router.post("/upload", response_model=TextbookSummaryResponse, status_code=status.HTTP_201_CREATED)
async def upload_full_textbook(
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
    file: UploadFile = File(...),
    board: str = Form(...),
    class_name: str = Form(...),
    subject: str = Form(...),
    title: str = Form(""),
    publisher: str = Form(""),
    is_default: bool = Form(True),
    textbook_id: int | None = Form(None),
):
    """Upload a complete textbook PDF for automatic chapter detection and segmentation.

    ``textbook_id`` fills an existing publisher that has no PDF yet (created via "Add Publisher").
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext != ".pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Full textbook automatic segmentation requires a PDF document. Received '{ext}'",
        )

    shell: Textbook | None = None
    if textbook_id is not None:
        shell = db.get(Textbook, textbook_id)
        if not shell:
            raise HTTPException(status_code=404, detail="Publisher not found.")
        if shell.file_path:
            raise HTTPException(
                status_code=400,
                detail="This publisher already has a complete book. Delete it first or add a new publisher.",
            )

    file_path = await _save_upload_file(file, board, class_name, subject)
    display_title = title.strip() or os.path.splitext(file.filename or "Textbook")[0]

    if shell is not None:
        tb = shell
        tb.title = display_title
        tb.file_name = file.filename or "textbook.pdf"
        tb.file_path = file_path
        tb.uploaded_by = current_user.id
        tb.status = TextbookStatusEnum.ANALYZING
    else:
        tb = Textbook(
            title=display_title,
            file_name=file.filename or "textbook.pdf",
            file_path=file_path,
            board=BoardEnum(board),
            class_level=ClassEnum(class_name),
            subject_name=subject.strip(),
            publisher=publisher.strip() or None,
            is_default=is_default,
            uploaded_by=current_user.id,
            status=TextbookStatusEnum.ANALYZING,
        )
        db.add(tb)
    db.commit()
    db.refresh(tb)

    from app.services.catalog_pipeline import dispatch_textbook_analysis

    dispatch_textbook_analysis(tb.id, background_tasks)

    return TextbookSummaryResponse(
        id=tb.id,
        title=tb.title,
        file_name=tb.file_name,
        board=tb.board,
        class_level=tb.class_level,
        subject_name=tb.subject_name,
        publisher=tb.publisher,
        is_default=tb.is_default,
        total_pages=tb.total_pages,
        pdf_type=tb.pdf_type,
        status=tb.status,
        chapter_count=0,
        created_at=tb.created_at,
        updated_at=tb.updated_at,
    )


@router.get("", response_model=list[TextbookSummaryResponse])
def list_textbooks(
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """List all full textbooks with chapter count and processing status."""
    rows = list(db.scalars(select(Textbook).order_by(Textbook.created_at.desc())))
    result = []
    for r in rows:
        ch_count = db.scalar(
            select(func.count())
            .select_from(TextbookChapter)
            .where(
                TextbookChapter.textbook_id == r.id,
                TextbookChapter.is_non_chapter_section == False,
            )
        ) or 0
        completed_count = db.scalar(
            select(func.count())
            .select_from(TextbookChapter)
            .where(
                TextbookChapter.textbook_id == r.id,
                TextbookChapter.is_non_chapter_section == False,
                TextbookChapter.status == ChapterStatusEnum.COMPLETED,
            )
        ) or 0
        if r.status == TextbookStatusEnum.COMPLETED and completed_count < ch_count:
            completed_count = ch_count

        result.append(
            TextbookSummaryResponse(
                id=r.id,
                title=r.title,
                file_name=r.file_name,
                board=r.board,
                class_level=r.class_level,
                subject_name=r.subject_name,
                publisher=r.publisher,
                is_default=r.is_default,
                total_pages=r.total_pages,
                pdf_type=r.pdf_type,
                status=r.status,
                chapter_count=ch_count,
                completed_chapters=completed_count,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
        )
    return result


@router.patch("/{textbook_id}", response_model=MessageResponse)
def update_textbook_publisher(
    textbook_id: int,
    payload: TextbookPublisherUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Set publisher / default flag (e.g. for books uploaded before publishers existed)."""
    tb = db.get(Textbook, textbook_id)
    if not tb:
        raise HTTPException(status_code=404, detail="Textbook not found.")
    if payload.publisher is not None:
        tb.publisher = payload.publisher.strip() or None
        if not tb.file_path and tb.publisher:
            tb.title = tb.publisher
    if payload.is_default is not None:
        tb.is_default = payload.is_default
    db.commit()
    return MessageResponse(message=f"Updated '{tb.title}'.")


@router.post("/publishers", response_model=TextbookSummaryResponse, status_code=status.HTTP_201_CREATED)
def create_publisher(
    payload: PublisherCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Publisher without a PDF yet: chapters can be uploaded to it directly, or a full book added later."""
    name = payload.publisher.strip()
    tb = Textbook(
        title=name,
        file_name="",
        file_path="",
        board=payload.board,
        class_level=payload.class_name,
        subject_name=payload.subject.strip(),
        publisher=name,
        is_default=False,
        uploaded_by=current_user.id,
        status=TextbookStatusEnum.COMPLETED,
    )
    db.add(tb)
    db.commit()
    db.refresh(tb)
    return TextbookSummaryResponse.model_validate(tb)


@router.delete("/{textbook_id}", response_model=MessageResponse)
def delete_textbook(
    textbook_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Delete a book / publisher with all of its chapters (files, images, vectors).

    Classes that had picked it fall back to the default book (FK is SET NULL).
    """
    from app.modules.catalog.router import delete_textbook_upload, file_still_referenced

    tb = db.get(Textbook, textbook_id)
    if not tb:
        raise HTTPException(status_code=404, detail="Textbook not found.")
    upload_ids = {
        uid
        for uid in db.scalars(
            select(TextbookChapter.textbook_upload_id).where(
                TextbookChapter.textbook_id == textbook_id, TextbookChapter.textbook_upload_id.is_not(None)
            )
        )
    }
    for uid in upload_ids:
        delete_textbook_upload(uid, db, current_user)

    if tb.file_path and not file_still_referenced(db, tb.file_path, exclude_textbook_id=tb.id):
        try:
            from app.services.image_service.storage_backend import delete_textbook_file

            delete_textbook_file(tb.file_path)
        except Exception as exc:
            logger.warning("Could not delete source file for textbook %s: %s", textbook_id, exc)

    title = tb.title
    db.delete(tb)
    db.commit()
    return MessageResponse(message=f"Deleted '{title}' and {len(upload_ids)} chapter(s).")


@router.get("/{textbook_id}/structure", response_model=TextbookStructureResponse)
def get_textbook_structure(
    textbook_id: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Get detected chapters and non-chapter sections for admin review."""
    tb = db.get(Textbook, textbook_id)
    if not tb:
        raise HTTPException(status_code=404, detail="Textbook not found.")

    all_sections = list(
        db.scalars(
            select(TextbookChapter)
            .where(TextbookChapter.textbook_id == textbook_id)
            .order_by(TextbookChapter.start_pdf_page)
        )
    )

    # Parse detection summary if available
    det_summary = {}
    if tb.detection_summary:
        try:
            det_summary = json.loads(tb.detection_summary)
        except Exception:
            pass

    chapters_by_id: dict[int, TextbookChapterItem] = {}
    sub_items: list[tuple[int, TextbookChapterItem]] = []
    chapters = []
    non_chapters = []

    for s in all_sections:
        if s.detection_method == MANUAL_UPLOAD:
            continue
        flags = []
        if s.confidence_flags:
            try:
                flags = json.loads(s.confidence_flags)
            except Exception:
                pass

        item = TextbookChapterItem(
            id=s.id,
            parent_id=s.parent_id,
            chapter_number=s.chapter_number,
            chapter_title=s.chapter_title,
            hierarchy_level=s.hierarchy_level or "chapter",
            start_pdf_page=s.start_pdf_page,
            end_pdf_page=s.end_pdf_page,
            printed_start_page=s.printed_start_page,
            printed_end_page=s.printed_end_page,
            is_non_chapter_section=s.is_non_chapter_section,
            section_type=s.section_type,
            detection_method=s.detection_method,
            confidence_score=s.confidence_score,
            confidence_flags=flags,
            status=s.status,
            error_message=s.error_message,
            textbook_upload_id=s.textbook_upload_id,
            sub_chapters=[],
        )

        if s.is_non_chapter_section:
            non_chapters.append(item)
        elif s.parent_id is not None:
            sub_items.append((s.parent_id, item))
        else:
            chapters.append(item)
            if s.id is not None:
                chapters_by_id[s.id] = item

    # Attach sub_items to parents
    for parent_id, child_item in sub_items:
        if parent_id in chapters_by_id:
            chapters_by_id[parent_id].sub_chapters.append(child_item)
        else:
            chapters.append(child_item)

    return TextbookStructureResponse(
        textbook_id=tb.id,
        title=tb.title,
        total_pages=tb.total_pages,
        pdf_type=tb.pdf_type,
        status=tb.status,
        page_offset=tb.page_offset or det_summary.get("page_offset", 0),
        confidence_score=tb.confidence_score or det_summary.get("confidence_score", 1.0),
        detection_method=tb.detection_method or det_summary.get("detection_method", "universal_hybrid"),
        needs_manual_review=det_summary.get("needs_manual_review", False),
        review_warnings=det_summary.get("review_warnings", []),
        chapters=chapters,
        non_chapter_sections=non_chapters,
    )


@router.put("/{textbook_id}/structure", response_model=TextbookStructureResponse)
def update_textbook_structure(
    textbook_id: int,
    payload: TextbookStructureUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Allow admin to modify, add, delete, split, or merge chapter boundaries."""
    tb = db.get(Textbook, textbook_id)
    if not tb:
        raise HTTPException(status_code=404, detail="Textbook not found.")

    # Validate chapter page ranges
    for idx, ch in enumerate(payload.chapters):
        if ch.start_pdf_page < 0 or ch.end_pdf_page < ch.start_pdf_page:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid page bounds for chapter {ch.chapter_number}: {ch.start_pdf_page}-{ch.end_pdf_page}",
            )
        if tb.total_pages > 0 and ch.end_pdf_page >= tb.total_pages:
            raise HTTPException(
                status_code=400,
                detail=f"Chapter {ch.chapter_number} end page {ch.end_pdf_page} exceeds total pages ({tb.total_pages})",
            )

    # Delete existing non-fixed chapter rows
    existing_chapters = list(
        db.scalars(
            select(TextbookChapter).where(
                TextbookChapter.textbook_id == textbook_id,
                TextbookChapter.is_non_chapter_section == False,
            )
        )
    )
    for ec in existing_chapters:
        db.delete(ec)
    db.commit()

    # Re-insert updated chapters (with hierarchy)
    for ch in payload.chapters:
        flags_json = json.dumps(ch.confidence_flags) if ch.confidence_flags else "[]"
        new_row = TextbookChapter(
            textbook_id=tb.id,
            chapter_number=ch.chapter_number,
            chapter_title=ch.chapter_title.strip(),
            hierarchy_level=ch.hierarchy_level or "chapter",
            start_pdf_page=ch.start_pdf_page,
            end_pdf_page=ch.end_pdf_page,
            printed_start_page=ch.printed_start_page,
            printed_end_page=ch.printed_end_page,
            is_non_chapter_section=False,
            section_type="chapter",
            detection_method="admin_manual",
            confidence_score=1.0,
            confidence_flags=flags_json,
            status=ChapterStatusEnum.CONFIRMED,
        )
        db.add(new_row)
        db.commit()
        db.refresh(new_row)

        for sub in ch.sub_chapters:
            sub_row = TextbookChapter(
                textbook_id=tb.id,
                parent_id=new_row.id,
                chapter_number=sub.chapter_number,
                chapter_title=sub.chapter_title.strip(),
                hierarchy_level=sub.hierarchy_level or "reading",
                start_pdf_page=sub.start_pdf_page,
                end_pdf_page=sub.end_pdf_page,
                printed_start_page=sub.printed_start_page,
                printed_end_page=sub.printed_end_page,
                is_non_chapter_section=False,
                section_type="reading",
                detection_method="admin_manual",
                confidence_score=1.0,
                confidence_flags="[]",
                status=ChapterStatusEnum.CONFIRMED,
            )
            db.add(sub_row)
        db.commit()

    tb.status = TextbookStatusEnum.AWAITING_REVIEW
    db.commit()

    return get_textbook_structure(textbook_id, db, _current_user)


@router.post("/{textbook_id}/confirm", response_model=MessageResponse)
def confirm_textbook_structure(
    textbook_id: int,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Admin confirms structure; triggers background chapter-wise chunking, image extraction, and vector indexing."""
    tb = db.get(Textbook, textbook_id)
    if not tb:
        raise HTTPException(status_code=404, detail="Textbook not found.")

    chapters = list(
        db.scalars(
            select(TextbookChapter).where(
                TextbookChapter.textbook_id == textbook_id,
                TextbookChapter.is_non_chapter_section == False,
            )
        )
    )
    if not chapters:
        raise HTTPException(status_code=400, detail="Cannot confirm textbook with 0 chapters.")

    for ch in chapters:
        ch.status = ChapterStatusEnum.QUEUED
    tb.status = TextbookStatusEnum.CONFIRMED
    db.commit()

    from app.services.catalog_pipeline import dispatch_textbook_chapters

    dispatch_textbook_chapters(tb.id, background_tasks)

    return MessageResponse(
        message=f"Textbook structure confirmed for '{tb.title}'. Processing {len(chapters)} chapter(s) in background."
    )


@router.post("/{textbook_id}/retry", response_model=MessageResponse)
def retry_textbook_processing(
    textbook_id: int,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Retry analysis or failed chapters without creating duplicate records."""
    tb = db.get(Textbook, textbook_id)
    if not tb:
        raise HTTPException(status_code=404, detail="Textbook not found.")

    from app.services.catalog_pipeline import (
        dispatch_textbook_analysis,
        dispatch_textbook_chapters,
    )

    if tb.status in (TextbookStatusEnum.FAILED, TextbookStatusEnum.UPLOADED):
        # Re-run detection
        tb.status = TextbookStatusEnum.ANALYZING
        db.commit()
        dispatch_textbook_analysis(tb.id, background_tasks)
        return MessageResponse(message=f"Retrying structure analysis for '{tb.title}'.")

    # Retry failed or un-embedded chapters
    dispatch_textbook_chapters(tb.id, background_tasks)
    return MessageResponse(message=f"Retrying chapter indexing for '{tb.title}'.")


@router.get("/{textbook_id}/status")
def get_textbook_status(
    textbook_id: int,
    db: Annotated[Session, Depends(get_db)],
    _current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    """Polling status endpoint for overall progress and chapter-by-chapter details."""
    tb = db.get(Textbook, textbook_id)
    if not tb:
        raise HTTPException(status_code=404, detail="Textbook not found.")

    chapters = list(
        db.scalars(
            select(TextbookChapter)
            .where(
                TextbookChapter.textbook_id == textbook_id,
                TextbookChapter.is_non_chapter_section == False,
            )
            .order_by(TextbookChapter.start_pdf_page)
        )
    )

    completed = sum(1 for c in chapters if c.status == ChapterStatusEnum.COMPLETED)
    failed = sum(1 for c in chapters if c.status == ChapterStatusEnum.FAILED)
    processing = sum(1 for c in chapters if c.status in (ChapterStatusEnum.PROCESSING, ChapterStatusEnum.QUEUED))

    pct = int((completed / max(1, len(chapters))) * 100) if chapters else 0

    return {
        "textbook_id": tb.id,
        "title": tb.title,
        "status": tb.status,
        "total_chapters": len(chapters),
        "completed": completed,
        "failed": failed,
        "processing": processing,
        "progress_percent": pct,
        "chapters": [
            {
                "id": c.id,
                "chapter_number": c.chapter_number,
                "chapter_title": c.chapter_title,
                "start_pdf_page": c.start_pdf_page,
                "end_pdf_page": c.end_pdf_page,
                "status": c.status,
                "error_message": c.error_message,
                "textbook_upload_id": c.textbook_upload_id,
            }
            for c in chapters
        ],
    }
