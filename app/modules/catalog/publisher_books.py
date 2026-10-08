"""Which publisher book a chapter upload belongs to.

Book key: a non-default Textbook id, or None for the "default pool"
(legacy chapter uploads + books marked is_default). A class sees uploads
whose book key equals the book chosen for that class+subject.
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.catalog.models import (
    BoardEnum,
    ChapterStatusEnum,
    ClassEnum,
    Textbook,
    TextbookChapter,
    TextbookUpload,
)

# Chapter uploaded directly (not split from the book PDF). Stored as a non-chapter section so the
# PDF pipeline (confirm / process / structure edit) never touches it; only the book link matters.
MANUAL_UPLOAD = "manual_upload"


def attach_upload_to_book(db: Session, book: Textbook, upload: TextbookUpload) -> None:
    db.add(
        TextbookChapter(
            textbook_id=book.id,
            chapter_number=str(upload.id),
            chapter_title=upload.content_label or upload.chapter or upload.file_name,
            hierarchy_level="chapter",
            start_pdf_page=0,
            end_pdf_page=0,
            is_non_chapter_section=True,
            section_type="chapter",
            detection_method=MANUAL_UPLOAD,
            status=ChapterStatusEnum.COMPLETED,
            textbook_upload_id=upload.id,
        )
    )


def publisher_book_of(db: Session, upload_ids: Iterable[int]) -> dict[int, int]:
    """upload id -> non-default textbook id. Uploads missing from the result are in the default pool."""
    ids = {int(i) for i in upload_ids}
    if not ids:
        return {}
    rows = db.execute(
        select(TextbookChapter.textbook_upload_id, Textbook.id)
        .join(Textbook, Textbook.id == TextbookChapter.textbook_id)
        .where(TextbookChapter.textbook_upload_id.in_(ids), Textbook.is_default.is_(False))
    )
    return {int(uid): int(tid) for uid, tid in rows}


def filter_uploads_for_book(
    uploads: list[TextbookUpload],
    book_of: dict[int, int],
    chosen: int | None,
) -> list[TextbookUpload]:
    return [u for u in uploads if book_of.get(u.id) == chosen]


def books_for(db: Session, board: BoardEnum, class_level: ClassEnum, subject_name: str) -> list[Textbook]:
    norm = " ".join(subject_name.strip().lower().split())
    rows = db.scalars(
        select(Textbook)
        .where(Textbook.board == board, Textbook.class_level == class_level)
        .order_by(Textbook.is_default.desc(), Textbook.publisher, Textbook.title)
    )
    return [t for t in rows if " ".join(t.subject_name.strip().lower().split()) == norm]
