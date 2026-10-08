from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import BigInteger, Boolean, DateTime, Enum, Float, ForeignKey, Identity, Integer, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class BoardEnum(StrEnum):
    CBSE = "CBSE"
    ICSE = "ICSE"
    STATE_BOARD = "STATE_BOARD"
    IB = "IB"
    CAMBRIDGE = "CAMBRIDGE"


class ClassEnum(StrEnum):
    CLASS_1 = "CLASS_1"
    CLASS_2 = "CLASS_2"
    CLASS_3 = "CLASS_3"
    CLASS_4 = "CLASS_4"
    CLASS_5 = "CLASS_5"
    CLASS_6 = "CLASS_6"
    CLASS_7 = "CLASS_7"
    CLASS_8 = "CLASS_8"
    CLASS_9 = "CLASS_9"
    CLASS_10 = "CLASS_10"


class ProcessingStatusEnum(StrEnum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    EMBEDDED = "EMBEDDED"
    FAILED = "FAILED"


class BoardDefinition(Base):
    __tablename__ = "board_definitions"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    board: Mapped[BoardEnum] = mapped_column(Enum(BoardEnum, name="board_enum"), nullable=False, unique=True, index=True)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC), nullable=False)


class SyllabusSubject(Base):
    __tablename__ = "syllabus_subjects"
    __table_args__ = (UniqueConstraint("board", "class_level", "subject_name", name="uq_syllabus_subject"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    board: Mapped[BoardEnum] = mapped_column(Enum(BoardEnum, name="board_enum"), nullable=False, index=True)
    class_level: Mapped[ClassEnum] = mapped_column(Enum(ClassEnum, name="class_enum"), nullable=False, index=True)
    subject_name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC), nullable=False)


class ContentTypeEnum(StrEnum):
    CHAPTER = "CHAPTER"
    POEM = "POEM"
    UNIT = "UNIT"
    LESSON = "LESSON"
    MASTER = "MASTER"


class TextbookUpload(Base):
    __tablename__ = "textbook_uploads"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    board: Mapped[BoardEnum] = mapped_column(Enum(BoardEnum, name="board_enum"), nullable=False, index=True)
    class_level: Mapped[ClassEnum] = mapped_column(Enum(ClassEnum, name="class_enum"), nullable=False, index=True)
    subject_name: Mapped[str] = mapped_column(String(120), nullable=False)
    chapter: Mapped[str | None] = mapped_column(String(150), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    content_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    uploaded_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    upload_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    ocr_status: Mapped[ProcessingStatusEnum] = mapped_column(
        Enum(ProcessingStatusEnum, name="processing_status_enum"),
        nullable=False,
        default=ProcessingStatusEnum.QUEUED,
    )
    chunk_status: Mapped[ProcessingStatusEnum] = mapped_column(
        Enum(ProcessingStatusEnum, name="processing_status_enum"),
        nullable=False,
        default=ProcessingStatusEnum.QUEUED,
    )
    embedding_status: Mapped[ProcessingStatusEnum] = mapped_column(
        Enum(ProcessingStatusEnum, name="processing_status_enum"),
        nullable=False,
        default=ProcessingStatusEnum.QUEUED,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC), nullable=False)

    images: Mapped[list["TextbookImage"]] = relationship(
        "TextbookImage",
        back_populates="upload",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class TextbookImage(Base):
    """Diagrams and figures extracted from a catalog textbook file."""

    __tablename__ = "textbook_images"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    textbook_upload_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("textbook_uploads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    caption: Mapped[str | None] = mapped_column(String(512), nullable=True)
    page_text_snippet: Mapped[str | None] = mapped_column(String(900), nullable=True)
    # Pedagogy-centric fields (populated at extraction time, nullable for backward compat)
    image_type: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    caption_normalized: Mapped[str | None] = mapped_column(String(512), nullable=True)
    is_decorative: Mapped[bool] = mapped_column(nullable=False, default=False)
    educational_salience: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    # Section grounding (added for concept-specific retrieval)
    section_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    subsection_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    educational_role: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    # Figure grounding (topic-centric retrieval)
    figure_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    chapter_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    has_caption: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    nearby_text_before_figure: Mapped[str | None] = mapped_column(Text, nullable=True)
    nearby_text_after_figure: Mapped[str | None] = mapped_column(Text, nullable=True)
    figure_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_bbox: Mapped[str | None] = mapped_column(String(128), nullable=True)
    caption_bbox: Mapped[str | None] = mapped_column(String(128), nullable=True)
    caption_distance: Mapped[float | None] = mapped_column(Float, nullable=True)
    pairing_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_embedding_bge: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    context_bge_indexed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    multimodal_indexed: Mapped[bool] = mapped_column(nullable=False, default=False)
    embedding_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    # Production metadata fields (migration 0015)
    generated_caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    semantic_keywords: Mapped[str | None] = mapped_column(String(512), nullable=True)
    educational_tags: Mapped[str | None] = mapped_column(String(256), nullable=True)
    grade_level: Mapped[str | None] = mapped_column(String(12), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(120), nullable=True)
    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Production extraction fields (migration 0016)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    educational_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    concept_tags: Mapped[str | None] = mapped_column(String(512), nullable=True)
    phash: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    source_type: Mapped[str | None] = mapped_column(String(32), nullable=True, default="embedded_image")
    caption_source: Mapped[str | None] = mapped_column(String(32), nullable=True, default="none")
    page_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)
    content_kind: Mapped[str] = mapped_column(String(16), nullable=False, default="figure")
    structured_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)

    upload: Mapped["TextbookUpload"] = relationship("TextbookUpload", back_populates="images")


class TextbookStatusEnum(StrEnum):
    UPLOADED = "UPLOADED"
    ANALYZING = "ANALYZING"
    AWAITING_REVIEW = "AWAITING_REVIEW"
    CONFIRMED = "CONFIRMED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ChapterStatusEnum(StrEnum):
    DETECTED = "DETECTED"
    CONFIRMED = "CONFIRMED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Textbook(Base):
    __tablename__ = "textbooks"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    board: Mapped[BoardEnum] = mapped_column(Enum(BoardEnum, name="board_enum"), nullable=False, index=True)
    class_level: Mapped[ClassEnum] = mapped_column(Enum(ClassEnum, name="class_enum"), nullable=False, index=True)
    subject_name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    publisher: Mapped[str | None] = mapped_column(String(120), nullable=True)
    # Default book = what classes without an explicit book choice see (along with legacy chapter uploads).
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pdf_type: Mapped[str] = mapped_column(String(32), nullable=False, default="text_based")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=TextbookStatusEnum.UPLOADED)
    detection_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    detection_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    uploaded_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC), nullable=False)

    chapters: Mapped[list["TextbookChapter"]] = relationship(
        "TextbookChapter",
        back_populates="textbook",
        cascade="all, delete-orphan",
        order_by="TextbookChapter.start_pdf_page",
    )


class TextbookChapter(Base):
    __tablename__ = "textbook_chapters"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    textbook_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("textbooks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("textbook_chapters.id", ondelete="SET NULL"), nullable=True, index=True
    )
    chapter_number: Mapped[str] = mapped_column(String(32), nullable=False)
    chapter_title: Mapped[str] = mapped_column(String(255), nullable=False)
    hierarchy_level: Mapped[str] = mapped_column(String(32), nullable=False, default="chapter")
    start_pdf_page: Mapped[int] = mapped_column(Integer, nullable=False)
    end_pdf_page: Mapped[int] = mapped_column(Integer, nullable=False)
    printed_start_page: Mapped[str | None] = mapped_column(String(32), nullable=True)
    printed_end_page: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_non_chapter_section: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    section_type: Mapped[str] = mapped_column(String(32), nullable=False, default="chapter")
    detection_method: Mapped[str] = mapped_column(String(32), nullable=False, default="toc_body_match")
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    confidence_flags: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default=ChapterStatusEnum.DETECTED)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    textbook_upload_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("textbook_uploads.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC), nullable=False)

    textbook: Mapped["Textbook"] = relationship("Textbook", back_populates="chapters")
    upload: Mapped["TextbookUpload | None"] = relationship("TextbookUpload")
    parent: Mapped["TextbookChapter | None"] = relationship(
        "TextbookChapter", remote_side=[id], back_populates="sub_chapters"
    )
    sub_chapters: Mapped[list["TextbookChapter"]] = relationship(
        "TextbookChapter", back_populates="parent", cascade="all, delete-orphan"
    )


