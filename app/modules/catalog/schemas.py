from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.modules.catalog.models import BoardEnum, ClassEnum, ProcessingStatusEnum
from app.core.text_clean import clean_display_label


class BoardCreateRequest(BaseModel):
    board: BoardEnum
    country: str = Field(min_length=2, max_length=100)


class BoardResponse(BaseModel):
    id: int
    board: BoardEnum
    country: str
    created_at: datetime

    model_config = {"from_attributes": True}


class SyllabusBulkCreateRequest(BaseModel):
    board: BoardEnum
    class_names: list[ClassEnum] = Field(min_length=1)
    subject_names: list[str] = Field(min_length=1)

    @field_validator("subject_names", mode="before")
    @classmethod
    def _clean_subjects(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("subject_names must be a list")
        out = [str(v).strip() for v in value if str(v).strip()]
        if not out:
            raise ValueError("At least one subject name is required.")
        return out


class SyllabusUpdateRequest(BaseModel):
    board: BoardEnum | None = None
    class_level: ClassEnum | None = None
    subject_name: str | None = Field(default=None, min_length=1, max_length=120)

    @field_validator("subject_name", mode="before")
    @classmethod
    def _strip(cls, v: object) -> str | None:
        return None if v is None else str(v).strip()


class SyllabusSubjectResponse(BaseModel):
    id: int
    board: BoardEnum
    class_level: ClassEnum
    subject_name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class TextbookUploadCreateRequest(BaseModel):
    file_name: str = Field(min_length=3, max_length=255)
    board: BoardEnum
    class_name: ClassEnum
    subject: str = Field(min_length=1, max_length=120)
    chapter: str | None = Field(default=None, max_length=150)
    content_type: str | None = Field(default=None, max_length=20)
    content_label: str | None = Field(default=None, max_length=255)

    @field_validator("chapter", mode="before")
    @classmethod
    def _blank_optional(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class TextbookUploadPatchStatusRequest(BaseModel):
    ocr_status: ProcessingStatusEnum
    chunk_status: ProcessingStatusEnum
    embedding_status: ProcessingStatusEnum


class TextbookUploadResponse(BaseModel):
    id: int
    file_name: str
    board: BoardEnum
    class_level: ClassEnum
    subject_name: str
    chapter: str | None
    content_type: str | None = None
    content_label: str | None = None
    file_path: str | None = None
    chunk_count: int = 0
    uploaded_by: int | None
    upload_date: datetime
    ocr_status: ProcessingStatusEnum
    chunk_status: ProcessingStatusEnum
    embedding_status: ProcessingStatusEnum
    # Book this chapter was split from (None = standalone chapter upload → default pool)
    textbook_id: int | None = None

    model_config = {"from_attributes": True}

    @field_validator("chapter", "content_label", mode="before")
    @classmethod
    def _clean_labels(cls, v: object) -> str | None:
        return clean_display_label(v if isinstance(v, str) or v is None else str(v))


class TextbookUploadUpdateRequest(BaseModel):
    chapter: str | None = None
    content_label: str | None = None
    content_type: str | None = None


class EmbeddingStatsResponse(BaseModel):
    total_documents: int
    embedded_count: int
    failed_count: int
    pending_count: int
    total_chunks: int
    embedding_model: str


class ProcessResponse(BaseModel):
    id: int
    chunk_count: int
    chunk_status: ProcessingStatusEnum
    embedding_status: ProcessingStatusEnum
    message: str


class CatalogEnumsResponse(BaseModel):
    boards: list[BoardEnum]
    classes: list[ClassEnum]


class StudentChapterResponse(BaseModel):
    id: int
    chapter: str | None
    file_name: str

    model_config = {"from_attributes": True}

    @field_validator("chapter", mode="before")
    @classmethod
    def _clean_chapter(cls, v: object) -> str | None:
        return clean_display_label(v if isinstance(v, str) or v is None else str(v))


class StudentSubjectResponse(BaseModel):
    id: int
    board: BoardEnum
    class_level: ClassEnum
    subject_name: str
    chapters: list[StudentChapterResponse] = []


class TextbookChapterItem(BaseModel):
    id: int | None = None
    parent_id: int | None = None
    chapter_number: str
    chapter_title: str
    hierarchy_level: str = "chapter"  # "unit", "chapter", "reading", "lesson", "appendix"
    start_pdf_page: int
    end_pdf_page: int
    printed_start_page: str | None = None
    printed_end_page: str | None = None
    is_non_chapter_section: bool = False
    section_type: str = "chapter"
    detection_method: str = "toc_body_match"
    confidence_score: float = 1.0
    confidence_flags: list[str] = []
    status: str = "DETECTED"
    error_message: str | None = None
    textbook_upload_id: int | None = None
    sub_chapters: list["TextbookChapterItem"] = []

    model_config = {"from_attributes": True}


class TextbookStructureUpdateRequest(BaseModel):
    chapters: list[TextbookChapterItem]
    ingestion_granularity: str = "chapter"  # "unit", "chapter", "reading"


class TextbookStructureResponse(BaseModel):
    textbook_id: int
    title: str
    total_pages: int
    pdf_type: str
    status: str
    page_offset: int | None = 0
    confidence_score: float = 1.0
    detection_method: str = "hybrid"
    needs_manual_review: bool = False
    review_warnings: list[str] = []
    chapters: list[TextbookChapterItem]
    non_chapter_sections: list[TextbookChapterItem] = []


class TextbookSummaryResponse(BaseModel):
    id: int
    title: str
    file_name: str
    board: BoardEnum
    class_level: ClassEnum
    subject_name: str
    publisher: str | None = None
    is_default: bool = True
    total_pages: int
    pdf_type: str
    status: str
    chapter_count: int = 0
    completed_chapters: int = 0
    page_offset: int | None = None
    confidence_score: float | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PublisherCreateRequest(BaseModel):
    board: BoardEnum
    class_name: ClassEnum
    subject: str = Field(min_length=1, max_length=120)
    publisher: str = Field(min_length=1, max_length=120)


class TextbookPublisherUpdateRequest(BaseModel):
    publisher: str | None = Field(default=None, max_length=120)
    is_default: bool | None = None


class TextbookUploadUpdateRequest(BaseModel):
    chapter: str | None = None
    content_label: str | None = None
    content_type: str | None = None


class WorkerNodeInfo(BaseModel):
    name: str
    status: str
    concurrency: int = 1
    active_tasks_count: int = 0
    processed_total: int = 0
    active_queues: list[str] = []


class QueueDepthInfo(BaseModel):
    catalog_ingest: int = 0
    lesson_generate: int = 0
    mail: int = 0


class ActiveJobInfo(BaseModel):
    task_id: str
    task_name: str
    args: list[str] = []
    worker: str
    time_start: float | None = None
    target_label: str = ""


class RecentProcessInfo(BaseModel):
    id: int
    file_name: str
    board: str
    class_level: str
    subject_name: str
    chunk_count: int = 0
    embedding_status: str
    updated_at: datetime | None = None


class WorkerStatusResponse(BaseModel):
    status: str
    redis_connected: bool
    workers: list[WorkerNodeInfo] = []
    queue_depths: QueueDepthInfo
    active_jobs: list[ActiveJobInfo] = []
    total_documents: int = 0
    embedded_count: int = 0
    failed_count: int = 0
    processing_count: int = 0
    queued_count: int = 0
    total_chunks: int = 0
    recent_processes: list[RecentProcessInfo] = []
    vector_backend: str = "qdrant"
    storage_backend: str = "s3"


class TextbookImageAdminResponse(BaseModel):
    id: int
    textbook_upload_id: int
    page_index: int
    sequence: int
    file_name: str
    image_url: str
    caption: str | None = None
    caption_normalized: str | None = None
    image_type: str = "unknown"
    figure_number: str | None = None
    title: str | None = None
    educational_role: str = "unknown"
    educational_description: str | None = None
    is_decorative: bool = False
    image_bbox: str | None = None
    structured_content: str | None = None
    content_kind: str = "figure"
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class TextbookImageUpdateRequest(BaseModel):
    caption: str | None = None
    title: str | None = None
    figure_number: str | None = None
    image_type: str | None = None
    educational_role: str | None = None
    is_decorative: bool | None = None
    educational_description: str | None = None

