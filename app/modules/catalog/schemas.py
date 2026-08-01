from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.modules.catalog.models import BoardEnum, ClassEnum, ProcessingStatusEnum


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

    model_config = {"from_attributes": True}


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


class StudentSubjectResponse(BaseModel):
    id: int
    board: BoardEnum
    class_level: ClassEnum
    subject_name: str
    chapters: list[StudentChapterResponse] = []
