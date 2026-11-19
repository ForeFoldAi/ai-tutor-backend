from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.modules.teacher.lesson_planner.constants import (
    ALL_ARTIFACT_TYPES,
    ArtifactType,
    ExportFormat,
    JobStatus,
    LessonPlanStatus,
)


class ChapterTopicItem(BaseModel):
    key: str
    title: str


class ChapterTopicsResponse(BaseModel):
    chapter_id: str
    topics: list[ChapterTopicItem] = Field(default_factory=list)


class PptThemeItem(BaseModel):
    id: str
    label: str
    description: str
    primary: str = ""
    accent: str = ""
    bg: str = ""


class PptThemesResponse(BaseModel):
    themes: list[PptThemeItem] = Field(default_factory=list)
    default_id: str = "clean_academic"


class GenerateLessonPlanRequest(BaseModel):
    grade: str = Field(..., min_length=1, max_length=50)
    subject: str = Field(..., min_length=1, max_length=120)
    chapter_id: str | None = Field(default=None, max_length=64)
    chapter_name: str = Field(..., min_length=1, max_length=255)
    duration_minutes: int = Field(default=45, ge=15, le=180)
    learning_objectives: str = Field(default="", max_length=5000)
    # Teacher-selected topics for this lesson (1+); empty = whole chapter.
    topics: list[str] = Field(default_factory=list, max_length=8)
    # Same-grade sections only (e.g. A, B of Class 9).
    sections: list[str] = Field(default_factory=list, max_length=12)
    # PPT visual theme for classroom decks.
    ppt_template: str = Field(default="clean_academic", max_length=64)
    # Target slide count for PPT Outline (8 / 12 / 16).
    ppt_slide_count: int = Field(default=12, ge=6, le=20)
    board: str | None = Field(default=None, max_length=50)
    title: str | None = Field(default=None, max_length=255)
    requested_artifacts: list[ArtifactType] = Field(
        default_factory=lambda: list(ALL_ARTIFACT_TYPES)
    )
    lesson_plan_id: int | None = None
    idempotency_key: str | None = Field(default=None, max_length=128)

    @field_validator("topics")
    @classmethod
    def clean_topics(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in v:
            title = (raw or "").strip()[:120]
            if not title:
                continue
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(title)
            if len(out) >= 8:
                break
        return out

    @field_validator("sections")
    @classmethod
    def clean_sections(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in v:
            section = (raw or "").strip()[:32]
            if not section:
                continue
            key = section.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(section)
            if len(out) >= 12:
                break
        return out

    @field_validator("ppt_template")
    @classmethod
    def clean_ppt_template(cls, v: str) -> str:
        from app.services.lesson_planner.export.pptx_themes import DEFAULT_THEME_ID, THEMES

        key = (v or "").strip().lower().replace("-", "_")
        return key if key in THEMES else DEFAULT_THEME_ID

    @field_validator("ppt_slide_count")
    @classmethod
    def clean_ppt_slide_count(cls, v: int) -> int:
        n = int(v or 12)
        if n <= 10:
            return 8
        if n <= 14:
            return 12
        return 16

    @field_validator("requested_artifacts")
    @classmethod
    def dedupe_artifacts(cls, v: list[ArtifactType]) -> list[ArtifactType]:
        seen: set[ArtifactType] = set()
        out: list[ArtifactType] = []
        for item in v:
            if item not in seen:
                seen.add(item)
                out.append(item)
        if not out:
            raise ValueError("At least one artifact type is required")
        return out


class RegenerateRequest(BaseModel):
    lesson_plan_id: int
    artifact_types: list[ArtifactType] = Field(default_factory=list)
    idempotency_key: str | None = Field(default=None, max_length=128)

    @field_validator("artifact_types")
    @classmethod
    def require_artifacts(cls, v: list[ArtifactType]) -> list[ArtifactType]:
        if not v:
            raise ValueError("At least one artifact type is required for regeneration")
        return v


class SaveLessonPlanRequest(BaseModel):
    lesson_plan_id: int
    title: str | None = Field(default=None, max_length=255)
    artifacts: dict[str, dict[str, Any]] = Field(default_factory=dict)
    change_summary: str | None = Field(default=None, max_length=500)


class ExportLessonPlanRequest(BaseModel):
    lesson_plan_id: int
    export_format: ExportFormat
    version_id: int | None = None
    artifact_types: list[ArtifactType] | None = None


class CancelJobRequest(BaseModel):
    job_id: int


class ResumeJobRequest(BaseModel):
    job_id: int


class LessonPlanPatchRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    grade: str | None = Field(default=None, max_length=50)
    subject: str | None = Field(default=None, max_length=120)
    chapter_name: str | None = Field(default=None, max_length=255)
    duration_minutes: int | None = Field(default=None, ge=15, le=180)
    learning_objectives: str | None = Field(default=None, max_length=5000)


class ArtifactResponse(BaseModel):
    id: int
    artifact_type: ArtifactType
    content: dict[str, Any] | None
    status: str
    version_number: int
    updated_at: datetime

    model_config = {"from_attributes": True}


class LessonPlanResponse(BaseModel):
    id: int
    title: str
    grade: str
    subject: str
    board: str | None
    chapter_id: str | None
    chapter_name: str
    duration_minutes: int
    learning_objectives: str
    status: LessonPlanStatus
    plan_metadata: dict[str, Any] | None = None
    artifacts: list[ArtifactResponse] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class LessonPlanSummaryResponse(BaseModel):
    id: int
    title: str
    grade: str
    subject: str
    chapter_name: str
    status: LessonPlanStatus
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobResponse(BaseModel):
    id: int
    lesson_plan_id: int | None
    status: JobStatus
    progress: int
    message: str | None
    requested_artifacts: list[str]
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class GenerateResponse(BaseModel):
    job_id: int
    lesson_plan_id: int
    status: JobStatus
    websocket_url: str


class ExportResponse(BaseModel):
    export_id: int
    status: str
    file_path: str | None = None
    download_url: str | None = None


class MessageResponse(BaseModel):
    message: str
