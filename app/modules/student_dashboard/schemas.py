"""Student dashboard API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.modules.catalog.models import BoardEnum, ClassEnum


class StudentDashboardStats(BaseModel):
    enrolled_subjects: int = 0
    lessons_completed: int = 0
    total_study_seconds: int = 0
    current_streak: int = 0


class StudentDashboardSubject(BaseModel):
    id: int
    subject_name: str
    board: BoardEnum
    class_level: ClassEnum
    progress: int = 0
    completed_chapters: int = 0
    total_chapters: int = 0
    status: Literal["not_started", "in_progress", "completed"] = "not_started"


class StudentDashboardContinue(BaseModel):
    subject_id: int
    subject_name: str
    board: BoardEnum
    class_level: ClassEnum
    chapter_id: int
    chapter_name: str | None = None
    file_name: str = ""
    status: Literal["not_started", "in_progress", "completed"]
    progress: int = 0
    last_accessed_at: datetime | None = None


class StudentDashboardRecentLesson(BaseModel):
    subject_id: int
    subject_name: str
    board: BoardEnum
    class_level: ClassEnum
    chapter_id: int
    chapter_name: str | None = None
    file_name: str = ""
    status: Literal["not_started", "in_progress", "completed"]
    last_accessed_at: datetime | None = None
    progress: int = 0


class StudentDashboardResponse(BaseModel):
    """Payload shaped for the student home dashboard."""

    overall_progress: int = 0
    total_chapters: int = 0
    stats: StudentDashboardStats = Field(default_factory=StudentDashboardStats)
    subjects: list[StudentDashboardSubject] = Field(default_factory=list)
    continue_learning: StudentDashboardContinue | None = None
    recent_lessons: list[StudentDashboardRecentLesson] = Field(default_factory=list)
