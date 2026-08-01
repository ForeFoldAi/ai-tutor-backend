from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.modules.catalog.models import BoardEnum, ClassEnum


class LearningChapterOut(BaseModel):
    id: int
    chapter: str | None
    file_name: str
    status: Literal["not_started", "in_progress", "completed"] = "not_started"
    progress: int = 0
    topics_total: int = 0
    topics_covered: int = 0
    topics_remaining: list[str] = []


class LearningSubjectOut(BaseModel):
    id: int
    board: BoardEnum
    class_level: ClassEnum
    subject_name: str
    chapters: list[LearningChapterOut] = []
    completed_chapters: int = 0
    total_chapters: int = 0
    progress: int = 0
    status: Literal["not_started", "in_progress", "completed"] = "not_started"


class LearningStatsOut(BaseModel):
    enrolled_subjects: int = 0
    lessons_completed: int = 0
    total_study_seconds: int = 0
    current_streak: int = 0


class ContinueLearningOut(BaseModel):
    subject_id: int
    subject_name: str
    board: BoardEnum
    class_level: ClassEnum
    chapter_id: int
    chapter_name: str | None
    file_name: str
    status: Literal["not_started", "in_progress", "completed"]
    progress: int = 0
    last_accessed_at: datetime | None = None


class RecentLessonOut(BaseModel):
    subject_id: int
    subject_name: str
    board: BoardEnum
    class_level: ClassEnum
    chapter_id: int
    chapter_name: str | None
    file_name: str
    status: Literal["not_started", "in_progress", "completed"]
    last_accessed_at: datetime | None = None


class RecommendedTopicOut(BaseModel):
    """AI Tutor home — suggested topics / chapters for the student."""

    title: str
    reason: str = "Recommended for you"
    subject_id: int
    subject_name: str
    board: BoardEnum
    class_level: ClassEnum
    chapter_id: int
    chapter_name: str | None = None


class LearningOverviewResponse(BaseModel):
    stats: LearningStatsOut
    subjects: list[LearningSubjectOut]
    continue_learning: ContinueLearningOut | None = None
    recent_lessons: list[RecentLessonOut] = []
    recommended_topics: list[RecommendedTopicOut] = []
    scope_key: str = ""


class SessionStartRequest(BaseModel):
    subject_name: str = Field(min_length=1, max_length=120)
    chapter_id: int | None = None
    chapter_name: str | None = Field(default=None, max_length=150)
    mode: Literal["ai_tutor", "ai_voice"] = "ai_tutor"
    agent_mode: Literal["ask", "practice", "explain", "free"] | None = None


class SessionStartResponse(BaseModel):
    session_id: int
    started_at: datetime


class SessionHeartbeatResponse(BaseModel):
    session_id: int
    duration_seconds: int


class SessionEndResponse(BaseModel):
    session_id: int
    duration_seconds: int
    ended_at: datetime


class ChapterCompleteResponse(BaseModel):
    chapter_id: int
    status: Literal["completed"]
    completed_at: datetime


class TutorChatMessageIn(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)
    created_at: str | None = None


class TutorChatPutRequest(BaseModel):
    subject_name: str = Field(default="", max_length=120)
    messages: list[TutorChatMessageIn] = Field(default_factory=list)
    thread_id: str | None = Field(default=None, max_length=80)
    active_thread_id: str | None = Field(default=None, max_length=80)
    new_thread: bool = False


class TutorChatMessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str | None = None


class TutorChatThreadOut(BaseModel):
    id: str
    title: str = "Chat"
    messages: list[TutorChatMessageOut] = Field(default_factory=list)
    updated_at: str | None = None


class TutorChatResponse(BaseModel):
    chapter_id: int
    subject_name: str = ""
    messages: list[TutorChatMessageOut] = Field(default_factory=list)
    threads: list[TutorChatThreadOut] = Field(default_factory=list)
    active_thread_id: str | None = None
    updated_at: datetime | None = None
