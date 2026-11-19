from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PaginationMeta(BaseModel):
    total: int
    limit: int
    offset: int
    default_limit: int


class SkillMetric(BaseModel):
    name: str
    score: int


class ActivityItem(BaseModel):
    title: str
    when: str


class QuizItem(BaseModel):
    name: str
    score: int


class AssignmentItem(BaseModel):
    title: str
    due: str
    status: str  # Completed | In Progress | Not Started


class TutorStudentRow(BaseModel):
    id: int
    slug: str
    user_id: str
    full_name: str
    grade: str | None
    section: str | None
    curriculum: str | None
    grade_label: str
    subjects: list[str] = Field(default_factory=list)
    completion: int
    risk_level: str
    last_active: str | None
    last_active_at: datetime | None = None
    is_active: bool


class TutorStudentListResponse(BaseModel):
    items: list[TutorStudentRow]
    meta: PaginationMeta


class TutorStudentOptionsResponse(BaseModel):
    grades: list[str]
    subjects: list[str]
    risk_levels: list[str]
    default_limit: int
    page_size_options: list[int]


class TutorStudentProfileResponse(TutorStudentRow):
    current_topic: str | None = None
    overall_progress: int = 0
    strengths: list[SkillMetric] = Field(default_factory=list)
    needs_improvement: list[SkillMetric] = Field(default_factory=list)
    ai_activity: list[ActivityItem] = Field(default_factory=list)
    average_quiz_score: int = 0
    quiz_trend: int = 0
    recent_quizzes: list[QuizItem] = Field(default_factory=list)
    assignments: list[AssignmentItem] = Field(default_factory=list)
    teacher_notes: str = ""
    notes_updated: str | None = None
