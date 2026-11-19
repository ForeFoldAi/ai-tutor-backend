from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.modules.teacher.assignments.constants import AssignableArtifactType


class CreateAssignmentRequest(BaseModel):
    lesson_plan_id: int
    artifact_types: list[AssignableArtifactType] = Field(min_length=1)
    deadline: datetime
    grade: str
    section: str
    curriculum: str = ""
    subject: str | None = None


class PreviewMatchRequest(BaseModel):
    grade: str
    section: str
    curriculum: str = ""
    subject: str | None = None


class PreviewMatchResponse(BaseModel):
    matched_count: int
    grade: str
    section: str
    curriculum: str
    subject: str | None = None


class PatchAssignmentRequest(BaseModel):
    deadline: datetime


class AssignmentCounts(BaseModel):
    pending: int = 0
    in_progress: int = 0
    submitted: int = 0
    graded: int = 0
    overdue: int = 0
    total: int = 0


class TutorAssignmentListItem(BaseModel):
    id: int
    lesson_plan_id: int | None = None
    title: str
    artifact_type: str
    subject: str
    grade: str
    section: str
    curriculum: str
    chapter_name: str
    deadline: datetime
    status: str
    counts: AssignmentCounts
    created_at: datetime


class TutorAssignmentListResponse(BaseModel):
    items: list[TutorAssignmentListItem]
    total: int


class CreateAssignmentsResponse(BaseModel):
    created: list[TutorAssignmentListItem]
    message: str


class SubmissionResultItem(BaseModel):
    id: str
    question: str
    student_answer: str = ""
    correct_answer: str | None = None
    is_correct: bool | None = None


class StudentResultRow(BaseModel):
    submission_id: int
    student_id: int
    student_name: str
    status: str
    score: float | None = None
    max_score: float | None = None
    started_at: datetime | None = None
    submitted_at: datetime | None = None
    result_items: list[SubmissionResultItem] = Field(default_factory=list)
    auto_scored: bool | None = None


class AssignmentResultsResponse(BaseModel):
    assignment: TutorAssignmentListItem
    students: list[StudentResultRow]


class StudentAssignmentListItem(BaseModel):
    id: int
    submission_id: int
    title: str
    artifact_type: str
    subject: str
    teacher_name: str
    grade: str
    section: str
    curriculum: str
    chapter_name: str
    deadline: datetime
    status: str
    question_count: int
    score: float | None = None
    max_score: float | None = None
    submitted_at: datetime | None = None


class StudentAssignmentListResponse(BaseModel):
    items: list[StudentAssignmentListItem]
    total: int


class StudentAssignmentDetail(BaseModel):
    id: int
    submission_id: int
    title: str
    artifact_type: str
    subject: str
    teacher_name: str
    deadline: datetime
    status: str
    content: dict[str, Any] | None = None
    answers: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    score: float | None = None
    max_score: float | None = None
    submitted_at: datetime | None = None


class SubmitAssignmentRequest(BaseModel):
    answers: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    message: str
