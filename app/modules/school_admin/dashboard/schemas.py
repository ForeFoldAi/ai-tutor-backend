from __future__ import annotations

from pydantic import BaseModel, Field


class DashboardMetrics(BaseModel):
    total_students: int
    total_teachers: int
    teacher_guided_students: int
    self_learning_students: int
    active_this_week: int
    students_trend: str
    teachers_trend: str
    active_trend: str
    teacher_guided_percent: str
    self_learning_percent: str


class OnboardingProgressItem(BaseModel):
    id: str
    label: str
    completed: int
    total: int
    percent: int


class PendingItem(BaseModel):
    id: str
    label: str
    count: int
    severity: str = Field(description="warning | danger")
    icon: str = Field(description="assignments | credentials | mapping")


class DashboardSummaryResponse(BaseModel):
    metrics: DashboardMetrics
    onboarding: list[OnboardingProgressItem]
    pending: list[PendingItem]
    curricula: list[str]
