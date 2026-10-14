"""Tutor dashboard API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TutorDashboardMetrics(BaseModel):
    total_assigned: int = 0
    active_this_week: int = 0
    average_completion: int = 0
    attention_required: int = 0
    assigned_trend: str = "—"
    active_trend: str = "—"
    completion_trend: str = "—"
    attention_trend: str = "—"


class TutorDashboardSession(BaseModel):
    id: int
    title: str
    grade: str
    section: str
    subject: str = ""
    starts_at: datetime
    duration_minutes: int = 60
    status: Literal["live", "upcoming", "completed"] = "upcoming"


class TutorAttentionStudent(BaseModel):
    student_user_id: int
    name: str
    grade: str = "—"
    topic: str = "General"
    risk_level: Literal["High", "Medium", "Low"] = "Medium"
    tag: str = "Needs attention"


class TutorAiRecommendation(BaseModel):
    id: str
    title: str
    description: str
    priority: Literal["High", "Medium", "Low"] = "Medium"
    icon: Literal["quiz", "worksheet", "revision", "concept"] = "concept"
    action_label: str = "Open AI Insights"
    action_href: str = "/tutor/ai-insights"


class TutorDashboardSummaryResponse(BaseModel):
    metrics: TutorDashboardMetrics = Field(default_factory=TutorDashboardMetrics)
    todays_sessions: list[TutorDashboardSession] = Field(default_factory=list)
    attention_students: list[TutorAttentionStudent] = Field(default_factory=list)
    ai_recommendations: list[TutorAiRecommendation] = Field(default_factory=list)
