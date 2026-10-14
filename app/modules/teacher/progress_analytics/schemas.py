"""Progress analytics API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassHealthOut(BaseModel):
    score: int = 0
    status: str = "Needs Attention"  # Good | Fair | Needs Attention
    trend_percent: int = 0
    trend_up: bool = True


class TopicMasteryOut(BaseModel):
    topic: str
    mastery: int


class CompletionTrendPointOut(BaseModel):
    date: str
    completion: int


class AtRiskStudentOut(BaseModel):
    id: int
    slug: str
    name: str
    grade: str
    subject: str
    risk_level: str  # High | Medium | Low


class ProgressAnalyticsResponse(BaseModel):
    class_health: ClassHealthOut
    topic_mastery: list[TopicMasteryOut] = Field(default_factory=list)
    completion_trend: list[CompletionTrendPointOut] = Field(default_factory=list)
    at_risk_students: list[AtRiskStudentOut] = Field(default_factory=list)
