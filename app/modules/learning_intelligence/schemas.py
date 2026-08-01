"""LIA API schemas — events, twin, tutor guidance (never student-facing content)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class LearningEventIn(BaseModel):
    event_type: str
    student_user_id: int
    school_id: int | None = None
    subject_name: str | None = None
    chapter_id: int | None = None
    chapter_name: str | None = None
    concept_key: str | None = None
    scope_key: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class LearningEventResponse(BaseModel):
    event_id: int
    accepted: bool = True


class TutorGuidanceObject(BaseModel):
    """Injected into AI Tutor system prompt — not shown to students directly."""

    learning_style: str = "balanced"
    preferred_examples: list[str] = Field(default_factory=list)
    preferred_explanation: str = "stepwise_with_analogy"
    teaching_pace: str = "medium"
    difficulty: str = "on_level"
    confidence_level: float = 0.5
    motivation_style: str = "effort_with_choice"
    question_frequency: str = "moderate"
    misconceptions: list[str] = Field(default_factory=list)
    knowledge_gaps: list[str] = Field(default_factory=list)
    topics_to_revise: list[str] = Field(default_factory=list)
    topics_to_connect: list[str] = Field(default_factory=list)
    topics_to_avoid: list[str] = Field(default_factory=list)
    animation_recommended: bool = False
    real_world_examples: bool = True
    encouragement_level: str = "medium"
    hint_level: str = "medium"
    cognitive_load: str = "moderate"
    next_best_action: str = "teach_current_topic"
    prompt_instructions: str = ""
    learner_guidance: str = ""
    adaptive_guidance: str = ""


class TutorGuidanceRequest(BaseModel):
    student_user_id: int
    query: str = ""
    topic: str = ""
    subject_name: str = ""
    chapter: str = ""
    chapter_ids: list[str] | None = None
    class_level: str = ""
    board: str = ""
    agent_mode: str | None = None
    understanding_scores: dict[str, Any] | None = None


class ConceptMasteryOut(BaseModel):
    concept_key: str
    mastery_score: float
    understanding_level: int
    memorized_likelihood: float


class MisconceptionOut(BaseModel):
    concept_key: str
    misconception_key: str
    description: str
    confidence: float
    occurrence_count: int


class KnowledgeGapOut(BaseModel):
    concept_key: str
    gap_type: str
    missing_prerequisites: list[str]
    confidence: float


class StudentDigitalTwinOut(BaseModel):
    student_user_id: int
    learning_style: dict[str, Any]
    teaching_preferences: dict[str, Any]
    strong_concepts: list[str]
    weak_concepts: list[str]
    strong_subjects: list[str]
    weak_subjects: list[str]
    confidence_score: float
    attention_score: float
    engagement_score: float
    motivation_score: float
    persistence_score: float
    improvement_trend: float
    regression_trend: float
    risk_level: str
    revision_behaviour: dict[str, Any]
    memory_profile: dict[str, Any]
    twin_version: int
    updated_at: datetime


class TeacherSummaryOut(BaseModel):
    student_user_id: int
    observations: list[str]
    recommendations: list[str]
    risk_level: str
    strengths: list[str]
    weaknesses: list[str]
    period_start: date | None = None
    period_end: date | None = None


class KnowledgeMapOut(BaseModel):
    student_user_id: int
    concepts: list[ConceptMasteryOut]
    misconceptions: list[MisconceptionOut]
    knowledge_gaps: list[KnowledgeGapOut]


class PredictionOut(BaseModel):
    prediction_type: str
    concept_key: str | None
    prediction: dict[str, Any]
    confidence: float


class WeakTopicInsight(BaseModel):
    topic: str
    struggle_percent: int
    student_count: int


class AffectedStudentInsight(BaseModel):
    student_user_id: int
    name: str
    grade: str
    topic: str
    risk_level: str


class InterventionInsight(BaseModel):
    id: str
    title: str
    description: str
    priority: str
    icon: str


class TutorClassInsightsOut(BaseModel):
    weak_topics: list[WeakTopicInsight]
    affected_students: list[AffectedStudentInsight]
    interventions: list[InterventionInsight]
