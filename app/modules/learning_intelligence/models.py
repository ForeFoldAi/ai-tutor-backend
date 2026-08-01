"""LIA durable stores: events, digital twin, mastery, guidance."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class LiaLearningEvent(Base):
    __tablename__ = "lia_learning_events"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    school_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("schools.id", ondelete="SET NULL"), nullable=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False, index=True
    )
    scope_key: Mapped[str] = mapped_column(String(512), nullable=False, default="", index=True)
    subject_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    chapter_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    chapter_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    concept_key: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class LiaStudentProfile(Base):
    """Student Digital Twin — continuously evolving aggregate state."""

    __tablename__ = "lia_student_profiles"

    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    learning_style: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    teaching_preferences: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    memory_profile: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    strong_subjects: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    weak_subjects: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    strong_concepts: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    weak_concepts: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    attention_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    engagement_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    motivation_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    persistence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    improvement_trend: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    regression_trend: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False, default="low")
    revision_behaviour: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    twin_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class LiaConcept(Base):
    __tablename__ = "lia_concepts"
    __table_args__ = (UniqueConstraint("concept_key", name="uq_lia_concepts_key"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    concept_key: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    subject_name: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    chapter_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    bloom_default: Mapped[str] = mapped_column(String(32), nullable=False, default="understand")
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class LiaConceptDependency(Base):
    __tablename__ = "lia_concept_dependencies"
    __table_args__ = (
        UniqueConstraint("prerequisite_key", "dependent_key", name="uq_lia_concept_dep"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    prerequisite_key: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    dependent_key: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)


class LiaConceptMastery(Base):
    __tablename__ = "lia_concept_mastery"
    __table_args__ = (
        UniqueConstraint("student_user_id", "concept_key", name="uq_lia_concept_mastery"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    concept_key: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    mastery_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    understanding_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    memorized_likelihood: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    evidence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class LiaMisconception(Base):
    __tablename__ = "lia_misconceptions"
    __table_args__ = (
        UniqueConstraint("student_user_id", "concept_key", "misconception_key", name="uq_lia_misconception"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    concept_key: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    misconception_key: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class LiaKnowledgeGap(Base):
    __tablename__ = "lia_knowledge_gaps"
    __table_args__ = (
        UniqueConstraint("student_user_id", "concept_key", name="uq_lia_knowledge_gap"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    concept_key: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    gap_type: Mapped[str] = mapped_column(String(64), nullable=False, default="incomplete")
    missing_prerequisites: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    last_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class LiaMetricHistory(Base):
    """Time-series: confidence, attention, engagement, performance."""

    __tablename__ = "lia_metric_history"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    metric_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    context: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False, index=True
    )


class LiaTeachingMemory(Base):
    """What explanations/methods worked or failed for this student."""

    __tablename__ = "lia_teaching_memory"
    __table_args__ = (
        UniqueConstraint(
            "student_user_id", "concept_key", "method_key", name="uq_lia_teaching_memory"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    concept_key: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    method_key: Mapped[str] = mapped_column(String(128), nullable=False)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_outcome: Mapped[str] = mapped_column(String(16), nullable=False, default="neutral")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class LiaTutorGuidanceCache(Base):
    __tablename__ = "lia_tutor_guidance_cache"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scope_key: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    topic_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    guidance_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class LiaTeacherRecommendation(Base):
    __tablename__ = "lia_teacher_recommendations"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tutor_user_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    summary_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class LiaPeriodSummary(Base):
    __tablename__ = "lia_period_summaries"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    period_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    summary_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )


class LiaRiskAssessment(Base):
    __tablename__ = "lia_risk_assessments"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    factors: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    assessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False, index=True
    )


class LiaLearningPrediction(Base):
    __tablename__ = "lia_learning_predictions"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    student_user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    concept_key: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    prediction_type: Mapped[str] = mapped_column(String(64), nullable=False)
    prediction_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
