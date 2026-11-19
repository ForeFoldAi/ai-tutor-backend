from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, Identity, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.modules.teacher.lesson_planner.constants import (
    ArtifactStatus,
    ArtifactType,
    ExportFormat,
    ExportStatus,
    JobStatus,
    LessonPlanStatus,
)


def _pg_enum(enum_cls: type, name: str) -> Enum:
    # ponytail: StrEnum names (DRAFT) != DB values (draft) — persist .value
    return Enum(enum_cls, name=name, values_callable=lambda members: [m.value for m in members])


class LessonPlan(Base):
    __tablename__ = "lesson_plans"
    __table_args__ = (
        Index("ix_lesson_plans_user_id_deleted_at", "user_id", "deleted_at"),
        Index("ix_lesson_plans_status", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="Untitled Lesson Plan")
    grade: Mapped[str] = mapped_column(String(50), nullable=False)
    subject: Mapped[str] = mapped_column(String(120), nullable=False)
    board: Mapped[str | None] = mapped_column(String(50), nullable=True)
    chapter_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    chapter_name: Mapped[str] = mapped_column(String(255), nullable=False)
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=45)
    learning_objectives: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[LessonPlanStatus] = mapped_column(
        _pg_enum(LessonPlanStatus, "lesson_plan_status_enum"),
        nullable=False,
        default=LessonPlanStatus.DRAFT,
    )
    plan_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    artifacts: Mapped[list["LessonArtifact"]] = relationship(
        "LessonArtifact",
        back_populates="lesson_plan",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    jobs: Mapped[list["LessonJob"]] = relationship(
        "LessonJob",
        back_populates="lesson_plan",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    versions: Mapped[list["LessonVersion"]] = relationship(
        "LessonVersion",
        back_populates="lesson_plan",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    exports: Mapped[list["LessonExport"]] = relationship(
        "LessonExport",
        back_populates="lesson_plan",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class LessonArtifact(Base):
    __tablename__ = "lesson_plan_artifacts"
    __table_args__ = (
        UniqueConstraint("lesson_plan_id", "artifact_type", "version_number", name="uq_lesson_artifact_version"),
        Index("ix_lesson_plan_artifacts_type_status", "artifact_type", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    lesson_plan_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("lesson_plans.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("lesson_plan_jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    artifact_type: Mapped[ArtifactType] = mapped_column(
        _pg_enum(ArtifactType, "lesson_artifact_type_enum"), nullable=False
    )
    content: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[ArtifactStatus] = mapped_column(
        _pg_enum(ArtifactStatus, "lesson_artifact_status_enum"),
        nullable=False,
        default=ArtifactStatus.PENDING,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    lesson_plan: Mapped["LessonPlan"] = relationship("LessonPlan", back_populates="artifacts")
    job: Mapped["LessonJob | None"] = relationship("LessonJob", back_populates="artifacts")


class LessonJob(Base):
    __tablename__ = "lesson_plan_jobs"
    __table_args__ = (
        Index("ix_lesson_plan_jobs_status", "status"),
        Index("ix_lesson_plan_jobs_celery_task_id", "celery_task_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    lesson_plan_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("lesson_plans.id", ondelete="CASCADE"), nullable=True, index=True
    )
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[JobStatus] = mapped_column(
        _pg_enum(JobStatus, "lesson_job_status_enum"), nullable=False, default=JobStatus.QUEUED
    )
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    requested_artifacts: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    input_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    lesson_plan: Mapped["LessonPlan | None"] = relationship("LessonPlan", back_populates="jobs")
    artifacts: Mapped[list["LessonArtifact"]] = relationship("LessonArtifact", back_populates="job")


class LessonVersion(Base):
    __tablename__ = "lesson_plan_versions"
    __table_args__ = (Index("ix_lesson_plan_versions_plan_version", "lesson_plan_id", "version_number"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    lesson_plan_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("lesson_plans.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    change_summary: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_by: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    lesson_plan: Mapped["LessonPlan"] = relationship("LessonPlan", back_populates="versions")


class LessonExport(Base):
    __tablename__ = "lesson_plan_exports"
    __table_args__ = (Index("ix_lesson_plan_exports_status", "status"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    lesson_plan_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("lesson_plans.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("lesson_plan_versions.id", ondelete="SET NULL"), nullable=True
    )
    export_format: Mapped[ExportFormat] = mapped_column(
        _pg_enum(ExportFormat, "lesson_export_format_enum"), nullable=False
    )
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[ExportStatus] = mapped_column(
        _pg_enum(ExportStatus, "lesson_export_status_enum"), nullable=False, default=ExportStatus.PENDING
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    lesson_plan: Mapped["LessonPlan"] = relationship("LessonPlan", back_populates="exports")
