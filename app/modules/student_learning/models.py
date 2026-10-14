"""Durable student learning analytics: sessions, chapter progress, streak."""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class StudentStudySession(Base):
    __tablename__ = "student_study_sessions"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_name: Mapped[str] = mapped_column(String(120), nullable=False)
    chapter_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("textbook_uploads.id", ondelete="SET NULL"), nullable=True, index=True
    )
    chapter_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="ai_tutor")
    agent_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    scope_key: Mapped[str] = mapped_column(String(512), nullable=False, default="", index=True)


class StudentChapterProgress(Base):
    __tablename__ = "student_chapter_progress"
    __table_args__ = (UniqueConstraint("user_id", "chapter_id", name="uq_student_chapter_progress"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chapter_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("textbook_uploads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_name: Mapped[str] = mapped_column(String(120), nullable=False)
    chapter_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="in_progress")
    first_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scope_key: Mapped[str] = mapped_column(String(512), nullable=False, default="", index=True)
    # Topic-coverage progress: covered headings / chapter headings
    progress_pct: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    topics_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    covered_topics: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)


class StudentLearningStreak(Base):
    __tablename__ = "student_learning_streak"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    current_streak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    longest_streak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_study_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    scope_key: Mapped[str] = mapped_column(String(512), nullable=False, default="")


class StudentTutorChat(Base):
    """One persisted AI Tutor thread per student chapter — resume Continue/Recent Lessons."""

    __tablename__ = "student_tutor_chats"
    __table_args__ = (UniqueConstraint("user_id", "chapter_id", name="uq_student_tutor_chat"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chapter_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("textbook_uploads.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subject_name: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    # [{id, role, content, created_at}] — cap in service
    messages: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
