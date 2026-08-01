"""Alembic revision: student learning sessions, chapter progress, streak.

Revision ID: 20260718_0034
Revises: 20260718_0033
Create Date: 2026-07-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260718_0034"
down_revision = "20260718_0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "student_study_sessions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("subject_name", sa.String(120), nullable=False),
        sa.Column(
            "chapter_id",
            sa.BigInteger(),
            sa.ForeignKey("textbook_uploads.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("chapter_name", sa.String(150), nullable=True),
        sa.Column("mode", sa.String(32), nullable=False, server_default="ai_tutor"),
        sa.Column("agent_mode", sa.String(32), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scope_key", sa.String(512), nullable=False, server_default=""),
    )
    op.create_index("ix_student_study_sessions_user_id", "student_study_sessions", ["user_id"])
    op.create_index("ix_student_study_sessions_chapter_id", "student_study_sessions", ["chapter_id"])
    op.create_index("ix_student_study_sessions_scope_key", "student_study_sessions", ["scope_key"])

    op.create_table(
        "student_chapter_progress",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "chapter_id",
            sa.BigInteger(),
            sa.ForeignKey("textbook_uploads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("subject_name", sa.String(120), nullable=False),
        sa.Column("chapter_name", sa.String(150), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="in_progress"),
        sa.Column("first_accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scope_key", sa.String(512), nullable=False, server_default=""),
        sa.UniqueConstraint("user_id", "chapter_id", name="uq_student_chapter_progress"),
    )
    op.create_index("ix_student_chapter_progress_user_id", "student_chapter_progress", ["user_id"])
    op.create_index("ix_student_chapter_progress_chapter_id", "student_chapter_progress", ["chapter_id"])
    op.create_index("ix_student_chapter_progress_scope_key", "student_chapter_progress", ["scope_key"])

    op.create_table(
        "student_learning_streak",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("current_streak", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("longest_streak", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_study_date", sa.Date(), nullable=True),
        sa.Column("scope_key", sa.String(512), nullable=False, server_default=""),
        sa.UniqueConstraint("user_id", name="uq_student_learning_streak_user"),
    )
    op.create_index("ix_student_learning_streak_user_id", "student_learning_streak", ["user_id"])


def downgrade() -> None:
    op.drop_table("student_learning_streak")
    op.drop_table("student_chapter_progress")
    op.drop_table("student_study_sessions")
