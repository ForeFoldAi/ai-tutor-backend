"""Live tutor sessions + joins.

Revision ID: 20260718_0036
Revises: 20260718_0035
Create Date: 2026-07-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260718_0036"
down_revision = "20260718_0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "live_tutor_sessions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("tutor_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("school_id", sa.BigInteger(), sa.ForeignKey("schools.id", ondelete="CASCADE"), nullable=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("subject", sa.String(120), nullable=False),
        sa.Column("chapter_id", sa.String(64), nullable=True),
        sa.Column("chapter", sa.String(150), nullable=True),
        sa.Column("grade", sa.String(20), nullable=False),
        sa.Column("section", sa.String(20), nullable=False),
        sa.Column("curriculum", sa.String(100), nullable=True),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_minutes", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("meeting_link", sa.String(512), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_live_tutor_sessions_tutor_id", "live_tutor_sessions", ["tutor_id"])
    op.create_index("ix_live_tutor_sessions_school_id", "live_tutor_sessions", ["school_id"])
    op.create_index("ix_live_tutor_sessions_starts_at", "live_tutor_sessions", ["starts_at"])

    op.create_table(
        "live_session_joins",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "session_id",
            sa.BigInteger(),
            sa.ForeignKey("live_tutor_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("student_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("joined_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_id", "student_id", name="uq_live_session_join"),
    )
    op.create_index("ix_live_session_joins_session_id", "live_session_joins", ["session_id"])
    op.create_index("ix_live_session_joins_student_id", "live_session_joins", ["student_id"])


def downgrade() -> None:
    op.drop_table("live_session_joins")
    op.drop_table("live_tutor_sessions")
