"""Alembic revision: persist AI Tutor chat per student chapter.

Revision ID: 20260722_0041
Revises: 20260721_0040
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260722_0041"
down_revision = "20260721_0040"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "student_tutor_chats",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("user_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "chapter_id",
            sa.BigInteger(),
            sa.ForeignKey("textbook_uploads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("subject_name", sa.String(120), nullable=False, server_default=""),
        sa.Column(
            "messages",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("user_id", "chapter_id", name="uq_student_tutor_chat"),
    )
    op.create_index("ix_student_tutor_chats_user_id", "student_tutor_chats", ["user_id"])
    op.create_index("ix_student_tutor_chats_chapter_id", "student_tutor_chats", ["chapter_id"])


def downgrade() -> None:
    op.drop_index("ix_student_tutor_chats_chapter_id", table_name="student_tutor_chats")
    op.drop_index("ix_student_tutor_chats_user_id", table_name="student_tutor_chats")
    op.drop_table("student_tutor_chats")
