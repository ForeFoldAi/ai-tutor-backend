"""Alembic revision: chapter topic coverage progress fields.

Revision ID: 20260721_0040
Revises: 20260721_0039
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260721_0040"
down_revision = "20260721_0039"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "student_chapter_progress",
        sa.Column("progress_pct", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "student_chapter_progress",
        sa.Column("topics_total", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "student_chapter_progress",
        sa.Column(
            "covered_topics",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("student_chapter_progress", "covered_topics")
    op.drop_column("student_chapter_progress", "topics_total")
    op.drop_column("student_chapter_progress", "progress_pct")
