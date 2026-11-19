"""Teacher assignments + student submissions for worksheet/quiz/homework.

Revision ID: 20260718_0035
Revises: 20260718_0034
Create Date: 2026-07-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260718_0035"
down_revision = "20260718_0034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "teacher_assignments",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("teacher_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "lesson_plan_id",
            sa.BigInteger(),
            sa.ForeignKey("lesson_plans.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("artifact_type", sa.String(32), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("grade", sa.String(50), nullable=False),
        sa.Column("section", sa.String(20), nullable=False),
        sa.Column("curriculum", sa.String(50), nullable=False, server_default=""),
        sa.Column("subject", sa.String(120), nullable=False),
        sa.Column("chapter_name", sa.String(255), nullable=False, server_default=""),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=False),
        sa.Column("student_content", postgresql.JSONB(), nullable=False),
        sa.Column("answer_key", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_teacher_assignments_teacher_id", "teacher_assignments", ["teacher_id"])
    op.create_index("ix_teacher_assignments_lesson_plan_id", "teacher_assignments", ["lesson_plan_id"])
    op.create_index("ix_teacher_assignments_artifact_type", "teacher_assignments", ["artifact_type"])

    op.create_table(
        "assignment_submissions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "assignment_id",
            sa.BigInteger(),
            sa.ForeignKey("teacher_assignments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("student_id", sa.BigInteger(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("answers", postgresql.JSONB(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("max_score", sa.Float(), nullable=True),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("assignment_id", "student_id", name="uq_assignment_student"),
    )
    op.create_index("ix_assignment_submissions_assignment_id", "assignment_submissions", ["assignment_id"])
    op.create_index("ix_assignment_submissions_student_id", "assignment_submissions", ["student_id"])
    op.create_index("ix_assignment_submissions_status", "assignment_submissions", ["status"])


def downgrade() -> None:
    op.drop_table("assignment_submissions")
    op.drop_table("teacher_assignments")
