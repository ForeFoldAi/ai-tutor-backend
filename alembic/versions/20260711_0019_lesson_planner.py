"""lesson planner tables

Revision ID: 20260711_0019
Revises: 20260629_0018
Create Date: 2026-07-11

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260711_0019"
down_revision = "20260629_0018"
branch_labels = None
depends_on = None

lesson_plan_status = postgresql.ENUM(
    "draft", "generating", "completed", "failed", "cancelled",
    name="lesson_plan_status_enum",
    create_type=False,
)
artifact_type = postgresql.ENUM(
    "lesson_plan", "teaching_notes", "examples", "worksheet", "quiz", "homework", "ppt_outline",
    name="lesson_artifact_type_enum",
    create_type=False,
)
artifact_status = postgresql.ENUM(
    "pending", "generating", "completed", "failed",
    name="lesson_artifact_status_enum",
    create_type=False,
)
job_status = postgresql.ENUM(
    "queued", "running", "completed", "failed", "cancelled",
    name="lesson_job_status_enum",
    create_type=False,
)
export_format = postgresql.ENUM("pdf", "docx", "pptx", name="lesson_export_format_enum", create_type=False)
export_status = postgresql.ENUM(
    "pending", "processing", "completed", "failed",
    name="lesson_export_status_enum",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    for enum in (
        postgresql.ENUM("draft", "generating", "completed", "failed", "cancelled", name="lesson_plan_status_enum"),
        postgresql.ENUM(
            "lesson_plan", "teaching_notes", "examples", "worksheet", "quiz", "homework", "ppt_outline",
            name="lesson_artifact_type_enum",
        ),
        postgresql.ENUM("pending", "generating", "completed", "failed", name="lesson_artifact_status_enum"),
        postgresql.ENUM("queued", "running", "completed", "failed", "cancelled", name="lesson_job_status_enum"),
        postgresql.ENUM("pdf", "docx", "pptx", name="lesson_export_format_enum"),
        postgresql.ENUM("pending", "processing", "completed", "failed", name="lesson_export_status_enum"),
    ):
        enum.create(bind, checkfirst=True)

    op.create_table(
        "lesson_plans",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("grade", sa.String(length=50), nullable=False),
        sa.Column("subject", sa.String(length=120), nullable=False),
        sa.Column("board", sa.String(length=50), nullable=True),
        sa.Column("chapter_id", sa.String(length=64), nullable=True),
        sa.Column("chapter_name", sa.String(length=255), nullable=False),
        sa.Column("duration_minutes", sa.Integer(), nullable=False),
        sa.Column("learning_objectives", sa.Text(), nullable=False),
        sa.Column("status", lesson_plan_status, nullable=False, server_default="draft"),
        sa.Column("plan_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_lesson_plans_user_id", "lesson_plans", ["user_id"])
    op.create_index("ix_lesson_plans_chapter_id", "lesson_plans", ["chapter_id"])
    op.create_index("ix_lesson_plans_status", "lesson_plans", ["status"])
    op.create_index("ix_lesson_plans_user_id_deleted_at", "lesson_plans", ["user_id", "deleted_at"])

    op.create_table(
        "lesson_plan_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("lesson_plan_id", sa.UUID(), nullable=True),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column("status", job_status, nullable=False, server_default="queued"),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("message", sa.String(length=500), nullable=True),
        sa.Column("requested_artifacts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("input_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["lesson_plan_id"], ["lesson_plans.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_index("ix_lesson_plan_jobs_lesson_plan_id", "lesson_plan_jobs", ["lesson_plan_id"])
    op.create_index("ix_lesson_plan_jobs_user_id", "lesson_plan_jobs", ["user_id"])
    op.create_index("ix_lesson_plan_jobs_status", "lesson_plan_jobs", ["status"])
    op.create_index("ix_lesson_plan_jobs_celery_task_id", "lesson_plan_jobs", ["celery_task_id"])

    op.create_table(
        "lesson_plan_artifacts",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("lesson_plan_id", sa.UUID(), nullable=False),
        sa.Column("job_id", sa.UUID(), nullable=True),
        sa.Column("artifact_type", artifact_type, nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", artifact_status, nullable=False, server_default="pending"),
        sa.Column("version_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["lesson_plan_jobs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["lesson_plan_id"], ["lesson_plans.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("lesson_plan_id", "artifact_type", "version_number", name="uq_lesson_artifact_version"),
    )
    op.create_index("ix_lesson_plan_artifacts_lesson_plan_id", "lesson_plan_artifacts", ["lesson_plan_id"])
    op.create_index("ix_lesson_plan_artifacts_job_id", "lesson_plan_artifacts", ["job_id"])
    op.create_index("ix_lesson_plan_artifacts_type_status", "lesson_plan_artifacts", ["artifact_type", "status"])

    op.create_table(
        "lesson_plan_versions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("lesson_plan_id", sa.UUID(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("change_summary", sa.String(length=500), nullable=True),
        sa.Column("created_by", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["lesson_plan_id"], ["lesson_plans.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_lesson_plan_versions_lesson_plan_id", "lesson_plan_versions", ["lesson_plan_id"])
    op.create_index("ix_lesson_plan_versions_plan_version", "lesson_plan_versions", ["lesson_plan_id", "version_number"])

    op.create_table(
        "lesson_plan_exports",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("lesson_plan_id", sa.UUID(), nullable=False),
        sa.Column("version_id", sa.UUID(), nullable=True),
        sa.Column("export_format", export_format, nullable=False),
        sa.Column("file_path", sa.String(length=500), nullable=True),
        sa.Column("status", export_status, nullable=False, server_default="pending"),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["lesson_plan_id"], ["lesson_plans.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["version_id"], ["lesson_plan_versions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_lesson_plan_exports_lesson_plan_id", "lesson_plan_exports", ["lesson_plan_id"])
    op.create_index("ix_lesson_plan_exports_status", "lesson_plan_exports", ["status"])


def downgrade() -> None:
    op.drop_table("lesson_plan_exports")
    op.drop_table("lesson_plan_versions")
    op.drop_table("lesson_plan_artifacts")
    op.drop_table("lesson_plan_jobs")
    op.drop_table("lesson_plans")

    bind = op.get_bind()
    for name in (
        "lesson_export_status_enum",
        "lesson_export_format_enum",
        "lesson_job_status_enum",
        "lesson_artifact_status_enum",
        "lesson_artifact_type_enum",
        "lesson_plan_status_enum",
    ):
        postgresql.ENUM(name=name).drop(bind, checkfirst=True)
