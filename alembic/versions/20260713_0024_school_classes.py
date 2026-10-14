"""School-scoped classes and subjects tables."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260713_0024"
down_revision = "20260713_0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "school_subjects",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("school_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["school_id"], ["schools.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("school_id", "code", name="uq_school_subjects_school_code"),
    )
    op.create_index("ix_school_subjects_school_id", "school_subjects", ["school_id"])

    op.create_table(
        "school_classes",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("school_id", sa.UUID(), nullable=False),
        sa.Column("grade", sa.String(length=20), nullable=False),
        sa.Column("section", sa.String(length=20), nullable=False),
        sa.Column("curriculum", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["school_id"], ["schools.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "school_id",
            "grade",
            "section",
            "curriculum",
            name="uq_school_classes_school_grade_section_curriculum",
        ),
    )
    op.create_index("ix_school_classes_school_id", "school_classes", ["school_id"])

    op.create_table(
        "school_class_subjects",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("school_class_id", sa.UUID(), nullable=False),
        sa.Column("subject_id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["school_class_id"], ["school_classes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subject_id"], ["school_subjects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("school_class_id", "subject_id", name="uq_school_class_subjects"),
    )
    op.create_index("ix_school_class_subjects_school_class_id", "school_class_subjects", ["school_class_id"])
    op.create_index("ix_school_class_subjects_subject_id", "school_class_subjects", ["subject_id"])


def downgrade() -> None:
    op.drop_index("ix_school_class_subjects_subject_id", table_name="school_class_subjects")
    op.drop_index("ix_school_class_subjects_school_class_id", table_name="school_class_subjects")
    op.drop_table("school_class_subjects")
    op.drop_index("ix_school_classes_school_id", table_name="school_classes")
    op.drop_table("school_classes")
    op.drop_index("ix_school_subjects_school_id", table_name="school_subjects")
    op.drop_table("school_subjects")
