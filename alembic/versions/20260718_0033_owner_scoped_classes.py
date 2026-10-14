"""Owner-scoped classes/subjects for individual tutors (no school).

Revision ID: 20260718_0033
Revises: 20260714_0032
Create Date: 2026-07-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260718_0033"
down_revision = "20260714_0032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "school_subjects",
        sa.Column("owner_user_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "school_classes",
        sa.Column("owner_user_id", sa.BigInteger(), nullable=True),
    )

    op.alter_column("school_subjects", "school_id", existing_type=sa.BigInteger(), nullable=True)
    op.alter_column("school_classes", "school_id", existing_type=sa.BigInteger(), nullable=True)

    op.drop_constraint("uq_school_subjects_school_code", "school_subjects", type_="unique")
    op.drop_constraint("uq_school_subjects_school_seq", "school_subjects", type_="unique")
    op.drop_constraint(
        "uq_school_classes_school_grade_section_curriculum",
        "school_classes",
        type_="unique",
    )
    op.drop_constraint("uq_school_classes_school_seq", "school_classes", type_="unique")

    op.create_index("ix_school_subjects_owner_user_id", "school_subjects", ["owner_user_id"])
    op.create_index("ix_school_classes_owner_user_id", "school_classes", ["owner_user_id"])

    # School-scoped uniqueness (existing rows)
    op.execute(
        """
        CREATE UNIQUE INDEX uq_school_subjects_school_code
        ON school_subjects (school_id, code)
        WHERE school_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_school_subjects_school_seq
        ON school_subjects (school_id, seq)
        WHERE school_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_school_classes_school_grade_section_curriculum
        ON school_classes (school_id, grade, section, curriculum)
        WHERE school_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_school_classes_school_seq
        ON school_classes (school_id, seq)
        WHERE school_id IS NOT NULL
        """
    )

    # Individual-tutor uniqueness
    op.execute(
        """
        CREATE UNIQUE INDEX uq_tutor_subjects_owner_code
        ON school_subjects (owner_user_id, code)
        WHERE owner_user_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_tutor_subjects_owner_seq
        ON school_subjects (owner_user_id, seq)
        WHERE owner_user_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_tutor_classes_owner_grade_section_curriculum
        ON school_classes (owner_user_id, grade, section, curriculum)
        WHERE owner_user_id IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_tutor_classes_owner_seq
        ON school_classes (owner_user_id, seq)
        WHERE owner_user_id IS NOT NULL
        """
    )

    op.create_foreign_key(
        "fk_school_subjects_owner_user_id",
        "school_subjects",
        "users",
        ["owner_user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_school_classes_owner_user_id",
        "school_classes",
        "users",
        ["owner_user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Exactly one of school_id / owner_user_id
    op.execute(
        """
        ALTER TABLE school_subjects
        ADD CONSTRAINT ck_school_subjects_school_xor_owner
        CHECK (
            (school_id IS NOT NULL AND owner_user_id IS NULL)
            OR (school_id IS NULL AND owner_user_id IS NOT NULL)
        )
        """
    )
    op.execute(
        """
        ALTER TABLE school_classes
        ADD CONSTRAINT ck_school_classes_school_xor_owner
        CHECK (
            (school_id IS NOT NULL AND owner_user_id IS NULL)
            OR (school_id IS NULL AND owner_user_id IS NOT NULL)
        )
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE school_classes DROP CONSTRAINT IF EXISTS ck_school_classes_school_xor_owner")
    op.execute("ALTER TABLE school_subjects DROP CONSTRAINT IF EXISTS ck_school_subjects_school_xor_owner")
    op.drop_constraint("fk_school_classes_owner_user_id", "school_classes", type_="foreignkey")
    op.drop_constraint("fk_school_subjects_owner_user_id", "school_subjects", type_="foreignkey")
    op.execute("DROP INDEX IF EXISTS uq_tutor_classes_owner_seq")
    op.execute("DROP INDEX IF EXISTS uq_tutor_classes_owner_grade_section_curriculum")
    op.execute("DROP INDEX IF EXISTS uq_tutor_subjects_owner_seq")
    op.execute("DROP INDEX IF EXISTS uq_tutor_subjects_owner_code")
    op.execute("DROP INDEX IF EXISTS uq_school_classes_school_seq")
    op.execute("DROP INDEX IF EXISTS uq_school_classes_school_grade_section_curriculum")
    op.execute("DROP INDEX IF EXISTS uq_school_subjects_school_seq")
    op.execute("DROP INDEX IF EXISTS uq_school_subjects_school_code")
    op.drop_index("ix_school_classes_owner_user_id", table_name="school_classes")
    op.drop_index("ix_school_subjects_owner_user_id", table_name="school_subjects")
    op.drop_column("school_classes", "owner_user_id")
    op.drop_column("school_subjects", "owner_user_id")
    # school_id back to NOT NULL would fail if owner rows exist — leave nullable on downgrade
