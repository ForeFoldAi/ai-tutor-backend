"""Add per-school sequential subject id; clear existing subject rows."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260713_0025"
down_revision = "20260713_0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("DELETE FROM school_class_subjects"))
    op.execute(sa.text("DELETE FROM school_subjects"))

    op.add_column("school_subjects", sa.Column("seq", sa.Integer(), nullable=True))
    op.execute(sa.text("UPDATE school_subjects SET seq = 1 WHERE seq IS NULL"))
    op.alter_column("school_subjects", "seq", nullable=False)
    op.create_unique_constraint("uq_school_subjects_school_seq", "school_subjects", ["school_id", "seq"])


def downgrade() -> None:
    op.drop_constraint("uq_school_subjects_school_seq", "school_subjects", type_="unique")
    op.drop_column("school_subjects", "seq")
