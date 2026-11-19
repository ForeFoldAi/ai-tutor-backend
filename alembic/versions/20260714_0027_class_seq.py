"""Add per-school sequential class id (API id = seq)."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260714_0027"
down_revision = "20260713_0026"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("school_classes", sa.Column("seq", sa.Integer(), nullable=True))
    # Assign seq 1,2,3… per school by created_at
    op.execute(
        sa.text(
            """
            WITH ranked AS (
              SELECT id,
                     ROW_NUMBER() OVER (
                       PARTITION BY school_id
                       ORDER BY created_at ASC, id ASC
                     ) AS rn
              FROM school_classes
            )
            UPDATE school_classes
            SET seq = ranked.rn
            FROM ranked
            WHERE school_classes.id = ranked.id
            """
        )
    )
    op.alter_column("school_classes", "seq", nullable=False)
    op.create_unique_constraint("uq_school_classes_school_seq", "school_classes", ["school_id", "seq"])


def downgrade() -> None:
    op.drop_constraint("uq_school_classes_school_seq", "school_classes", type_="unique")
    op.drop_column("school_classes", "seq")
