"""Add per-system sequential school id (API id = seq)."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260714_0028"
down_revision = "20260714_0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("schools", sa.Column("seq", sa.Integer(), nullable=True))
    op.execute(
        sa.text(
            """
            WITH ranked AS (
              SELECT id,
                     ROW_NUMBER() OVER (ORDER BY created_at ASC, id ASC) AS rn
              FROM schools
            )
            UPDATE schools
            SET seq = ranked.rn
            FROM ranked
            WHERE schools.id = ranked.id
            """
        )
    )
    op.execute(sa.text("CREATE SEQUENCE IF NOT EXISTS schools_seq_seq OWNED BY schools.seq"))
    op.execute(
        sa.text(
            "SELECT setval('schools_seq_seq', COALESCE((SELECT MAX(seq) FROM schools), 0))"
        )
    )
    op.alter_column(
        "schools",
        "seq",
        nullable=False,
        server_default=sa.text("nextval('schools_seq_seq')"),
    )
    op.create_unique_constraint("uq_schools_seq", "schools", ["seq"])


def downgrade() -> None:
    op.drop_constraint("uq_schools_seq", "schools", type_="unique")
    op.drop_column("schools", "seq")
    op.execute(sa.text("DROP SEQUENCE IF EXISTS schools_seq_seq"))
