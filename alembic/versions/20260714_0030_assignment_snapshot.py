"""assignment_snapshot JSON on users (park teacher assigns when inactive)."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260714_0030"
down_revision = "20260714_0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("assignment_snapshot", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "assignment_snapshot")
