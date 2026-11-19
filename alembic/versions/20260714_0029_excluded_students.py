"""excluded_student_ids JSON on users (tutor student unassign)."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260714_0029"
down_revision = "20260714_0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("excluded_student_ids", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "excluded_student_ids")
