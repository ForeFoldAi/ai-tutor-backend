"""teaching_subjects JSON on users

Revision ID: 20260713_0026
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260713_0026"
down_revision = "20260713_0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("teaching_subjects", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "teaching_subjects")
