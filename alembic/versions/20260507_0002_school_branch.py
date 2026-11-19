"""school branch column

Revision ID: 20260507_0002
Revises: 20260505_0001
Create Date: 2026-05-07

"""

import sqlalchemy as sa
from alembic import op

revision = "20260507_0002"
down_revision = "20260505_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("schools", sa.Column("branch", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("schools", "branch")
