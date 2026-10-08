"""per-student publisher override

Revision ID: 20261008_0046
Revises: 20261008_0045
Create Date: 2026-10-08
"""
from alembic import op
import sqlalchemy as sa

revision = "20261008_0046"
down_revision = "20261008_0045"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("publisher_choices", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "publisher_choices")
