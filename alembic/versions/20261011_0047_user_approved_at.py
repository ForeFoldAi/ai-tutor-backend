"""admin approval for public signups

Revision ID: 20261011_0047
Revises: 20261008_0046
Create Date: 2026-10-11
"""
from alembic import op
import sqlalchemy as sa

revision = "20261011_0047"
down_revision = "20261008_0046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE users SET approved_at = created_at")


def downgrade() -> None:
    op.drop_column("users", "approved_at")
