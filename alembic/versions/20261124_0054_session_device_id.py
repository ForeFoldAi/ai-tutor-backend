"""per-browser device id on refresh sessions (same browser re-login is not prompted)

Revision ID: 20261124_0054
Revises: 20261124_0053
Create Date: 2026-11-24
"""
from alembic import op
import sqlalchemy as sa

revision = "20261124_0054"
down_revision = "20261124_0053"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("sessions", sa.Column("device_id", sa.String(length=64), nullable=True))
    op.create_index("ix_sessions_device_id", "sessions", ["device_id"])


def downgrade() -> None:
    op.drop_index("ix_sessions_device_id", table_name="sessions")
    op.drop_column("sessions", "device_id")
