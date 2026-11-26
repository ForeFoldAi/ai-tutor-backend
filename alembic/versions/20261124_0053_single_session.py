"""one active login per student: session id on refresh sessions + the user's current session

Revision ID: 20261124_0053
Revises: 20261119_0052
Create Date: 2026-11-24
"""
from alembic import op
import sqlalchemy as sa

revision = "20261124_0053"
down_revision = "20261119_0052"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Stable id of one login; copied onto every rotated refresh row so a login stays one "session".
    op.add_column("sessions", sa.Column("sid", sa.String(length=36), nullable=True))
    op.create_index("ix_sessions_sid", "sessions", ["sid"])
    # The one session a single-session user (students) may currently use. NULL = nothing enforced yet.
    op.add_column("users", sa.Column("active_sid", sa.String(length=36), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "active_sid")
    op.drop_index("ix_sessions_sid", table_name="sessions")
    op.drop_column("sessions", "sid")
