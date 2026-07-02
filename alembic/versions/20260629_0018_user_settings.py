"""user_settings table for profile preferences and notifications

Revision ID: 20260629_0018
Revises: 20260605_0017
Create Date: 2026-06-29

"""

import sqlalchemy as sa
from alembic import op

revision = "20260629_0018"
down_revision = "20260605_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_settings",
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=True),
        sa.Column("language", sa.String(length=32), nullable=False, server_default="en"),
        sa.Column("theme", sa.String(length=16), nullable=False, server_default="light"),
        sa.Column("notify_email", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("notify_push", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("notify_assignments", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("notify_sessions", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("notify_messages", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("username"),
    )
    op.create_index("ix_user_settings_username", "user_settings", ["username"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_user_settings_username", table_name="user_settings")
    op.drop_table("user_settings")
