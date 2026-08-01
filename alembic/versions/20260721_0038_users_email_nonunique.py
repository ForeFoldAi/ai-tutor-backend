"""Allow shared emails across users (siblings / teacher+parent).

Revision ID: 20260721_0038
Revises: 20260721_0037
Create Date: 2026-07-21
"""

from __future__ import annotations

from alembic import op

revision = "20260721_0038"
down_revision = "20260721_0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_users_email", table_name="users")
    op.create_index("ix_users_email", "users", ["email"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_users_email", table_name="users")
    op.create_index("ix_users_email", "users", ["email"], unique=True)
