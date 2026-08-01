"""Credential share/delivery tracking on users."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260714_0031"
down_revision = "20260714_0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("credentials_generated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("credentials_shared_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("credential_delivery_status", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "credential_delivery_status")
    op.drop_column("users", "credentials_shared_at")
    op.drop_column("users", "credentials_generated_at")
