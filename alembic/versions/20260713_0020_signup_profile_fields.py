"""signup profile fields on users and organizations

Revision ID: 20260713_0020
Revises: 20260711_0019
Create Date: 2026-07-13 14:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260713_0020"
down_revision = "20260711_0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("phone", sa.String(length=50), nullable=True))
    op.add_column("users", sa.Column("profile_data", postgresql.JSON(astext_type=sa.Text()), nullable=True))
    op.create_index("ix_users_phone", "users", ["phone"], unique=False)

    op.add_column("organizations", sa.Column("email", sa.String(length=320), nullable=True))
    op.add_column("organizations", sa.Column("website", sa.String(length=500), nullable=True))
    op.add_column("organizations", sa.Column("preferences", postgresql.JSON(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("organizations", "preferences")
    op.drop_column("organizations", "website")
    op.drop_column("organizations", "email")

    op.drop_index("ix_users_phone", table_name="users")
    op.drop_column("users", "profile_data")
    op.drop_column("users", "phone")
