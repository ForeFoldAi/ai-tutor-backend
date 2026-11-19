"""Add content_kind and structured_content for ML table/formula assets.

Revision ID: 20260605_0017
Revises: 20260602_0016
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260605_0017"
down_revision = "20260602_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "textbook_images",
        sa.Column("content_kind", sa.String(16), nullable=False, server_default="figure"),
    )
    op.add_column(
        "textbook_images",
        sa.Column("structured_content", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("textbook_images", "structured_content")
    op.drop_column("textbook_images", "content_kind")
