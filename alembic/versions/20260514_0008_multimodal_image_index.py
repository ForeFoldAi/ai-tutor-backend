"""Add multimodal index fields to textbook_images

Revision ID: 20260514_0008
Revises: 20260513_0007
Create Date: 2026-05-14
"""

from alembic import op
import sqlalchemy as sa

revision = "20260514_0008"
down_revision = "20260513_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "textbook_images",
        sa.Column("multimodal_indexed", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "textbook_images",
        sa.Column("embedding_model", sa.String(120), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("textbook_images", "embedding_model")
    op.drop_column("textbook_images", "multimodal_indexed")
