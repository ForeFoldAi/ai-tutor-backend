"""Add caption_distance for layout extraction verification

Revision ID: 20260601_0014
Revises: 20260529_0013
"""

from alembic import op
import sqlalchemy as sa

revision = "20260601_0014"
down_revision = "20260529_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("textbook_images", sa.Column("caption_distance", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("textbook_images", "caption_distance")
