"""Layout extraction: image/caption bboxes and pairing confidence

Revision ID: 20260529_0013
Revises: 20260529_0012
"""

from alembic import op
import sqlalchemy as sa

revision = "20260529_0013"
down_revision = "20260529_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("textbook_images", sa.Column("image_bbox", sa.String(length=128), nullable=True))
    op.add_column("textbook_images", sa.Column("caption_bbox", sa.String(length=128), nullable=True))
    op.add_column("textbook_images", sa.Column("pairing_confidence", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("textbook_images", "pairing_confidence")
    op.drop_column("textbook_images", "caption_bbox")
    op.drop_column("textbook_images", "image_bbox")
