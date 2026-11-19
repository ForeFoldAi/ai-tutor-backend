"""textbook image visual summary (vision-model description, caption-independent)

Revision ID: 20261107_0051
Revises: 20261026_0050
Create Date: 2026-11-07
"""
from alembic import op
import sqlalchemy as sa

revision = "20261107_0051"
down_revision = "20261026_0050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("textbook_images", sa.Column("visual_summary", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("textbook_images", "visual_summary")
