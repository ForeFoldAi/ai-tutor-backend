"""syllabus subject cover image

Revision ID: 20261012_0048
Revises: 20261011_0047
Create Date: 2026-10-12
"""
from alembic import op
import sqlalchemy as sa

revision = "20261012_0048"
down_revision = "20261011_0047"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("syllabus_subjects", sa.Column("image_path", sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column("syllabus_subjects", "image_path")
