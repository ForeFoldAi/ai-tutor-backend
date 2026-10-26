"""textbook upload processing start/finish times

Revision ID: 20261026_0050
Revises: 20261013_0049
Create Date: 2026-10-26
"""
from alembic import op
import sqlalchemy as sa

revision = "20261026_0050"
down_revision = "20261013_0049"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("textbook_uploads", sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("textbook_uploads", sa.Column("processing_finished_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("textbook_uploads", "processing_finished_at")
    op.drop_column("textbook_uploads", "processing_started_at")
