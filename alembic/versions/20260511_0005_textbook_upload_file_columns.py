"""Add file_path, content_type, content_label, chunk_count to textbook_uploads

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-11
"""

from alembic import op
import sqlalchemy as sa

revision = "20260511_0005"
down_revision = "20260508_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("textbook_uploads", sa.Column("file_path", sa.String(500), nullable=True))
    op.add_column("textbook_uploads", sa.Column("content_type", sa.String(20), nullable=True))
    op.add_column("textbook_uploads", sa.Column("content_label", sa.String(255), nullable=True))
    op.add_column("textbook_uploads", sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("textbook_uploads", "chunk_count")
    op.drop_column("textbook_uploads", "content_label")
    op.drop_column("textbook_uploads", "content_type")
    op.drop_column("textbook_uploads", "file_path")
