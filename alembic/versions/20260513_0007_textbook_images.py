"""textbook_images table for diagram retrieval

Revision ID: 20260513_0007
Revises: 20260512_0006
Create Date: 2026-05-13
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260513_0007"
down_revision = "20260512_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "textbook_images",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "textbook_upload_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("textbook_uploads.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("page_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("sequence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("caption", sa.String(512), nullable=True),
        sa.Column("page_text_snippet", sa.String(900), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index(
        "ix_textbook_images_upload_page",
        "textbook_images",
        ["textbook_upload_id", "page_index"],
    )


def downgrade() -> None:
    op.drop_index("ix_textbook_images_upload_page", table_name="textbook_images")
    op.drop_table("textbook_images")
