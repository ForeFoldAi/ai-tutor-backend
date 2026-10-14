"""Add production metadata fields to textbook_images

Adds: generated_caption, content_hash, semantic_keywords, educational_tags,
grade_level, subject, ocr_text for production-grade image retrieval.

Revision ID: 20260601_0015
Revises: 20260601_0014
"""

from alembic import op
import sqlalchemy as sa

revision = "20260601_0015"
down_revision = "20260601_0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Auto-generated caption for figures with only a number label or no caption
    op.add_column(
        "textbook_images",
        sa.Column("generated_caption", sa.Text(), nullable=True),
    )
    # SHA-256 hex digest of image bytes — deduplication and change detection
    op.add_column(
        "textbook_images",
        sa.Column("content_hash", sa.String(64), nullable=True),
    )
    # Pipe-separated semantic keywords extracted from figure context
    op.add_column(
        "textbook_images",
        sa.Column("semantic_keywords", sa.String(512), nullable=True),
    )
    # Pipe-separated educational tags (subject-domain labels, e.g. biology|diagram|cell)
    op.add_column(
        "textbook_images",
        sa.Column("educational_tags", sa.String(256), nullable=True),
    )
    # Denormalized from TextbookUpload.class_level for grade-appropriate filtering
    op.add_column(
        "textbook_images",
        sa.Column("grade_level", sa.String(12), nullable=True),
    )
    # Denormalized from TextbookUpload.subject_name for subject-scoped retrieval
    op.add_column(
        "textbook_images",
        sa.Column("subject", sa.String(120), nullable=True),
    )
    # OCR-extracted text from the image pixels (for scanned PDFs without text layer)
    op.add_column(
        "textbook_images",
        sa.Column("ocr_text", sa.Text(), nullable=True),
    )

    # Index on content_hash for fast deduplication lookups
    op.create_index(
        "ix_textbook_images_content_hash",
        "textbook_images",
        ["content_hash"],
        unique=False,
    )
    # Index for grade+subject scoped retrieval without join
    op.create_index(
        "ix_textbook_images_grade_subject",
        "textbook_images",
        ["grade_level", "subject"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_textbook_images_grade_subject", table_name="textbook_images")
    op.drop_index("ix_textbook_images_content_hash", table_name="textbook_images")
    op.drop_column("textbook_images", "ocr_text")
    op.drop_column("textbook_images", "subject")
    op.drop_column("textbook_images", "grade_level")
    op.drop_column("textbook_images", "educational_tags")
    op.drop_column("textbook_images", "semantic_keywords")
    op.drop_column("textbook_images", "content_hash")
    op.drop_column("textbook_images", "generated_caption")
