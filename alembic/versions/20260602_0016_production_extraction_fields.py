"""Add production extraction fields to textbook_images (Stage 2-9).

Revision ID: 20260602_0016
Revises: 20260601_0015

New columns:
  title                — educational short title (≤80 chars, includes section context)
  educational_description — 1-2 sentence description with chapter/section prefix
  concept_tags         — pipe-separated curriculum concept tags
  phash                — 64-bit perceptual hash stored as signed BigInteger (Stage 9)
  source_type          — figure discovery source: embedded_image|vector_drawing|layout_detected
  caption_source       — how caption was found: detected|none
  page_coverage        — image area / page area ratio (0.0–1.0)
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260602_0016"
down_revision = "20260601_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "textbook_images",
        sa.Column("title", sa.String(255), nullable=True),
    )
    op.add_column(
        "textbook_images",
        sa.Column("educational_description", sa.Text(), nullable=True),
    )
    op.add_column(
        "textbook_images",
        sa.Column("concept_tags", sa.String(512), nullable=True),
    )
    # Numeric perceptual hash (signed int64 for PostgreSQL bigint).
    # Hamming distance threshold is _PHASH_HAMMING_THRESHOLD in textbook_image_extraction.py.
    op.add_column(
        "textbook_images",
        sa.Column("phash", sa.BigInteger(), nullable=True),
    )
    op.create_index(
        "ix_textbook_images_phash",
        "textbook_images",
        ["phash"],
        unique=False,
    )
    op.add_column(
        "textbook_images",
        sa.Column(
            "source_type",
            sa.String(32),
            nullable=True,
            server_default="embedded_image",
        ),
    )
    op.add_column(
        "textbook_images",
        sa.Column(
            "caption_source",
            sa.String(32),
            nullable=True,
            server_default="none",
        ),
    )
    op.add_column(
        "textbook_images",
        sa.Column("page_coverage", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("textbook_images", "page_coverage")
    op.drop_column("textbook_images", "caption_source")
    op.drop_column("textbook_images", "source_type")
    op.drop_index("ix_textbook_images_phash", table_name="textbook_images")
    op.drop_column("textbook_images", "phash")
    op.drop_column("textbook_images", "concept_tags")
    op.drop_column("textbook_images", "educational_description")
    op.drop_column("textbook_images", "title")
