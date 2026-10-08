"""add_parent_and_hierarchy_to_textbook_chapters

Revision ID: 4c4e812a8d1e
Revises: c979a3f6b816
Create Date: 2026-10-03 19:27:36.012440
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4c4e812a8d1e'
down_revision = 'c979a3f6b816'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "textbook_chapters",
        sa.Column(
            "parent_id",
            sa.BigInteger(),
            sa.ForeignKey("textbook_chapters.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "textbook_chapters",
        sa.Column("hierarchy_level", sa.String(32), nullable=False, server_default="chapter"),
    )
    op.create_index("ix_textbook_chapters_parent_id", "textbook_chapters", ["parent_id"])

    op.add_column("textbooks", sa.Column("page_offset", sa.Integer(), nullable=True))
    op.add_column("textbooks", sa.Column("confidence_score", sa.Float(), nullable=True))
    op.add_column("textbooks", sa.Column("detection_method", sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column("textbooks", "detection_method")
    op.drop_column("textbooks", "confidence_score")
    op.drop_column("textbooks", "page_offset")

    op.drop_index("ix_textbook_chapters_parent_id", "textbook_chapters")
    op.drop_column("textbook_chapters", "hierarchy_level")
    op.drop_column("textbook_chapters", "parent_id")

