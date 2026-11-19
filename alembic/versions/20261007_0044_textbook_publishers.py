"""textbook publishers + per class-subject book choice

Revision ID: 20261007_0044
Revises: 4c4e812a8d1e
Create Date: 2026-10-07
"""
from alembic import op
import sqlalchemy as sa


revision = "20261007_0044"
down_revision = "4c4e812a8d1e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("textbooks", sa.Column("publisher", sa.String(120), nullable=True))
    # Existing books stay visible to every class exactly as before.
    op.add_column("textbooks", sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.true()))

    op.add_column(
        "school_class_subjects",
        sa.Column(
            "textbook_id",
            sa.BigInteger(),
            sa.ForeignKey("textbooks.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_school_class_subjects_textbook_id", "school_class_subjects", ["textbook_id"])


def downgrade() -> None:
    op.drop_index("ix_school_class_subjects_textbook_id", "school_class_subjects")
    op.drop_column("school_class_subjects", "textbook_id")
    op.drop_column("textbooks", "is_default")
    op.drop_column("textbooks", "publisher")
