"""Alembic revision: textbook segmentation tables.

Revision ID: 20261004_0043
Revises: 20260724_0042
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20261004_0043"
down_revision = "20260724_0042"
branch_labels = None
depends_on = None


def upgrade() -> None:
    board_enum_ref = postgresql.ENUM(name="board_enum", create_type=False)
    class_enum_ref = postgresql.ENUM(name="class_enum", create_type=False)

    op.create_table(
        "textbooks",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("file_path", sa.String(500), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=True),
        sa.Column("board", board_enum_ref, nullable=False),
        sa.Column("class_level", class_enum_ref, nullable=False),
        sa.Column("subject_name", sa.String(120), nullable=False),
        sa.Column("total_pages", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pdf_type", sa.String(32), nullable=False, server_default="text_based"),
        sa.Column("status", sa.String(32), nullable=False, server_default="UPLOADED"),
        sa.Column("detection_summary", sa.Text(), nullable=True),
        sa.Column(
            "uploaded_by",
            sa.BigInteger(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_textbooks_board", "textbooks", ["board"])
    op.create_index("ix_textbooks_class_level", "textbooks", ["class_level"])
    op.create_index("ix_textbooks_subject_name", "textbooks", ["subject_name"])
    op.create_index("ix_textbooks_file_hash", "textbooks", ["file_hash"])

    op.create_table(
        "textbook_chapters",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "textbook_id",
            sa.BigInteger(),
            sa.ForeignKey("textbooks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chapter_number", sa.String(32), nullable=False),
        sa.Column("chapter_title", sa.String(255), nullable=False),
        sa.Column("start_pdf_page", sa.Integer(), nullable=False),
        sa.Column("end_pdf_page", sa.Integer(), nullable=False),
        sa.Column("printed_start_page", sa.String(32), nullable=True),
        sa.Column("printed_end_page", sa.String(32), nullable=True),
        sa.Column("is_non_chapter_section", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("section_type", sa.String(32), nullable=False, server_default="chapter"),
        sa.Column("detection_method", sa.String(32), nullable=False, server_default="toc_body_match"),
        sa.Column("confidence_score", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("confidence_flags", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="DETECTED"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "textbook_upload_id",
            sa.BigInteger(),
            sa.ForeignKey("textbook_uploads.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_textbook_chapters_textbook_id", "textbook_chapters", ["textbook_id"])
    op.create_index("ix_textbook_chapters_upload_id", "textbook_chapters", ["textbook_upload_id"])


def downgrade() -> None:
    op.drop_index("ix_textbook_chapters_upload_id", table_name="textbook_chapters")
    op.drop_index("ix_textbook_chapters_textbook_id", table_name="textbook_chapters")
    op.drop_table("textbook_chapters")

    op.drop_index("ix_textbooks_file_hash", table_name="textbooks")
    op.drop_index("ix_textbooks_subject_name", table_name="textbooks")
    op.drop_index("ix_textbooks_class_level", table_name="textbooks")
    op.drop_index("ix_textbooks_board", table_name="textbooks")
    op.drop_table("textbooks")
