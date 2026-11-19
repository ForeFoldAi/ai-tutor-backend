"""master admin catalog tables and enums

Revision ID: 20260508_0004
Revises: 20260507_0003
Create Date: 2026-05-08
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260508_0004"
down_revision = "20260507_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    board_enum = postgresql.ENUM("CBSE", "ICSE", "STATE_BOARD", "IB", "CAMBRIDGE", name="board_enum")
    class_enum = postgresql.ENUM(
        "CLASS_1",
        "CLASS_2",
        "CLASS_3",
        "CLASS_4",
        "CLASS_5",
        "CLASS_6",
        "CLASS_7",
        "CLASS_8",
        "CLASS_9",
        "CLASS_10",
        name="class_enum",
    )
    processing_status_enum = postgresql.ENUM("QUEUED", "PROCESSING", "EMBEDDED", "FAILED", name="processing_status_enum")

    bind = op.get_bind()
    board_enum.create(bind, checkfirst=True)
    class_enum.create(bind, checkfirst=True)
    processing_status_enum.create(bind, checkfirst=True)
    board_enum_ref = postgresql.ENUM(name="board_enum", create_type=False)
    class_enum_ref = postgresql.ENUM(name="class_enum", create_type=False)
    processing_status_enum_ref = postgresql.ENUM(name="processing_status_enum", create_type=False)

    op.create_table(
        "board_definitions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("board", board_enum_ref, nullable=False),
        sa.Column("country", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_board_definitions_board", "board_definitions", ["board"], unique=True)

    op.create_table(
        "syllabus_subjects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("board", board_enum_ref, nullable=False),
        sa.Column("class_level", class_enum_ref, nullable=False),
        sa.Column("subject_name", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("board", "class_level", "subject_name", name="uq_syllabus_subject"),
    )
    op.create_index("ix_syllabus_subjects_board", "syllabus_subjects", ["board"], unique=False)
    op.create_index("ix_syllabus_subjects_class_level", "syllabus_subjects", ["class_level"], unique=False)

    op.create_table(
        "textbook_uploads",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column("board", board_enum_ref, nullable=False),
        sa.Column("class_level", class_enum_ref, nullable=False),
        sa.Column("subject_name", sa.String(length=120), nullable=False),
        sa.Column("chapter", sa.String(length=150), nullable=True),
        sa.Column("uploaded_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("upload_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ocr_status", processing_status_enum_ref, nullable=False),
        sa.Column("chunk_status", processing_status_enum_ref, nullable=False),
        sa.Column("embedding_status", processing_status_enum_ref, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_textbook_uploads_board", "textbook_uploads", ["board"], unique=False)
    op.create_index("ix_textbook_uploads_class_level", "textbook_uploads", ["class_level"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_textbook_uploads_class_level", table_name="textbook_uploads")
    op.drop_index("ix_textbook_uploads_board", table_name="textbook_uploads")
    op.drop_table("textbook_uploads")

    op.drop_index("ix_syllabus_subjects_class_level", table_name="syllabus_subjects")
    op.drop_index("ix_syllabus_subjects_board", table_name="syllabus_subjects")
    op.drop_table("syllabus_subjects")

    op.drop_index("ix_board_definitions_board", table_name="board_definitions")
    op.drop_table("board_definitions")

    bind = op.get_bind()
    sa.Enum(name="processing_status_enum").drop(bind, checkfirst=True)
    sa.Enum(name="class_enum").drop(bind, checkfirst=True)
    sa.Enum(name="board_enum").drop(bind, checkfirst=True)
