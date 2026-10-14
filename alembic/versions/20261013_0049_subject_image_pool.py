"""subject image pool: shared images picked per subject

Revision ID: 20261013_0049
Revises: 20261012_0048
Create Date: 2026-10-13
"""
from alembic import op
import sqlalchemy as sa

revision = "20261013_0049"
down_revision = "20261012_0048"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "subject_images",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("image_path", sa.String(500), nullable=False),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.add_column(
        "syllabus_subjects",
        sa.Column("image_id", sa.BigInteger(), sa.ForeignKey("subject_images.id"), nullable=True),
    )
    op.create_index("ix_syllabus_subjects_image_id", "syllabus_subjects", ["image_id"])
    # Keep already-uploaded covers: one pool row per distinct stored key.
    op.execute(
        """
        INSERT INTO subject_images (image_path, file_name)
        SELECT DISTINCT image_path, regexp_replace(image_path, '^.*/', '')
        FROM syllabus_subjects WHERE image_path IS NOT NULL
        """
    )
    op.execute(
        """
        UPDATE syllabus_subjects s SET image_id = i.id
        FROM subject_images i WHERE s.image_path = i.image_path
        """
    )
    op.drop_column("syllabus_subjects", "image_path")


def downgrade() -> None:
    op.add_column("syllabus_subjects", sa.Column("image_path", sa.String(500), nullable=True))
    op.execute(
        """
        UPDATE syllabus_subjects s SET image_path = i.image_path
        FROM subject_images i WHERE s.image_id = i.id
        """
    )
    op.drop_index("ix_syllabus_subjects_image_id", table_name="syllabus_subjects")
    op.drop_column("syllabus_subjects", "image_id")
    op.drop_table("subject_images")
