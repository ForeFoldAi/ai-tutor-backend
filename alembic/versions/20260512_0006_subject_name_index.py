"""Add subject_name index to textbook_uploads for faster student queries

Revision ID: 20260512_0006
Revises: 20260511_0005
Create Date: 2026-05-12
"""

from alembic import op

revision = "20260512_0006"
down_revision = "20260511_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_textbook_uploads_subject_name",
        "textbook_uploads",
        ["subject_name"],
    )


def downgrade() -> None:
    op.drop_index("ix_textbook_uploads_subject_name", table_name="textbook_uploads")
