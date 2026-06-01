"""tutor teaching_board and teaching_grades on users

Revision ID: 20260507_0003
Revises: 20260507_0002
Create Date: 2026-05-07

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260507_0003"
down_revision = "20260507_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("teaching_board", sa.String(length=100), nullable=True))
    op.add_column(
        "users",
        sa.Column("teaching_grades", postgresql.JSON(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "teaching_grades")
    op.drop_column("users", "teaching_board")
