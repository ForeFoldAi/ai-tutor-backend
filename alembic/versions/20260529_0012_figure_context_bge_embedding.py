"""Store BGE embeddings for figure_context

Revision ID: 20260529_0012
Revises: 20260529_0011
"""

from alembic import op
import sqlalchemy as sa

revision = "20260529_0012"
down_revision = "20260529_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "textbook_images",
        sa.Column("context_embedding_bge", sa.LargeBinary(), nullable=True),
    )
    op.add_column(
        "textbook_images",
        sa.Column("context_bge_indexed", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("textbook_images", "context_bge_indexed")
    op.drop_column("textbook_images", "context_embedding_bge")
