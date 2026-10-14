"""Figure grounding fields for topic-centric retrieval

Revision ID: 20260529_0011
Revises: 20260529_0010
"""

from alembic import op
import sqlalchemy as sa

revision = "20260529_0011"
down_revision = "20260529_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("textbook_images", sa.Column("figure_number", sa.String(32), nullable=True))
    op.add_column("textbook_images", sa.Column("chapter_title", sa.String(255), nullable=True))
    op.add_column("textbook_images", sa.Column("has_caption", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.add_column("textbook_images", sa.Column("nearby_text_before_figure", sa.Text(), nullable=True))
    op.add_column("textbook_images", sa.Column("nearby_text_after_figure", sa.Text(), nullable=True))
    op.add_column("textbook_images", sa.Column("figure_context", sa.Text(), nullable=True))

    conn = op.get_bind()
    conn.execute(
        sa.text(
            """
            UPDATE textbook_images SET
              has_caption = CASE
                WHEN caption IS NOT NULL AND length(trim(caption)) >= 8 THEN true
                ELSE false
              END,
              figure_context = left(
                coalesce(section_title || '. ', '') ||
                coalesce(subsection_title || '. ', '') ||
                coalesce(caption, '') || ' ' ||
                coalesce(page_text_snippet, ''),
                4000
              )
            WHERE figure_context IS NULL OR figure_context = ''
            """
        )
    )


def downgrade() -> None:
    op.drop_column("textbook_images", "figure_context")
    op.drop_column("textbook_images", "nearby_text_after_figure")
    op.drop_column("textbook_images", "nearby_text_before_figure")
    op.drop_column("textbook_images", "has_caption")
    op.drop_column("textbook_images", "chapter_title")
    op.drop_column("textbook_images", "figure_number")
