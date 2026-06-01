"""Add pedagogy-centric fields to textbook_images

Revision ID: 20260529_0009
Revises: 20260514_0008
Create Date: 2026-05-29

Adds four new columns that support the pedagogy-centric multimodal retrieval system:
- image_type: rule-based figure type (landscape, wildlife, map, diagram, etc.)
- caption_normalized: stop-word-stripped lowercase caption for BGE scoring
- is_decorative: True when the figure carries no educational signal
- educational_salience: 0-1 float proxy for curricular importance
"""

from alembic import op
import sqlalchemy as sa

revision = "20260529_0009"
down_revision = "20260514_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "textbook_images",
        sa.Column("image_type", sa.String(32), nullable=False, server_default="unknown"),
    )
    op.add_column(
        "textbook_images",
        sa.Column("caption_normalized", sa.String(512), nullable=True),
    )
    op.add_column(
        "textbook_images",
        sa.Column("is_decorative", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "textbook_images",
        sa.Column("educational_salience", sa.Float(), nullable=False, server_default="0.5"),
    )

    # Best-effort backfill: classify existing rows based on caption text using simple SQL ILIKE
    # Ordered from most-specific to most-general so first match wins.
    _backfill_image_types()


def _backfill_image_types() -> None:
    """Classify existing rows via SQL pattern matching on caption + page_text_snippet."""
    conn = op.get_bind()

    type_patterns = [
        ("satellite_image", ["%satellite%", "%aerial%", "%from above%", "%top view%"]),
        ("map", ["%map%", "%distribution%", "% region %", "%boundary%", "%border%"]),
        ("wildlife", [
            "%gharial%", "%crocodile%", "%tiger%", "%lion%", "%elephant%",
            "%leopard%", "%peacock%", "%macaque%", "%cobra%", "%species%",
            "%fauna%", "%wildlife%", "%bird%", "%reptile%",
        ]),
        ("chart", ["%graph%", "%chart%", "%statistics%", "%data %", "%percentage%"]),
        ("diagram", ["%diagram%", "%cross section%", "%cross-section%", "%process%", "%cycle%", "%structure%", "%formation%"]),
        ("historical_photo", ["%fort%", "%palace%", "%temple%", "%monument%", "%heritage%", "%ancient%", "%ruins%"]),
        ("landscape", [
            "%mountain%", "%himalaya%", "%glacier%", "%waterfall%", "%desert%",
            "%sand dune%", "%plain%", "%plateau%", "%valley%", "%lake%",
            "%river%", "%coast%", "%beach%", "%island%", "%forest%",
        ]),
    ]

    for img_type, patterns in type_patterns:
        like_clauses = " OR ".join(
            f"(caption ILIKE '{p}' OR page_text_snippet ILIKE '{p}')"
            for p in patterns
        )
        conn.execute(
            sa.text(
                f"UPDATE textbook_images SET image_type = '{img_type}' "
                f"WHERE image_type = 'unknown' AND ({like_clauses})"
            )
        )

    # Backfill caption_normalized: lowercase, strip fig prefix
    conn.execute(
        sa.text(
            "UPDATE textbook_images "
            "SET caption_normalized = LOWER(TRIM(REGEXP_REPLACE(caption, "
            r"'(?i)^fig\.?\s*\d+(\.\d+)*\s*[:.]\s*', ''))) "
            "WHERE caption IS NOT NULL AND caption_normalized IS NULL"
        )
    )

    # Educational salience: crude estimate based on caption length
    conn.execute(
        sa.text(
            "UPDATE textbook_images "
            "SET educational_salience = LEAST(1.0, CHAR_LENGTH(COALESCE(caption,'')) / 200.0) "
            "WHERE educational_salience = 0.5 AND caption IS NOT NULL"
        )
    )


def downgrade() -> None:
    op.drop_column("textbook_images", "educational_salience")
    op.drop_column("textbook_images", "is_decorative")
    op.drop_column("textbook_images", "caption_normalized")
    op.drop_column("textbook_images", "image_type")
