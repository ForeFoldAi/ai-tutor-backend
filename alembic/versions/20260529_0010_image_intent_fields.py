"""Add section grounding and educational role to textbook_images

Revision ID: 20260529_0010
Revises: 20260529_0009
Create Date: 2026-05-29

Adds:
  section_title    — section heading detected near the figure during PDF ingestion
  subsection_title — sub-heading detected near the figure
  educational_role — primary_concept / supporting_example / sidebar_example / decorative / activity

Extends image_type with new fine-grained types:
  weather_station, instrument, process, activity
(existing rows classified as 'unknown' or broad types are backfilled using SQL ILIKE)
"""

from alembic import op
import sqlalchemy as sa

revision = "20260529_0010"
down_revision = "20260529_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "textbook_images",
        sa.Column("section_title", sa.String(255), nullable=True),
    )
    op.add_column(
        "textbook_images",
        sa.Column("subsection_title", sa.String(255), nullable=True),
    )
    op.add_column(
        "textbook_images",
        sa.Column("educational_role", sa.String(32), nullable=False, server_default="unknown"),
    )

    _backfill(op.get_bind())


def _backfill(conn) -> None:
    # Upgrade existing image_type rows to finer types where caption gives clues
    finer_type_patterns = [
        ("weather_station", [
            "%weather station%", "%automated weather%", "%aws%", "%met station%",
            "%meteorological station%",
        ]),
        ("instrument", [
            "%instrument%", "%sensor%", "%gauge%", "%thermometer%",
            "%barometer%", "%anemometer%", "%hygrometer%", "%apparatus%",
            "%equipment%", "%device%",
        ]),
        ("process", [
            "%step%", "%stage%", "%procedure%", "%sequence%",
            "%phase%", "%operation%",
        ]),
        ("activity", [
            "%students%", "%children%", "%people%performing%",
            "%activity%", "%exercise%", "%experiment%",
        ]),
    ]

    for img_type, patterns in finer_type_patterns:
        like_clauses = " OR ".join(
            f"(caption ILIKE '{p}' OR page_text_snippet ILIKE '{p}')"
            for p in patterns
        )
        conn.execute(
            sa.text(
                f"UPDATE textbook_images SET image_type = '{img_type}' "
                f"WHERE image_type IN ('unknown', 'diagram', 'landscape') "
                f"AND ({like_clauses})"
            )
        )

    # Populate educational_role based on image_type
    role_map = {
        "primary_concept": ("weather_station", "instrument", "diagram", "process"),
        "supporting_example": ("map", "chart", "satellite_image", "landscape", "historical_photo", "activity"),
        "sidebar_example": ("wildlife",),
        "decorative": (),  # handled separately below
    }
    for role, types in role_map.items():
        if not types:
            continue
        type_list = ", ".join(f"'{t}'" for t in types)
        conn.execute(
            sa.text(
                f"UPDATE textbook_images SET educational_role = '{role}' "
                f"WHERE image_type IN ({type_list}) AND educational_role = 'unknown'"
            )
        )

    # Rows with no caption or very short caption → decorative
    conn.execute(
        sa.text(
            "UPDATE textbook_images SET educational_role = 'decorative' "
            "WHERE educational_role = 'unknown' "
            "AND (caption IS NULL OR CHAR_LENGTH(TRIM(caption)) < 8)"
        )
    )

    # Remaining unknowns → sidebar_example (conservative)
    conn.execute(
        sa.text(
            "UPDATE textbook_images SET educational_role = 'sidebar_example' "
            "WHERE educational_role = 'unknown'"
        )
    )


def downgrade() -> None:
    op.drop_column("textbook_images", "educational_role")
    op.drop_column("textbook_images", "subsection_title")
    op.drop_column("textbook_images", "section_title")
