"""Fold organizations into schools; school signup uses SCHOOL_ADMIN only

Revision ID: 20260713_0023
Revises: 20260713_0022
Create Date: 2026-07-13 21:10:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260713_0023"
down_revision = "20260713_0022"
branch_labels = None
depends_on = None

_curriculum_enum = postgresql.ENUM(
    "CBSE", "ICSE", "State Board", "IB", "IGCSE", "Cambridge",
    name="signup_curriculum_enum",
    create_type=False,
)
_grade_range_enum = postgresql.ENUM(
    "1st Grade - 5th Grade",
    "6th Grade - 10th Grade",
    "1st Grade - 10th Grade",
    name="signup_school_grade_range_enum",
    create_type=False,
)
_student_strength_enum = postgresql.ENUM(
    "1 - 100 Students",
    "101 - 500 Students",
    "501 - 1000 Students",
    "1000+ Students",
    name="signup_student_strength_enum",
    create_type=False,
)


def upgrade() -> None:
    op.add_column("schools", sa.Column("email", sa.String(length=320), nullable=True))
    op.add_column("schools", sa.Column("phone", sa.String(length=50), nullable=True))
    op.add_column("schools", sa.Column("website", sa.String(length=500), nullable=True))
    op.add_column("schools", sa.Column("address", sa.String(length=500), nullable=True))
    op.add_column("schools", sa.Column("grades_offered", _grade_range_enum, nullable=True))
    op.add_column("schools", sa.Column("student_strength", _student_strength_enum, nullable=True))
    op.add_column("schools", sa.Column("curricula", postgresql.ARRAY(_curriculum_enum), nullable=True))
    op.add_column(
        "schools",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )

    op.execute(
        """
        UPDATE schools s SET
            email = o.email,
            phone = o.phone,
            website = o.website,
            address = o.address,
            grades_offered = o.grades_offered,
            student_strength = o.student_strength,
            curricula = o.curricula,
            is_active = o.is_active
        FROM organizations o
        WHERE s.organization_id = o.id
        """
    )

    op.execute("UPDATE users SET role = 'SCHOOL_ADMIN' WHERE role = 'ORG_ADMIN'")

    op.drop_constraint("users_organization_id_fkey", "users", type_="foreignkey")
    op.drop_column("users", "organization_id")

    op.drop_constraint("schools_organization_id_fkey", "schools", type_="foreignkey")
    op.drop_column("schools", "organization_id")

    op.drop_table("organizations")


def downgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("phone", sa.String(length=50), nullable=True),
        sa.Column("website", sa.String(length=500), nullable=True),
        sa.Column("address", sa.String(length=500), nullable=True),
        sa.Column("grades_offered", _grade_range_enum, nullable=True),
        sa.Column("student_strength", _student_strength_enum, nullable=True),
        sa.Column("curricula", postgresql.ARRAY(_curriculum_enum), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.add_column(
        "schools",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("organization_id", postgresql.UUID(as_uuid=True), nullable=True),
    )

    op.drop_column("schools", "is_active")
    op.drop_column("schools", "curricula")
    op.drop_column("schools", "student_strength")
    op.drop_column("schools", "grades_offered")
    op.drop_column("schools", "address")
    op.drop_column("schools", "website")
    op.drop_column("schools", "phone")
    op.drop_column("schools", "email")
