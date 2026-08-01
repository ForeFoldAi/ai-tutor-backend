"""PostgreSQL signup enum types + user_signup_profiles table

Revision ID: 20260713_0022
Revises: 20260713_0021
Create Date: 2026-07-13 15:30:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260713_0022"
down_revision = "20260713_0021"
branch_labels = None
depends_on = None

SIGNUP_ENUMS: list[tuple[str, list[str]]] = [
    (
        "signup_curriculum_enum",
        ["CBSE", "ICSE", "State Board", "IB", "IGCSE", "Cambridge"],
    ),
    ("signup_student_grade_enum", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]),
    (
        "signup_student_subject_enum",
        ["Mathematics", "Science", "Social", "English", "Computer", "Hindi", "Sanskrit"],
    ),
    (
        "signup_learning_goal_enum",
        [
            "Improve Grades",
            "Exam Preparation",
            "Build Concepts",
            "Homework Help",
            "Competitive Exams",
        ],
    ),
    ("signup_learning_method_enum", ["ai-tutor", "ai-voice", "videos"]),
    (
        "signup_teaching_subject_enum",
        ["Mathematics", "Science", "Physics", "Chemistry", "English", "Biology"],
    ),
    (
        "signup_teaching_experience_enum",
        ["Less than 1 Year", "1-3 Years", "3-5 Years", "5+ Years"],
    ),
    (
        "signup_tutor_grade_range_enum",
        [
            "1st Grade - 5th Grade",
            "6th Grade - 10th Grade",
            "11th Grade - 12th Grade",
            "All Grades",
        ],
    ),
    (
        "signup_class_size_enum",
        ["1 - 20 Students", "21 - 40 Students", "41 - 60 Students", "60+ Students"],
    ),
    ("signup_teaching_mode_enum", ["online", "in-person", "hybrid"]),
    (
        "signup_school_grade_range_enum",
        ["1st Grade - 5th Grade", "1st Grade - 10th Grade", "6th Grade - 12th Grade"],
    ),
    (
        "signup_student_strength_enum",
        [
            "1 - 100 Students",
            "101 - 500 Students",
            "501 - 1000 Students",
            "1000+ Students",
        ],
    ),
]


def _pg_enum(name: str, values: list[str], *, create_type: bool) -> postgresql.ENUM:
    return postgresql.ENUM(*values, name=name, create_type=create_type)


def upgrade() -> None:
    bind = op.get_bind()
    for name, values in SIGNUP_ENUMS:
        _pg_enum(name, values, create_type=True).create(bind, checkfirst=True)

    curriculum = _pg_enum("signup_curriculum_enum", SIGNUP_ENUMS[0][1], create_type=False)
    student_grade = _pg_enum("signup_student_grade_enum", SIGNUP_ENUMS[1][1], create_type=False)
    student_subject = _pg_enum("signup_student_subject_enum", SIGNUP_ENUMS[2][1], create_type=False)
    learning_goal = _pg_enum("signup_learning_goal_enum", SIGNUP_ENUMS[3][1], create_type=False)
    learning_method = _pg_enum("signup_learning_method_enum", SIGNUP_ENUMS[4][1], create_type=False)
    teaching_subject = _pg_enum("signup_teaching_subject_enum", SIGNUP_ENUMS[5][1], create_type=False)
    teaching_experience = _pg_enum("signup_teaching_experience_enum", SIGNUP_ENUMS[6][1], create_type=False)
    tutor_grade_range = _pg_enum("signup_tutor_grade_range_enum", SIGNUP_ENUMS[7][1], create_type=False)
    class_size = _pg_enum("signup_class_size_enum", SIGNUP_ENUMS[8][1], create_type=False)
    teaching_mode = _pg_enum("signup_teaching_mode_enum", SIGNUP_ENUMS[9][1], create_type=False)
    school_grade_range = _pg_enum("signup_school_grade_range_enum", SIGNUP_ENUMS[10][1], create_type=False)
    student_strength = _pg_enum("signup_student_strength_enum", SIGNUP_ENUMS[11][1], create_type=False)

    op.create_table(
        "user_signup_profiles",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("student_grade", student_grade, nullable=True),
        sa.Column("curricula", postgresql.ARRAY(curriculum), nullable=True),
        sa.Column("favorite_subjects", postgresql.ARRAY(student_subject), nullable=True),
        sa.Column("learning_goals", postgresql.ARRAY(learning_goal), nullable=True),
        sa.Column("preferred_learning_method", learning_method, nullable=True),
        sa.Column("parent_email", sa.String(length=320), nullable=True),
        sa.Column("teaching_experience", teaching_experience, nullable=True),
        sa.Column("grades_teach", tutor_grade_range, nullable=True),
        sa.Column("class_size", class_size, nullable=True),
        sa.Column("teaching_subjects", postgresql.ARRAY(teaching_subject), nullable=True),
        sa.Column("teaching_modes", postgresql.ARRAY(teaching_mode), nullable=True),
        sa.Column("bio", sa.Text(), nullable=True),
        sa.Column("designation", sa.String(length=100), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )

    op.add_column(
        "organizations",
        sa.Column("grades_offered", school_grade_range, nullable=True),
    )
    op.add_column(
        "organizations",
        sa.Column("student_strength", student_strength, nullable=True),
    )
    op.add_column(
        "organizations",
        sa.Column("curricula", postgresql.ARRAY(curriculum), nullable=True),
    )

    # ponytail: preferences JSON was short-lived — drop if present from prior migration
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS preferences")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS profile_data")


def downgrade() -> None:
    bind = op.get_bind()

    op.drop_column("organizations", "curricula")
    op.drop_column("organizations", "student_strength")
    op.drop_column("organizations", "grades_offered")
    op.add_column("organizations", sa.Column("preferences", postgresql.JSON(astext_type=sa.Text()), nullable=True))

    op.drop_table("user_signup_profiles")
    op.add_column("users", sa.Column("profile_data", postgresql.JSON(astext_type=sa.Text()), nullable=True))

    for name, _ in reversed(SIGNUP_ENUMS):
        postgresql.ENUM(name=name).drop(bind, checkfirst=True)
