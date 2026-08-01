"""Signup profile rows with PostgreSQL enum columns."""

from __future__ import annotations

from sqlalchemy import BigInteger, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.modules.auth.signup.enums import (
    ClassSizeEnum,
    CurriculumEnum,
    LearningGoalEnum,
    LearningMethodEnum,
    StudentGradeEnum,
    StudentSubjectEnum,
    TeachingExperienceEnum,
    TeachingModeEnum,
    TeachingSubjectEnum,
    TutorGradeRangeEnum,
    signup_pg_enum,
)

_student_grade = signup_pg_enum(StudentGradeEnum, "signup_student_grade_enum")
_curriculum = signup_pg_enum(CurriculumEnum, "signup_curriculum_enum")
_student_subject = signup_pg_enum(StudentSubjectEnum, "signup_student_subject_enum")
_learning_goal = signup_pg_enum(LearningGoalEnum, "signup_learning_goal_enum")
_learning_method = signup_pg_enum(LearningMethodEnum, "signup_learning_method_enum")
_teaching_experience = signup_pg_enum(TeachingExperienceEnum, "signup_teaching_experience_enum")
_tutor_grade_range = signup_pg_enum(TutorGradeRangeEnum, "signup_tutor_grade_range_enum")
_class_size = signup_pg_enum(ClassSizeEnum, "signup_class_size_enum")
_teaching_subject = signup_pg_enum(TeachingSubjectEnum, "signup_teaching_subject_enum")
_teaching_mode = signup_pg_enum(TeachingModeEnum, "signup_teaching_mode_enum")


class UserSignupProfile(Base):
    """Role-specific signup fields stored as DB enums (1:1 with users)."""

    __tablename__ = "user_signup_profiles"

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )

    student_grade: Mapped[StudentGradeEnum | None] = mapped_column(_student_grade, nullable=True)
    curricula: Mapped[list[CurriculumEnum] | None] = mapped_column(ARRAY(_curriculum), nullable=True)
    favorite_subjects: Mapped[list[StudentSubjectEnum] | None] = mapped_column(
        ARRAY(_student_subject), nullable=True
    )
    learning_goals: Mapped[list[LearningGoalEnum] | None] = mapped_column(
        ARRAY(_learning_goal), nullable=True
    )
    preferred_learning_method: Mapped[LearningMethodEnum | None] = mapped_column(
        _learning_method, nullable=True
    )
    parent_email: Mapped[str | None] = mapped_column(String(320), nullable=True)

    teaching_experience: Mapped[TeachingExperienceEnum | None] = mapped_column(
        _teaching_experience, nullable=True
    )
    grades_teach: Mapped[TutorGradeRangeEnum | None] = mapped_column(_tutor_grade_range, nullable=True)
    class_size: Mapped[ClassSizeEnum | None] = mapped_column(_class_size, nullable=True)
    teaching_subjects: Mapped[list[TeachingSubjectEnum] | None] = mapped_column(
        ARRAY(_teaching_subject), nullable=True
    )
    teaching_modes: Mapped[list[TeachingModeEnum] | None] = mapped_column(
        ARRAY(_teaching_mode), nullable=True
    )
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)

    designation: Mapped[str | None] = mapped_column(String(100), nullable=True)
