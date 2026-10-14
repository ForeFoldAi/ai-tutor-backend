from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Identity, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.modules.auth.signup.enums import (
    CurriculumEnum,
    SchoolGradeRangeEnum,
    StudentStrengthEnum,
    signup_pg_enum,
)

_curriculum_enum = signup_pg_enum(CurriculumEnum, "signup_curriculum_enum")
_school_grade_range_enum = signup_pg_enum(SchoolGradeRangeEnum, "signup_school_grade_range_enum")
_student_strength_enum = signup_pg_enum(StudentStrengthEnum, "signup_student_strength_enum")


class School(Base):
    __tablename__ = "schools"

    id: Mapped[int] = mapped_column(Integer, Identity(), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    board: Mapped[str | None] = mapped_column(String(100), nullable=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    website: Mapped[str | None] = mapped_column(String(500), nullable=True)
    address: Mapped[str | None] = mapped_column(String(500), nullable=True)
    grades_offered: Mapped[SchoolGradeRangeEnum | None] = mapped_column(
        _school_grade_range_enum,
        nullable=True,
    )
    student_strength: Mapped[StudentStrengthEnum | None] = mapped_column(
        _student_strength_enum,
        nullable=True,
    )
    curricula: Mapped[list[CurriculumEnum] | None] = mapped_column(
        ARRAY(_curriculum_enum),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
