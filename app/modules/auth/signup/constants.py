"""Allowed values for public signup forms — derived from DB enum definitions."""

from app.modules.auth.signup.enums import (
    ClassSizeEnum,
    CurriculumEnum,
    LearningGoalEnum,
    LearningMethodEnum,
    SchoolGradeRangeEnum,
    StudentGradeEnum,
    StudentStrengthEnum,
    StudentSubjectEnum,
    TeachingExperienceEnum,
    TeachingModeEnum,
    TeachingSubjectEnum,
    TutorGradeRangeEnum,
    enum_values,
)

STUDENT_GRADE_LABELS = tuple(f"Grade {g.value}" for g in StudentGradeEnum)


def signup_options_payload() -> dict[str, list[str]]:
    """Served via GET /auth/signup/options — mirrors PostgreSQL enum values."""
    return {
        "curricula": sorted(enum_values(CurriculumEnum)),
        "student_grades": list(STUDENT_GRADE_LABELS),
        "student_subjects": sorted(enum_values(StudentSubjectEnum)),
        "learning_goals": sorted(enum_values(LearningGoalEnum)),
        "learning_methods": sorted(enum_values(LearningMethodEnum)),
        "tutor_subjects": sorted(enum_values(TeachingSubjectEnum)),
        "teaching_experience": sorted(enum_values(TeachingExperienceEnum)),
        "tutor_grade_ranges": sorted(enum_values(TutorGradeRangeEnum)),
        "class_sizes": sorted(enum_values(ClassSizeEnum)),
        "teaching_modes": sorted(enum_values(TeachingModeEnum)),
        "school_grade_ranges": sorted(enum_values(SchoolGradeRangeEnum)),
        "student_strength": sorted(enum_values(StudentStrengthEnum)),
    }
