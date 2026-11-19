"""PostgreSQL-backed signup enums — values match the public API."""

from enum import StrEnum

from sqlalchemy import Enum as SAEnum


def signup_pg_enum(enum_cls: type[StrEnum], name: str) -> SAEnum:
    """Map StrEnum members to PG enum labels via .value (e.g. English), not .name (ENGLISH)."""
    return SAEnum(
        enum_cls,
        name=name,
        create_type=False,
        values_callable=lambda cls: [m.value for m in cls],
    )


class CurriculumEnum(StrEnum):
    CBSE = "CBSE"
    ICSE = "ICSE"
    STATE_BOARD = "State Board"
    IB = "IB"
    IGCSE = "IGCSE"
    CAMBRIDGE = "Cambridge"


class StudentGradeEnum(StrEnum):
    GRADE_1 = "1"
    GRADE_2 = "2"
    GRADE_3 = "3"
    GRADE_4 = "4"
    GRADE_5 = "5"
    GRADE_6 = "6"
    GRADE_7 = "7"
    GRADE_8 = "8"
    GRADE_9 = "9"
    GRADE_10 = "10"


class StudentSubjectEnum(StrEnum):
    MATHEMATICS = "Mathematics"
    SCIENCE = "Science"
    SOCIAL = "Social"
    ENGLISH = "English"
    COMPUTER = "Computer"
    HINDI = "Hindi"
    SANSKRIT = "Sanskrit"


class LearningGoalEnum(StrEnum):
    IMPROVE_GRADES = "Improve Grades"
    EXAM_PREPARATION = "Exam Preparation"
    BUILD_CONCEPTS = "Build Concepts"
    HOMEWORK_HELP = "Homework Help"
    COMPETITIVE_EXAMS = "Competitive Exams"


class LearningMethodEnum(StrEnum):
    AI_TUTOR = "ai-tutor"
    AI_VOICE = "ai-voice"
    VIDEOS = "videos"


class TeachingSubjectEnum(StrEnum):
    MATHEMATICS = "Mathematics"
    SCIENCE = "Science"
    PHYSICS = "Physics"
    CHEMISTRY = "Chemistry"
    ENGLISH = "English"
    BIOLOGY = "Biology"


class TeachingExperienceEnum(StrEnum):
    LESS_THAN_1_YEAR = "Less than 1 Year"
    ONE_TO_3_YEARS = "1-3 Years"
    THREE_TO_5_YEARS = "3-5 Years"
    FIVE_PLUS_YEARS = "5+ Years"


class TutorGradeRangeEnum(StrEnum):
    GRADES_1_TO_5 = "1st Grade - 5th Grade"
    GRADES_6_TO_10 = "6th Grade - 10th Grade"
    GRADES_11_TO_12 = "11th Grade - 12th Grade"
    ALL_GRADES = "All Grades"


class ClassSizeEnum(StrEnum):
    SMALL = "1 - 20 Students"
    MEDIUM = "21 - 40 Students"
    LARGE = "41 - 60 Students"
    EXTRA_LARGE = "60+ Students"


class TeachingModeEnum(StrEnum):
    ONLINE = "online"
    IN_PERSON = "in-person"
    HYBRID = "hybrid"


class SchoolGradeRangeEnum(StrEnum):
    GRADES_1_TO_5 = "1st Grade - 5th Grade"
    GRADES_1_TO_10 = "1st Grade - 10th Grade"
    GRADES_6_TO_12 = "6th Grade - 12th Grade"


class StudentStrengthEnum(StrEnum):
    UP_TO_100 = "1 - 100 Students"
    UP_TO_500 = "101 - 500 Students"
    UP_TO_1000 = "501 - 1000 Students"
    OVER_1000 = "1000+ Students"


def enum_values(enum_cls: type[StrEnum]) -> list[str]:
    return [m.value for m in enum_cls]


def parse_enum(enum_cls: type[StrEnum], raw: str, label: str) -> StrEnum:
    try:
        return enum_cls(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {raw}") from exc


def parse_enum_list(enum_cls: type[StrEnum], values: list[str], label: str) -> list[StrEnum]:
    if not values:
        raise ValueError(f"Select at least one {label}.")
    out: list[StrEnum] = []
    for raw in values:
        item = parse_enum(enum_cls, raw, label)
        if item not in out:
            out.append(item)
    return out
