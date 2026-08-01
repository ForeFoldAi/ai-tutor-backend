"""Individual student enrollment uses signup profile, not SchoolClass."""

from app.modules.student_learning.enrollment import (
    _is_individual_student,
    _parse_board,
    _parse_class_level,
)


def test_individual_student_flag():
    class U:
        school_id = None
        created_by = None

    assert _is_individual_student(U()) is True


def test_tagged_student_is_not_individual():
    class SchoolTagged:
        school_id = 1
        created_by = None

    class TutorTagged:
        school_id = None
        created_by = 9

    assert _is_individual_student(SchoolTagged()) is False
    assert _is_individual_student(TutorTagged()) is False


def test_board_aliases_for_signup_curricula():
    assert _parse_board("State Board") == _parse_board("STATE_BOARD")
    assert _parse_board("CBSE").value == "CBSE"


def test_grade_to_class_level():
    assert _parse_class_level("9").value == "CLASS_9"
    assert _parse_class_level("Grade 9").value == "CLASS_9"
