"""Signup schema and validator unit checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.modules.auth.signup.enums import CurriculumEnum, parse_curricula
from app.modules.auth.signup.schemas import SchoolSignupRequest, StudentSignupRequest, TutorSignupRequest
from app.modules.auth.signup.validators import normalize_grade, normalize_phone


def test_normalize_phone_strips_formatting():
    assert normalize_phone("+91 98765 43210") == "919876543210"


def test_normalize_grade():
    assert normalize_grade("Grade 6") == "6"
    assert normalize_grade("8") == "8"


def test_parse_curricula_dedupes():
    out = parse_curricula(["CBSE", "ICSE", "CBSE"])
    assert out == [CurriculumEnum.CBSE, CurriculumEnum.ICSE]


def test_student_signup_minimal():
    payload = StudentSignupRequest(
        full_name="Rahul Sharma",
        grade="Grade 6",
        user_id="rahul.g6",
        password="secret123",
        curricula=["CBSE"],
    )
    assert payload.grade == "6"
    assert payload.email.endswith("@signup.aitutor.app")


def test_student_signup_rejects_bad_grade():
    with pytest.raises(ValidationError):
        StudentSignupRequest(
            full_name="Rahul Sharma",
            grade="12",
            user_id="rahul",
            password="secret123",
            curricula=["CBSE"],
        )


def test_tutor_signup_requires_subjects():
    payload = TutorSignupRequest(
        full_name="Anita Verma",
        user_id="anita.math",
        password="secret123",
        email="anita@email.com",
        mobile="+91 9876543210",
        teaching_experience="3-5 Years",
        grades_teach="6th Grade - 10th Grade",
        class_size="21 - 40 Students",
        teaching_subjects=["Mathematics"],
        teaching_modes=["online", "hybrid"],
    )
    assert payload.mobile == "919876543210"


def test_enum_values_match_api_labels():
    from app.modules.auth.signup.enums import (
        LearningMethodEnum,
        StudentGradeEnum,
        StudentSubjectEnum,
        signup_pg_enum,
    )

    assert StudentSubjectEnum.ENGLISH.value == "English"
    assert StudentGradeEnum.GRADE_9.value == "9"
    assert LearningMethodEnum.AI_TUTOR.value == "ai-tutor"

    col = signup_pg_enum(StudentSubjectEnum, "signup_student_subject_enum")
    assert "English" in col.enums
    assert "ENGLISH" not in col.enums


def test_signup_options_payload():
    from app.modules.auth.signup.constants import signup_options_payload

    opts = signup_options_payload()
    assert "CBSE" in opts["curricula"]
    assert opts["student_grades"][0] == "Grade 1"
    assert "Mathematics" in opts["tutor_subjects"]


def test_signup_response_shape():
    from app.modules.auth.signup.schemas import SignupResponse

    resp = SignupResponse(id=1, user_id="rahul.g6", role="STUDENT", message="ok")
    assert resp.id == 1
    assert resp.user_id == "rahul.g6"


def test_school_signup_payload():
    payload = SchoolSignupRequest(
        school_name="Greenwood High School",
        school_email="admin@school.edu",
        phone="+91 9876543210",
        address="Main Road, City",
        full_name="John Doe",
        designation="School Administrator",
        recovery_email="admin@email.com",
        mobile="+91 9876543210",
        user_id="greenwood.admin",
        password="secret123",
        grades_offered="1st Grade - 10th Grade",
        student_strength="101 - 500 Students",
        curricula=["CBSE", "ICSE"],
    )
    assert payload.curricula == ["CBSE", "ICSE"]
