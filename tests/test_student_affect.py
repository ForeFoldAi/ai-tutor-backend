"""Hybrid student affect classification tests."""

import pytest


@pytest.mark.parametrize(
    "text,expected_primary",
    [
        ("I don't get it, this is too hard", "confused"),
        ("this is boring can we do something else", "bored"),
        ("wow that's so cool", "excited"),
        ("thanks I'm done for today", "closing"),
        ("got it", "affirmation"),
        ("yesterday it rained near my house", "personal"),
        ("sure whatever", "bored"),
        ("what is weather", "curious"),
    ],
)
def test_regex_affect_primary(text, expected_primary):
    from app.services.student_affect import evaluate_student_affect

    aff = evaluate_student_affect(text)
    assert aff.primary == expected_primary
    assert aff.confidence >= 0.7


def test_understanding_scores_bridge():
    from app.services.student_affect import evaluate_student_affect

    aff = evaluate_student_affect("got it")
    scores = aff.to_understanding_scores()
    assert scores["is_affirmation"] is True
    assert scores["understanding"] >= 0.8


def test_merge_affect_trajectory():
    from app.services.student_affect import merge_affect_trajectory

    out = merge_affect_trajectory(["confused", "confused"], "affirmation")
    assert out == ["confused", "confused", "affirmation"]


def test_rapport_from_trajectory_progress():
    from app.services.student_affect import rapport_from_trajectory

    hint = rapport_from_trajectory(["confused", "frustrated", "affirmation"])
    assert "struggled earlier" in hint.lower()


def test_to_hint_affirmation_does_not_teach():
    from app.services.student_affect import StudentAffect

    hint = StudentAffect(primary="affirmation").to_hint().lower()
    assert "acknowledgment" in hint or "acknowledgement" in hint
    assert "new fact" in hint
    assert "next step" not in hint


def test_to_hint_confused():
    from app.services.student_affect import StudentAffect

    assert "simplify" in StudentAffect(primary="confused").to_hint().lower()


def test_golden_weather_question_not_confused():
    from app.services.student_affect import evaluate_student_affect

    aff = evaluate_student_affect("What is weather?")
    assert aff.primary != "confused"
