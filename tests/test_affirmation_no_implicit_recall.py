"""Affirmations must not be answered like implicit recall-meta questions."""

from app.services.voice_tutor import UnderstandingScores


def test_affirmation_hint_forbids_implicit_last_question_recap():
    scores = UnderstandingScores(is_affirmation=True, understanding=0.9)
    hint = scores.to_hint().lower()
    assert "your last question" in hint
    assert "unless they explicitly asked" in hint
