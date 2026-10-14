"""Tests for Learning Intelligence Agent — phases 3 & 5."""

from __future__ import annotations

from app.modules.learning_intelligence.schemas import LearningEventIn, TutorGuidanceObject
from app.services.learning_intelligence.algorithms.concept_graph import (
    humanize_concept_key,
    transitive_prerequisites,
)
from app.services.learning_intelligence.algorithms.mastery import (
    concept_key_from_topic,
    dominant_learning_style,
    infer_learning_style_signals,
    memorized_likelihood,
    update_mastery_ema,
)
from app.services.learning_intelligence.algorithms.trends import _slope
from app.services.learning_intelligence.prompt_builder import guidance_to_prompt_sections


def test_mastery_ema_bounds():
    assert update_mastery_ema(0.5, "correct") > 0.5
    assert update_mastery_ema(0.5, "incorrect") < 0.5
    assert 0.0 <= update_mastery_ema(0.99, "correct") <= 1.0


def test_concept_key_stable():
    k1 = concept_key_from_topic("linear equations", subject="Math", chapter="Algebra")
    k2 = concept_key_from_topic("linear equations", subject="Math", chapter="Algebra")
    assert k1 == k2
    assert "math" in k1


def test_learning_style_inference():
    events = [
        {"event_type": "chat_user_question", "payload": {"animation_used": True}},
        {"event_type": "voice_user_utterance", "payload": {}},
    ]
    scores = infer_learning_style_signals(events)
    assert scores[dominant_learning_style(scores)] > 0


def test_guidance_prompt_sections():
    guidance = TutorGuidanceObject(
        learning_style="visual",
        teaching_pace="slow",
        preferred_explanation="visual_stepwise",
        misconceptions=["confuses area and perimeter"],
        topics_to_revise=["basic_shapes"],
    )
    profile = {"weak_concepts": ["geometry_basics"], "strong_concepts": ["addition"]}
    learner_g, adaptive_g, instructions = guidance_to_prompt_sections(
        guidance,
        profile,
        understanding_scores={"confusion": 0.6},
        topic="geometry",
        long_term={"monthly_summary": {"regression_trend": 0.1}},
        prediction={"risk_level": "high", "mastery_forecast_7d": 0.55},
    )
    assert "LEARNER PROFILE" in learner_g or learner_g == ""
    assert "SIMPLIFY" in adaptive_g or "MISCONCEPTION" in adaptive_g
    assert "LIA TUTOR COACH" in instructions
    assert "RISK SIGNAL" in adaptive_g or "LONG-TERM" in adaptive_g


def test_learning_event_schema():
    ev = LearningEventIn(
        event_type="chat_user_question",
        student_user_id=1,
        payload={"query": "what is photosynthesis"},
    )
    assert ev.student_user_id == 1


def test_transitive_prerequisites():
    edges = [
        ("arith", "algebra", 1.0),
        ("algebra", "linear_eq", 1.0),
    ]
    chain = transitive_prerequisites("linear_eq", edges)
    assert "arith" in chain
    assert "algebra" in chain


def test_memorization_likelihood():
    assert memorized_likelihood(mastery_score=0.8, understanding_level=1) >= 0.65


def test_trend_slope_improving():
    assert _slope([0.4, 0.5, 0.6, 0.7]) > 0


def test_humanize_concept_key():
    assert humanize_concept_key("math|algebra|linear_equations") == "linear equations"
