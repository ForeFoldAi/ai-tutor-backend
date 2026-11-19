"""Learning Strategy Agent — infers teaching strategies from behaviour."""

from __future__ import annotations

from typing import Any

from app.services.learning_intelligence.algorithms.mastery import (
    difficulty_recommendation,
    teaching_pace_from_signals,
)


def build_teaching_strategy(
    profile: dict[str, Any],
    *,
    understanding_scores: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = understanding_scores or {}
    style = profile.get("learning_style") or {}
    dominant = style.get("dominant", "balanced")
    style_scores = style.get("scores") or {}

    preferred_explanation = "stepwise_with_analogy"
    if dominant in ("visual", "animation"):
        preferred_explanation = "visual_stepwise"
    elif dominant == "reading":
        preferred_explanation = "text_structured"
    elif dominant == "practice":
        preferred_explanation = "example_then_practice"
    elif dominant == "conversation":
        preferred_explanation = "socratic_dialogue"
    elif dominant == "story":
        preferred_explanation = "narrative_analogy"

    mastery_avg = 0.5
    strong = profile.get("strong_concepts") or []
    weak = profile.get("weak_concepts") or []
    if strong or weak:
        mastery_avg = len(strong) / max(1, len(strong) + len(weak))

    pace = teaching_pace_from_signals(
        confusion=float(scores.get("confusion", 0)),
        attention=float(profile.get("attention_score", 0.5)),
        mastery_avg=mastery_avg,
    )

    animation_recommended = style_scores.get("animation", 0) >= 0.2 or style_scores.get("visual", 0) >= 0.25
    question_frequency = "high" if scores.get("wants_quiz") or dominant == "practice" else "moderate"
    if float(scores.get("confusion", 0)) >= 0.55:
        question_frequency = "low"

    return {
        "learning_style": dominant,
        "preferred_explanation": preferred_explanation,
        "teaching_pace": pace,
        "difficulty": difficulty_recommendation(
            mastery_avg, float(profile.get("confidence_score", 0.5))
        ),
        "animation_recommended": animation_recommended,
        "real_world_examples": style_scores.get("example", 0) >= 0.15 or dominant in ("story", "example"),
        "question_frequency": question_frequency,
        "motivation_style": _motivation_style(profile, scores),
        "encouragement_level": _encouragement_level(scores),
        "hint_level": _hint_level(scores, profile),
        "cognitive_load": _cognitive_load(scores, profile),
    }


def _motivation_style(profile: dict, scores: dict) -> str:
    if float(profile.get("confidence_score", 0.5)) < 0.4:
        return "effort_reinforcement_with_choice"
    if scores.get("wants_quiz"):
        return "challenge_with_support"
    return "effort_with_choice"


def _encouragement_level(scores: dict) -> str:
    if float(scores.get("confusion", 0)) >= 0.55:
        return "high"
    if scores.get("is_affirmation"):
        return "low"
    return "medium"


def _hint_level(scores: dict, profile: dict) -> str:
    if float(scores.get("confusion", 0)) >= 0.55:
        return "high"
    if float(profile.get("confidence_score", 0.5)) >= 0.75:
        return "low"
    return "medium"


def _cognitive_load(scores: dict, profile: dict) -> str:
    if float(scores.get("confusion", 0)) >= 0.55 or float(profile.get("attention_score", 0.5)) < 0.4:
        return "low"
    if float(profile.get("engagement_score", 0.5)) >= 0.7:
        return "moderate_high"
    return "moderate"
