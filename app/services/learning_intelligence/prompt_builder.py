"""Convert Tutor Guidance Object into AI Tutor prompt injection strings."""

from __future__ import annotations

from typing import Any

from app.modules.learning_intelligence.schemas import TutorGuidanceObject
from app.services.learning_intelligence.algorithms.concept_graph import humanize_concept_key


def guidance_to_prompt_sections(
    guidance: TutorGuidanceObject,
    profile: dict[str, Any],
    *,
    understanding_scores: dict[str, Any] | None = None,
    topic: str = "",
    long_term: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    """Returns (learner_guidance, adaptive_guidance, full_instructions)."""
    scores = understanding_scores or {}
    parts: list[str] = []

    if profile.get("weak_concepts"):
        weak = [humanize_concept_key(c) for c in profile["weak_concepts"][-5:]]
        parts.append(f"Student struggled before with: {', '.join(weak)}.")
    if profile.get("strong_concepts"):
        strong = [humanize_concept_key(c) for c in profile["strong_concepts"][-5:]]
        parts.append(f"Student is confident with: {', '.join(strong)}.")

    style = guidance.learning_style
    if style and style != "balanced":
        parts.append(f"Inferred learning style: {style.replace('_', ' ')}.")

    rev = (profile.get("revision_behaviour") or {})
    if rev.get("prefers_revision"):
        parts.append("Student benefits from revision sessions on previously covered material.")

    learner_guidance = ""
    if parts:
        learner_guidance = (
            "LEARNER PROFILE (use to personalize — do not mention explicitly):\n"
            + " ".join(parts)
        )

    adaptive_parts: list[str] = []
    if float(scores.get("confusion", 0)) >= 0.55 or scores.get("wants_expansion"):
        adaptive_parts.append(
            "ADAPTIVE MODE — SIMPLIFY: The student seems confused or asked for more help. "
            f"Use {guidance.preferred_explanation.replace('_', ' ')}, pace={guidance.teaching_pace}, "
            f"hint_level={guidance.hint_level}. Do not add new topics."
        )
    elif topic and any(
        w.lower() in topic.lower() or topic.lower() in w.lower()
        for w in (profile.get("weak_concepts") or [])
        if w
    ):
        adaptive_parts.append(
            "ADAPTIVE MODE — REINFORCE: The student struggled with this topic before. "
            "Use extra clarity, a fresh analogy, and connect to something familiar."
        )
    elif float(scores.get("understanding", 0)) >= 0.75 and scores.get("is_affirmation"):
        adaptive_parts.append(
            "ADAPTIVE MODE — ADVANCE: The student understood the last point. "
            "Briefly acknowledge, then offer the next small step or a gentle challenge."
        )
    elif scores.get("wants_quiz"):
        adaptive_parts.append(
            "ADAPTIVE MODE — ASSESS: The student wants to be tested. "
            "Focus on quiz questions from the chapter — mentor tone, not exam pressure."
        )

    if guidance.misconceptions:
        adaptive_parts.append(
            "MISCONCEPTION WATCH: Avoid reinforcing these errors — "
            + "; ".join(guidance.misconceptions[:3])
        )
    if guidance.topics_to_revise:
        adaptive_parts.append(
            "REVISION REMINDER: Briefly connect or revisit — "
            + ", ".join(guidance.topics_to_revise[:4])
        )
    if guidance.topics_to_avoid:
        adaptive_parts.append(
            "AVOID: Do not jump ahead to — " + ", ".join(guidance.topics_to_avoid[:3])
        )

    lt = long_term or {}
    weekly = lt.get("weekly_summary") or {}
    monthly = lt.get("monthly_summary") or {}
    if monthly.get("regression_trend", 0) > 0.05:
        adaptive_parts.append(
            "LONG-TERM TREND: Performance has regressed over the past month — slow down and reinforce fundamentals."
        )
    elif weekly.get("improvement_trend", 0) > 0.05:
        adaptive_parts.append(
            "LONG-TERM TREND: Steady improvement this week — you may increase challenge slightly."
        )

    pred = prediction or {}
    if pred.get("risk_level") == "high":
        adaptive_parts.append(
            "RISK SIGNAL: Student is at elevated risk of falling behind — prioritize mastery over coverage."
        )

    adaptive_guidance = "\n".join(adaptive_parts)

    instructions = (
        f"LIA TUTOR COACH (follow silently — never mention LIA to the student):\n"
        f"- Pace: {guidance.teaching_pace}; Difficulty: {guidance.difficulty}\n"
        f"- Explanation style: {guidance.preferred_explanation.replace('_', ' ')}\n"
        f"- Question frequency: {guidance.question_frequency}\n"
        f"- Encouragement: {guidance.encouragement_level}; Cognitive load: {guidance.cognitive_load}\n"
        f"- Next best action: {guidance.next_best_action.replace('_', ' ')}\n"
    )
    if pred.get("mastery_forecast_7d") is not None:
        instructions += f"- 7-day mastery forecast: {pred['mastery_forecast_7d']:.0%}\n"
    if guidance.animation_recommended:
        instructions += "- Prefer visual/diagram references when available.\n"
    if guidance.real_world_examples:
        instructions += "- Include a relatable real-world example when teaching.\n"

    return learner_guidance, adaptive_guidance, instructions
