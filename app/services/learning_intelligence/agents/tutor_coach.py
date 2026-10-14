"""Tutor Coach Agent — produces Tutor Guidance Object (never student-facing content)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.learning_intelligence.models import LiaKnowledgeGap, LiaMisconception
from app.modules.learning_intelligence.schemas import TutorGuidanceObject, TutorGuidanceRequest
from app.services.learning_intelligence.agents.learning_strategy import build_teaching_strategy
from app.services.learning_intelligence.agents.memory import best_teaching_methods, get_long_term_context
from app.services.learning_intelligence.agents.prediction import build_prediction_context
from app.services.learning_intelligence.agents.student_modeling import get_or_create_profile, profile_to_dict
from app.services.learning_intelligence.algorithms.concept_graph import humanize_concept_key
from app.services.learning_intelligence.algorithms.mastery import concept_key_from_topic
from app.services.learning_intelligence.prompt_builder import guidance_to_prompt_sections


def build_tutor_guidance(
    db: Session,
    req: TutorGuidanceRequest,
) -> TutorGuidanceObject:
    profile = get_or_create_profile(db, req.student_user_id)
    profile_dict = profile_to_dict(profile)
    long_term = get_long_term_context(db, req.student_user_id)

    topic = req.topic or req.query
    concept = concept_key_from_topic(topic, subject=req.subject_name, chapter=req.chapter)

    prediction = build_prediction_context(db, req.student_user_id, concept)
    strategy = build_teaching_strategy(profile_dict, understanding_scores=req.understanding_scores)

    # Prefer historically successful explanation methods for this concept
    best_methods = best_teaching_methods(db, req.student_user_id, concept)
    if best_methods:
        strategy["preferred_explanation"] = best_methods[0]

    misconceptions = _misconception_labels(db, req.student_user_id, concept)
    gaps, revise, avoid = _gap_guidance(db, req.student_user_id, concept, profile_dict)

    if prediction.get("risk_level") == "high":
        strategy["teaching_pace"] = "slow"
        strategy["hint_level"] = "high"
    if float(profile_dict.get("regression_trend", 0)) > 0.05:
        revise = list(dict.fromkeys(revise + (profile_dict.get("weak_concepts") or [])[:3]))[:8]

    monthly = long_term.get("monthly_summary") or {}
    if monthly.get("regression_trend", 0) > 0.05:
        avoid = list(dict.fromkeys(avoid + (profile_dict.get("weak_concepts") or [])[:2]))[:4]

    guidance = TutorGuidanceObject(
        learning_style=strategy["learning_style"],
        preferred_examples=_preferred_examples(strategy),
        preferred_explanation=strategy["preferred_explanation"],
        teaching_pace=strategy["teaching_pace"],
        difficulty=strategy["difficulty"],
        confidence_level=float(profile_dict.get("confidence_score", 0.5)),
        motivation_style=strategy["motivation_style"],
        question_frequency=strategy["question_frequency"],
        misconceptions=misconceptions,
        knowledge_gaps=gaps,
        topics_to_revise=[humanize_concept_key(r) for r in revise],
        topics_to_connect=_connect_topics(concept, profile_dict),
        topics_to_avoid=[humanize_concept_key(a) for a in avoid],
        animation_recommended=bool(strategy["animation_recommended"]),
        real_world_examples=bool(strategy["real_world_examples"]),
        encouragement_level=strategy["encouragement_level"],
        hint_level=strategy["hint_level"],
        cognitive_load=strategy["cognitive_load"],
        next_best_action=_next_best_action(req, strategy, gaps, prediction),
    )

    learner_g, adaptive_g, instructions = guidance_to_prompt_sections(
        guidance,
        profile_dict,
        understanding_scores=req.understanding_scores,
        topic=topic,
        long_term=long_term,
        prediction=prediction,
    )
    guidance.learner_guidance = learner_g
    guidance.adaptive_guidance = adaptive_g
    guidance.prompt_instructions = instructions
    return guidance


def _misconception_labels(db: Session, student_id: int, concept: str) -> list[str]:
    rows = db.scalars(
        select(LiaMisconception)
        .where(LiaMisconception.student_user_id == student_id)
        .order_by(LiaMisconception.occurrence_count.desc())
        .limit(5)
    ).all()
    out: list[str] = []
    for r in rows:
        if r.concept_key == concept or r.occurrence_count >= 2:
            out.append(r.description[:200] or r.misconception_key)
    return out[:5]


def _gap_guidance(
    db: Session, student_id: int, concept: str, profile: dict
) -> tuple[list[str], list[str], list[str]]:
    rows = db.scalars(
        select(LiaKnowledgeGap).where(LiaKnowledgeGap.student_user_id == student_id).limit(15)
    ).all()
    gaps: list[str] = []
    revise: list[str] = list(profile.get("weak_concepts") or [])[:5]
    avoid: list[str] = []
    for r in rows:
        gaps.append(humanize_concept_key(r.concept_key))
        revise.extend(r.missing_prerequisites or [])
        if r.gap_type == "surface_memorization":
            avoid.append(r.concept_key)
    if concept in revise or any(concept in (r.missing_prerequisites or []) for r in rows):
        revise.insert(0, concept)
    return gaps[:5], list(dict.fromkeys(revise))[:8], avoid[:3]


def _connect_topics(concept: str, profile: dict) -> list[str]:
    strong = profile.get("strong_concepts") or []
    return [humanize_concept_key(s) for s in strong[:3] if s != concept]


def _preferred_examples(strategy: dict) -> list[str]:
    out = ["worked_example"]
    if strategy.get("real_world_examples"):
        out.append("real_world")
    if strategy.get("animation_recommended"):
        out.append("visual_diagram")
    return out


def _next_best_action(req: TutorGuidanceRequest, strategy: dict, gaps: list[str], prediction: dict) -> str:
    scores = req.understanding_scores or {}
    if prediction.get("risk_level") == "high":
        return "revision_session_recommended"
    if float(scores.get("confusion", 0)) >= 0.55:
        return "simplify_and_reteach"
    if gaps:
        return "revise_prerequisite_then_teach"
    if scores.get("wants_quiz"):
        return "micro_quiz_check"
    if scores.get("is_affirmation"):
        return "advance_to_next_step"
    if strategy.get("teaching_pace") == "slow":
        return "smaller_steps_same_topic"
    return prediction.get("recommended_action", "teach_current_topic")
