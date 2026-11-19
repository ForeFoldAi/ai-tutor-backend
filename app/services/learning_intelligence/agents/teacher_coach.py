"""Teacher Coach Agent — concise insights for teachers (not charts-only)."""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.modules.learning_intelligence.constants import RISK_NOT_STARTED
from app.modules.learning_intelligence.schemas import TeacherSummaryOut
from app.services.learning_intelligence.agents.memory import get_long_term_context
from app.services.learning_intelligence.agents.observation import recent_events
from app.services.learning_intelligence.agents.student_modeling import get_or_create_profile, profile_to_dict
from app.services.learning_intelligence.algorithms.concept_graph import humanize_concept_key


def build_teacher_summary(db: Session, student_user_id: int) -> TeacherSummaryOut:
    profile = get_or_create_profile(db, student_user_id)
    p = profile_to_dict(profile)
    period_end = date.today()
    period_start = period_end - timedelta(days=7)

    # No learning events yet — don't invent medium risk / "engagement dropped" from empty twin defaults.
    if not recent_events(db, student_user_id, limit=1):
        return TeacherSummaryOut(
            student_user_id=student_user_id,
            observations=["Insufficient data for deep insights yet — continue monitoring interactions."],
            recommendations=[],
            risk_level=RISK_NOT_STARTED,
            strengths=[],
            weaknesses=[],
            period_start=period_start,
            period_end=period_end,
        )

    long_term = get_long_term_context(db, student_user_id)
    observations: list[str] = []
    recommendations: list[str] = []
    strengths: list[str] = []
    weaknesses: list[str] = []

    style = (p.get("learning_style") or {}).get("dominant", "balanced")
    if style in ("visual", "animation"):
        observations.append("The student learns significantly better after visual explanations.")
        recommendations.append("Use diagrams, animations, or labeled figures before text-heavy explanations.")
    elif style == "practice":
        observations.append("Interactive exercises and practice questions improve retention for this student.")
        recommendations.append("Assign short practice sets after each concept introduction.")

    if p.get("weak_concepts"):
        weaknesses = [humanize_concept_key(c) for c in p["weak_concepts"][:5]]
        observations.append(
            f"The student shows recurring difficulty with: {', '.join(weaknesses[:3])}."
        )
        recommendations.append("Schedule a short revision session on weak concepts before advancing.")

    if p.get("strong_concepts"):
        strengths = [humanize_concept_key(c) for c in p["strong_concepts"][:5]]
        observations.append(f"Strong understanding observed in: {', '.join(strengths[:3])}.")

    if p.get("confidence_score", 0.5) < 0.4:
        observations.append("Confidence appears low during recent learning interactions.")
        recommendations.append("Use encouragement and smaller wins before increasing difficulty.")
    elif p.get("confidence_score", 0.5) >= 0.7:
        observations.append("Confidence has increased during recent tutoring sessions.")

    monthly = long_term.get("monthly_summary") or {}
    if monthly.get("regression_trend", 0) > 0.05:
        observations.append("Performance regression detected over the past month.")
        recommendations.append("Revisit fundamentals and check for prerequisite gaps.")
    elif monthly.get("improvement_trend", 0) > 0.05:
        observations.append("Steady improvement trend over the past month.")

    rev = long_term.get("revision_behaviour") or {}
    if rev.get("prefers_revision"):
        observations.append("The student revisits completed chapters — revision helps retention.")
        recommendations.append("Build in brief recap segments at the start of new topics.")

    if p.get("regression_trend", 0) > 0.1:
        observations.append("Recent performance regression detected in the learning window.")
        recommendations.append("Revisit fundamentals and check for prerequisite gaps.")

    if p.get("engagement_score", 0.5) < 0.35:
        observations.append("Engagement has dropped — shorter, more interactive sessions may help.")
        recommendations.append("Break sessions into 15–20 minute focused blocks with variety.")

    if not observations:
        observations.append("Insufficient data for deep insights yet — continue monitoring interactions.")

    return TeacherSummaryOut(
        student_user_id=student_user_id,
        observations=observations,
        recommendations=recommendations,
        risk_level=p.get("risk_level") or RISK_NOT_STARTED,
        strengths=strengths,
        weaknesses=weaknesses,
        period_start=period_start,
        period_end=period_end,
    )
