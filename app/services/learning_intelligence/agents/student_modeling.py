"""Student Modeling Agent — builds and updates the Student Digital Twin."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.learning_intelligence.constants import (
    MASTERY_STRONG,
    MASTERY_WEAK,
    METRIC_ATTENTION,
    METRIC_CONFIDENCE,
    METRIC_ENGAGEMENT,
    METRIC_PERFORMANCE,
    RISK_NOT_STARTED,
)
from app.modules.learning_intelligence.models import LiaConceptMastery, LiaMetricHistory, LiaStudentProfile
from app.services.learning_intelligence.algorithms.mastery import (
    attention_score_from_signals,
    compute_risk,
    dominant_learning_style,
    engagement_score_from_events,
    infer_learning_style_signals,
    update_confidence_ema,
)
from app.services.learning_intelligence.algorithms.trends import compute_learning_trends
from app.services.learning_intelligence.agents.memory import detect_revision_behaviour, get_long_term_context
from app.services.learning_intelligence.agents.observation import recent_events


def get_or_create_profile(db: Session, student_user_id: int) -> LiaStudentProfile:
    profile = db.get(LiaStudentProfile, student_user_id)
    if profile is None:
        profile = LiaStudentProfile(student_user_id=student_user_id, updated_at=datetime.now(UTC))
        db.add(profile)
        db.flush()
    return profile


def update_twin_from_events(db: Session, student_user_id: int) -> LiaStudentProfile:
    profile = get_or_create_profile(db, student_user_id)
    events = recent_events(db, student_user_id, limit=200)
    style_scores = infer_learning_style_signals(events)
    profile.learning_style = {
        "scores": style_scores,
        "dominant": dominant_learning_style(style_scores),
    }

    conf_signal = 0.5
    for ev in events[:20]:
        p = ev.get("payload") or {}
        if "confidence" in p:
            conf_signal = update_confidence_ema(conf_signal, float(p["confidence"]))
        elif "confusion" in p:
            conf_signal = update_confidence_ema(conf_signal, 1.0 - float(p["confusion"]))

    profile.confidence_score = conf_signal
    profile.attention_score = attention_score_from_signals(
        idle_ratio=float((events[0].get("payload") or {}).get("idle_ratio", 0) if events else 0),
        session_minutes=float((events[0].get("payload") or {}).get("session_minutes", 0) if events else 0),
        topic_switches=sum(1 for e in events if e.get("event_type") == "topic_switch"),
    )
    profile.engagement_score = engagement_score_from_events(len(events), days=7)

    masteries = db.scalars(
        select(LiaConceptMastery).where(LiaConceptMastery.student_user_id == student_user_id)
    ).all()
    profile.strong_concepts = [m.concept_key for m in masteries if m.mastery_score >= MASTERY_STRONG][-20:]
    profile.weak_concepts = [m.concept_key for m in masteries if m.mastery_score < MASTERY_WEAK][-20:]

    subjects: dict[str, list[float]] = {}
    for m in masteries:
        subj = m.concept_key.split("|")[0] if "|" in m.concept_key else "general"
        subjects.setdefault(subj, []).append(m.mastery_score)
    profile.strong_subjects = [s for s, scores in subjects.items() if sum(scores) / len(scores) >= MASTERY_STRONG]
    profile.weak_subjects = [s for s, scores in subjects.items() if sum(scores) / len(scores) < MASTERY_WEAK]

    mastery_avg = sum(m.mastery_score for m in masteries) / len(masteries) if masteries else 0.5
    trends = compute_learning_trends(db, student_user_id)
    profile.improvement_trend = trends["improvement_trend"]
    profile.regression_trend = trends["regression_trend"]

    profile.revision_behaviour = detect_revision_behaviour(db, student_user_id)

    # 0 events + 0 mastery → not started (compute_risk treats 0 engagement as medium).
    if not events and not masteries:
        profile.risk_level = RISK_NOT_STARTED
    else:
        risk_level, _risk_score = compute_risk(
            mastery_avg=mastery_avg,
            confidence=profile.confidence_score,
            engagement=profile.engagement_score,
            regression_trend=profile.regression_trend,
        )
        profile.risk_level = risk_level

    ltc = get_long_term_context(db, student_user_id)
    memory = dict(profile.memory_profile or {})
    memory.update(
        {
            "weekly": ltc.get("weekly_summary"),
            "monthly": ltc.get("monthly_summary"),
        }
    )
    profile.memory_profile = memory

    profile.twin_version += 1
    profile.updated_at = datetime.now(UTC)

    _record_metric(db, student_user_id, METRIC_CONFIDENCE, profile.confidence_score)
    _record_metric(db, student_user_id, METRIC_ATTENTION, profile.attention_score)
    _record_metric(db, student_user_id, METRIC_ENGAGEMENT, profile.engagement_score)
    _record_metric(db, student_user_id, METRIC_PERFORMANCE, mastery_avg)

    return profile


def _record_metric(db: Session, student_user_id: int, metric_type: str, value: float) -> None:
    db.add(
        LiaMetricHistory(
            student_user_id=student_user_id,
            metric_type=metric_type,
            value=value,
            recorded_at=datetime.now(UTC),
        )
    )


def profile_to_dict(profile: LiaStudentProfile) -> dict[str, Any]:
    return {
        "student_user_id": profile.student_user_id,
        "learning_style": profile.learning_style,
        "teaching_preferences": profile.teaching_preferences,
        "strong_concepts": profile.strong_concepts,
        "weak_concepts": profile.weak_concepts,
        "strong_subjects": profile.strong_subjects,
        "weak_subjects": profile.weak_subjects,
        "confidence_score": profile.confidence_score,
        "attention_score": profile.attention_score,
        "engagement_score": profile.engagement_score,
        "motivation_score": profile.motivation_score,
        "persistence_score": profile.persistence_score,
        "improvement_trend": profile.improvement_trend,
        "regression_trend": profile.regression_trend,
        "risk_level": profile.risk_level,
        "revision_behaviour": profile.revision_behaviour,
        "memory_profile": profile.memory_profile,
        "twin_version": profile.twin_version,
        "updated_at": profile.updated_at,
    }
