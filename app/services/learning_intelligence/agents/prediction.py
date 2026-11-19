"""Prediction Agent — risk + mastery forecast for tutor coaching."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.learning_intelligence.models import LiaConceptMastery, LiaLearningPrediction, LiaRiskAssessment
from app.modules.learning_intelligence.constants import RISK_HIGH
from app.services.learning_intelligence.agents.student_modeling import get_or_create_profile
from app.services.learning_intelligence.algorithms.mastery import compute_risk
from app.services.learning_intelligence.algorithms.trends import compute_learning_trends


def build_prediction_context(db: Session, student_user_id: int, concept_key: str | None = None) -> dict:
    profile = get_or_create_profile(db, student_user_id)
    masteries = db.scalars(
        select(LiaConceptMastery).where(LiaConceptMastery.student_user_id == student_user_id)
    ).all()
    mastery_avg = sum(m.mastery_score for m in masteries) / len(masteries) if masteries else 0.5

    trends = compute_learning_trends(db, student_user_id)
    risk_level, risk_score = compute_risk(
        mastery_avg=mastery_avg,
        confidence=profile.confidence_score,
        engagement=profile.engagement_score,
        regression_trend=trends["regression_trend"],
    )

    concept_mastery = None
    if concept_key:
        row = db.scalar(
            select(LiaConceptMastery).where(
                LiaConceptMastery.student_user_id == student_user_id,
                LiaConceptMastery.concept_key == concept_key,
            )
        )
        concept_mastery = row.mastery_score if row else None

    forecast_7d = min(1.0, max(0.0, mastery_avg + trends["improvement_trend"] * 0.15 - trends["regression_trend"] * 0.1))
    recommended = "revision_session" if risk_level == RISK_HIGH else "continue_with_support" if risk_level == "medium" else "continue_current_pace"

    prediction = {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "mastery_avg": round(mastery_avg, 3),
        "concept_mastery": concept_mastery,
        "mastery_forecast_7d": round(forecast_7d, 3),
        "recommended_action": recommended,
        "trends": trends,
    }

    db.add(
        LiaLearningPrediction(
            student_user_id=student_user_id,
            concept_key=concept_key,
            prediction_type="risk_and_mastery",
            prediction_json=prediction,
            confidence=0.75,
            created_at=datetime.now(UTC),
        )
    )
    db.add(
        LiaRiskAssessment(
            student_user_id=student_user_id,
            risk_level=risk_level,
            risk_score=risk_score,
            factors=prediction,
            assessed_at=datetime.now(UTC),
        )
    )
    return prediction
