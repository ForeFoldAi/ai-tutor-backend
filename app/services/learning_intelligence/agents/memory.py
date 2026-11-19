"""Memory Agent — long-term teaching memory and period summaries."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.learning_intelligence.constants import (
    EVENT_CHAPTER_COMPLETED,
    PERIOD_MONTHLY,
    PERIOD_WEEKLY,
)
from app.modules.learning_intelligence.models import LiaPeriodSummary, LiaStudentProfile, LiaTeachingMemory
from app.services.learning_intelligence.agents.observation import recent_events


def record_teaching_outcome(
    db: Session,
    *,
    student_user_id: int,
    concept_key: str,
    method_key: str,
    success: bool,
) -> None:
    from app.services.learning_intelligence.agents.student_modeling import get_or_create_profile

    row = db.scalar(
        select(LiaTeachingMemory).where(
            LiaTeachingMemory.student_user_id == student_user_id,
            LiaTeachingMemory.concept_key == concept_key,
            LiaTeachingMemory.method_key == method_key,
        )
    )
    if row is None:
        row = LiaTeachingMemory(
            student_user_id=student_user_id,
            concept_key=concept_key,
            method_key=method_key,
            updated_at=datetime.now(UTC),
        )
        db.add(row)
    if success:
        row.success_count += 1
        row.last_outcome = "success"
    else:
        row.failure_count += 1
        row.last_outcome = "failure"
    row.updated_at = datetime.now(UTC)

    profile = get_or_create_profile(db, student_user_id)
    memory = dict(profile.memory_profile or {})
    methods = memory.setdefault("methods", {})
    key = f"{concept_key}:{method_key}"
    methods[key] = {
        "success": row.success_count,
        "failure": row.failure_count,
        "last_outcome": row.last_outcome,
    }
    profile.memory_profile = memory


def build_period_summary(db: Session, student_user_id: int, period_type: str) -> dict:
    from app.services.learning_intelligence.agents.student_modeling import get_or_create_profile, profile_to_dict

    profile = get_or_create_profile(db, student_user_id)
    p = profile_to_dict(profile)
    days = 7 if period_type == PERIOD_WEEKLY else 30
    period_end = date.today()
    period_start = period_end - timedelta(days=days)

    summary = {
        "period_type": period_type,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "confidence_avg": p.get("confidence_score"),
        "engagement_avg": p.get("engagement_score"),
        "strong_concepts": p.get("strong_concepts", [])[:10],
        "weak_concepts": p.get("weak_concepts", [])[:10],
        "learning_style": p.get("learning_style"),
        "risk_level": p.get("risk_level"),
        "improvement_trend": p.get("improvement_trend"),
        "regression_trend": p.get("regression_trend"),
        "revision_behaviour": p.get("revision_behaviour"),
    }

    db.add(
        LiaPeriodSummary(
            student_user_id=student_user_id,
            period_type=period_type,
            period_start=period_start,
            period_end=period_end,
            summary_json=summary,
            created_at=datetime.now(UTC),
        )
    )

    memory = dict(profile.memory_profile or {})
    memory[period_type] = summary
    profile.memory_profile = memory
    return summary


def _latest_period_summary(db: Session, student_user_id: int, period_type: str) -> dict | None:
    row = db.scalar(
        select(LiaPeriodSummary)
        .where(
            LiaPeriodSummary.student_user_id == student_user_id,
            LiaPeriodSummary.period_type == period_type,
        )
        .order_by(LiaPeriodSummary.created_at.desc())
        .limit(1)
    )
    return dict(row.summary_json) if row else None


def best_teaching_methods(db: Session, student_user_id: int, concept_key: str, *, limit: int = 3) -> list[str]:
    rows = db.scalars(
        select(LiaTeachingMemory)
        .where(
            LiaTeachingMemory.student_user_id == student_user_id,
            LiaTeachingMemory.concept_key == concept_key,
        )
        .order_by(LiaTeachingMemory.success_count.desc())
        .limit(limit)
    ).all()
    out: list[str] = []
    for r in rows:
        if r.success_count > r.failure_count:
            out.append(r.method_key)
    return out


def detect_revision_behaviour(db: Session, student_user_id: int) -> dict[str, Any]:
    """Revisiting completed chapters or weak concepts counts as revision behaviour."""
    events = recent_events(db, student_user_id, limit=300)
    chapter_completes: dict[str, int] = {}
    weak_revisits = 0
    for ev in events:
        if ev.get("event_type") != EVENT_CHAPTER_COMPLETED:
            continue
        ch = str((ev.get("payload") or {}).get("chapter_id") or ev.get("concept_key") or "")
        if ch:
            chapter_completes[ch] = chapter_completes.get(ch, 0) + 1

    revisions = sum(1 for c in chapter_completes.values() if c > 1)
    return {
        "revision_sessions": revisions,
        "chapters_revisited": [k for k, v in chapter_completes.items() if v > 1][:10],
        "weak_concept_revisits": weak_revisits,
        "prefers_revision": revisions >= 2,
    }


def get_long_term_context(db: Session, student_user_id: int) -> dict[str, Any]:
    weekly = _latest_period_summary(db, student_user_id, PERIOD_WEEKLY)
    monthly = _latest_period_summary(db, student_user_id, PERIOD_MONTHLY)
    revision = detect_revision_behaviour(db, student_user_id)
    prof = db.get(LiaStudentProfile, student_user_id)
    return {
        "weekly_summary": weekly,
        "monthly_summary": monthly,
        "revision_behaviour": revision,
        "memory_profile": (prof.memory_profile if prof else {}) or {},
    }
