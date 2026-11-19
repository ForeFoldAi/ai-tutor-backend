"""Observation Agent — validates and persists learning events."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.learning_intelligence.models import LiaLearningEvent
from app.modules.learning_intelligence.schemas import LearningEventIn
from app.services.learning_intelligence.orchestration.scope_keys import build_scope_key


def ingest_event(db: Session, event: LearningEventIn) -> LiaLearningEvent:
    scope = event.scope_key or build_scope_key(
        subject_name=event.subject_name or "",
        chapter_ids=[str(event.chapter_id)] if event.chapter_id else None,
    )
    row = LiaLearningEvent(
        student_user_id=event.student_user_id,
        school_id=event.school_id,
        event_type=event.event_type,
        occurred_at=datetime.now(UTC),
        scope_key=scope,
        subject_name=event.subject_name,
        chapter_id=event.chapter_id,
        chapter_name=event.chapter_name,
        concept_key=event.concept_key,
        payload=event.payload,
    )
    db.add(row)
    db.flush()
    return row


def recent_events(
    db: Session,
    student_user_id: int,
    *,
    limit: int = 100,
    event_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    stmt = (
        select(LiaLearningEvent)
        .where(LiaLearningEvent.student_user_id == student_user_id)
        .order_by(LiaLearningEvent.occurred_at.desc())
        .limit(limit)
    )
    if event_types:
        stmt = stmt.where(LiaLearningEvent.event_type.in_(event_types))
    rows = db.scalars(stmt).all()
    return [
        {
            "event_type": r.event_type,
            "payload": r.payload,
            "concept_key": r.concept_key,
            "subject_name": r.subject_name,
            "occurred_at": r.occurred_at.isoformat(),
        }
        for r in rows
    ]
