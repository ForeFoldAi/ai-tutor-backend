"""LIA orchestrator — coordinates sub-agents."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.modules.learning_intelligence.constants import GUIDANCE_CACHE_TTL
from app.modules.learning_intelligence.models import LiaTutorGuidanceCache
from app.modules.learning_intelligence.schemas import (
    LearningEventIn,
    LearningEventResponse,
    TutorGuidanceObject,
    TutorGuidanceRequest,
)
from app.services.learning_intelligence.agents.knowledge_intelligence import process_event_for_knowledge
from app.services.learning_intelligence.agents.memory import record_teaching_outcome
from app.services.learning_intelligence.agents.observation import ingest_event
from app.services.learning_intelligence.agents.student_modeling import update_twin_from_events
from app.services.learning_intelligence.agents.teacher_coach import build_teacher_summary
from app.services.learning_intelligence.agents.tutor_coach import build_tutor_guidance
from app.services.learning_intelligence.algorithms.mastery import concept_key_from_topic, topic_hash
from app.services.learning_intelligence.orchestration.scope_keys import build_scope_key

logger = logging.getLogger(__name__)


def process_learning_event(db: Session, event: LearningEventIn) -> LearningEventResponse:
    """Full event pipeline: observe → knowledge → twin update."""
    row = ingest_event(db, event)
    process_event_for_knowledge(db, event)
    _record_memory_from_event(db, event)
    update_twin_from_events(db, event.student_user_id)
    return LearningEventResponse(event_id=row.id)


def get_tutor_guidance(db: Session, req: TutorGuidanceRequest) -> TutorGuidanceObject:
    scope = build_scope_key(
        board=req.board,
        class_level=req.class_level,
        subject_name=req.subject_name,
        chapter_ids=req.chapter_ids,
    )
    topic = req.topic or req.query
    th = topic_hash(topic, scope)

    cached = db.scalar(
        select(LiaTutorGuidanceCache).where(
            LiaTutorGuidanceCache.student_user_id == req.student_user_id,
            LiaTutorGuidanceCache.scope_key == scope,
            LiaTutorGuidanceCache.topic_hash == th,
            LiaTutorGuidanceCache.expires_at > datetime.now(UTC),
        )
    )
    if cached and cached.guidance_json:
        return TutorGuidanceObject.model_validate(cached.guidance_json)

    guidance = build_tutor_guidance(db, req)
    _cache_guidance(db, req.student_user_id, scope, th, guidance)
    return guidance


def _cache_guidance(
    db: Session,
    student_user_id: int,
    scope_key: str,
    th: str,
    guidance: TutorGuidanceObject,
) -> None:
    now = datetime.now(UTC)
    db.execute(
        delete(LiaTutorGuidanceCache).where(
            LiaTutorGuidanceCache.student_user_id == student_user_id,
            LiaTutorGuidanceCache.scope_key == scope_key,
            LiaTutorGuidanceCache.topic_hash == th,
        )
    )
    db.add(
        LiaTutorGuidanceCache(
            student_user_id=student_user_id,
            scope_key=scope_key,
            topic_hash=th,
            guidance_json=guidance.model_dump(),
            created_at=now,
            expires_at=now + timedelta(seconds=GUIDANCE_CACHE_TTL),
        )
    )


def _record_memory_from_event(db: Session, event: LearningEventIn) -> None:
    concept = event.concept_key or concept_key_from_topic(
        event.payload.get("topic", ""),
        subject=event.subject_name or "",
        chapter=event.chapter_name or "",
    )
    method = event.payload.get("explanation_style") or event.payload.get("agent_mode") or "default"
    if event.payload.get("is_correct") is True:
        record_teaching_outcome(db, student_user_id=event.student_user_id, concept_key=concept, method_key=method, success=True)
    elif event.payload.get("is_correct") is False or float(event.payload.get("confusion", 0)) >= 0.55:
        record_teaching_outcome(db, student_user_id=event.student_user_id, concept_key=concept, method_key=method, success=False)


def get_teacher_summary_for_student(db: Session, student_user_id: int):
    return build_teacher_summary(db, student_user_id)
