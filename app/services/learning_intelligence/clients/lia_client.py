"""LIA client — in-process integration for AI Tutor (fail-open)."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from app.core.database import SessionLocal
from app.modules.learning_intelligence.constants import (
    EVENT_ASSIGNMENT_SUBMITTED,
    EVENT_CHAPTER_COMPLETED,
    EVENT_CHAT_ASSISTANT_RESPONSE,
    EVENT_CHAT_USER_QUESTION,
    EVENT_QUIZ_ANSWER,
    EVENT_STUDY_SESSION_END,
    EVENT_VOICE_ASSISTANT_RESPONSE,
    EVENT_VOICE_USER_UTTERANCE,
)
from app.modules.learning_intelligence.schemas import LearningEventIn, TutorGuidanceRequest
from app.services.learning_intelligence.algorithms.mastery import concept_key_from_topic
from app.services.learning_intelligence.orchestration.scope_keys import build_scope_key

logger = logging.getLogger(__name__)

_QUIZ_ANSWER = re.compile(
    r"^\s*(?:(?:\d+\s*[-:.]\s*)?[a-dA-D](?:\s*,\s*(?:\d+\s*[-:.]\s*)?[a-dA-D])*|[a-dA-D]\s*$)",
    re.I,
)
_QUIZ_CONTEXT = re.compile(r"\*\*Questions?\*\*|Quiz Time|\*\*Quiz Time\*\*", re.I)


def lia_enabled() -> bool:
    return os.environ.get("LIA_ENABLED", "true").lower() in ("1", "true", "yes")


def emit_event_sync(
    *,
    event_type: str,
    student_user_id: int,
    school_id: int | None = None,
    subject_name: str | None = None,
    chapter_id: int | None = None,
    chapter_name: str | None = None,
    concept_key: str | None = None,
    scope_key: str = "",
    payload: dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget event ingestion — never blocks tutor on failure."""
    if not lia_enabled() or not student_user_id:
        return
    try:
        from app.modules.learning_intelligence import service as lia_svc

        event = LearningEventIn(
            event_type=event_type,
            student_user_id=student_user_id,
            school_id=school_id,
            subject_name=subject_name,
            chapter_id=chapter_id,
            chapter_name=chapter_name,
            concept_key=concept_key,
            scope_key=scope_key,
            payload=payload or {},
        )
        db = SessionLocal()
        try:
            lia_svc.ingest_event(db, event)
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.debug("LIA emit_event skipped: %s", exc)


async def emit_event_async(**kwargs: Any) -> None:
    """Schedule event emission without blocking the event loop."""
    import asyncio

    await asyncio.to_thread(emit_event_sync, **kwargs)


def get_guidance_for_turn_sync(
    *,
    student_user_id: int,
    query: str = "",
    topic: str = "",
    subject_name: str = "",
    chapter: str = "",
    chapter_ids: list[str] | None = None,
    class_level: str = "",
    board: str = "",
    agent_mode: str | None = None,
    understanding_scores: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    """Returns {learner_guidance, adaptive_guidance, prompt_instructions} or None."""
    if not lia_enabled() or not student_user_id:
        return None
    try:
        from app.modules.learning_intelligence import service as lia_svc

        req = TutorGuidanceRequest(
            student_user_id=student_user_id,
            query=query,
            topic=topic or query,
            subject_name=subject_name,
            chapter=chapter,
            chapter_ids=chapter_ids,
            class_level=class_level,
            board=board,
            agent_mode=agent_mode,
            understanding_scores=understanding_scores,
        )
        db = SessionLocal()
        try:
            guidance = lia_svc.fetch_tutor_guidance(db, req)
            db.commit()
            return {
                "learner_guidance": guidance.learner_guidance,
                "adaptive_guidance": guidance.adaptive_guidance,
                "prompt_instructions": guidance.prompt_instructions,
            }
        finally:
            db.close()
    except Exception as exc:
        logger.debug("LIA get_guidance skipped: %s", exc)
        return None


async def get_guidance_for_turn_async(**kwargs: Any) -> dict[str, str] | None:
    import asyncio

    return await asyncio.to_thread(get_guidance_for_turn_sync, **kwargs)


def emit_chat_user_question(
    *,
    student_user_id: int,
    school_id: int | None,
    query: str,
    subject_name: str,
    chapter: str,
    chapter_ids: list[str] | None,
    class_level: str,
    board: str,
    understanding_scores: dict | None,
    agent_mode: str | None,
    last_assistant: str = "",
) -> None:
    scope = build_scope_key(board=board, class_level=class_level, subject_name=subject_name, chapter_ids=chapter_ids)
    emit_event_sync(
        event_type=EVENT_CHAT_USER_QUESTION,
        student_user_id=student_user_id,
        school_id=school_id,
        subject_name=subject_name,
        chapter_name=chapter,
        concept_key=concept_key_from_topic(query, subject=subject_name, chapter=chapter),
        scope_key=scope,
        payload={
            "query": query[:500],
            "agent_mode": agent_mode,
            "board": board,
            "class_level": class_level,
            **(understanding_scores or {}),
        },
    )
    _maybe_emit_quiz_answer(
        student_user_id=student_user_id,
        query=query,
        subject_name=subject_name,
        chapter=chapter,
        last_assistant=last_assistant,
        understanding_scores=understanding_scores,
    )


def _maybe_emit_quiz_answer(
    *,
    student_user_id: int,
    query: str,
    subject_name: str,
    chapter: str,
    last_assistant: str,
    understanding_scores: dict | None,
) -> None:
    if not last_assistant or not _QUIZ_CONTEXT.search(last_assistant):
        return
    q = (query or "").strip()
    if not q or not _QUIZ_ANSWER.match(q):
        return
    emit_event_sync(
        event_type=EVENT_QUIZ_ANSWER,
        student_user_id=student_user_id,
        subject_name=subject_name,
        chapter_name=chapter,
        concept_key=concept_key_from_topic(q, subject=subject_name, chapter=chapter),
        payload={
            "query": q[:200],
            "is_correct": None,
            "response_words": len(q.split()),
            **(understanding_scores or {}),
        },
    )


def emit_chat_assistant_response(
    *,
    student_user_id: int,
    subject_name: str,
    chapter: str,
    topic: str,
    agent_mode: str | None,
) -> None:
    emit_event_sync(
        event_type=EVENT_CHAT_ASSISTANT_RESPONSE,
        student_user_id=student_user_id,
        subject_name=subject_name,
        chapter_name=chapter,
        concept_key=concept_key_from_topic(topic, subject=subject_name, chapter=chapter),
        payload={"topic": topic, "agent_mode": agent_mode},
    )


def emit_voice_turn(
    *,
    student_user_id: int,
    is_user: bool,
    text: str,
    subject_name: str,
    chapter: str,
    understanding_scores: dict | None,
) -> None:
    emit_event_sync(
        event_type=EVENT_VOICE_USER_UTTERANCE if is_user else EVENT_VOICE_ASSISTANT_RESPONSE,
        student_user_id=student_user_id,
        subject_name=subject_name,
        chapter_name=chapter,
        concept_key=concept_key_from_topic(text, subject=subject_name, chapter=chapter),
        payload={"text": text[:300], **(understanding_scores or {})},
    )


def emit_assignment_submitted(
    *,
    student_user_id: int,
    subject_name: str,
    score: float | None,
    max_score: float | None,
    assignment_title: str,
    result_items: list[dict] | None = None,
    chapter_name: str = "",
) -> None:
    emit_event_sync(
        event_type=EVENT_ASSIGNMENT_SUBMITTED,
        student_user_id=student_user_id,
        subject_name=subject_name,
        chapter_name=chapter_name or None,
        payload={
            "title": assignment_title,
            "score": score,
            "max_score": max_score,
            "is_correct": (score / max_score >= 0.6) if score is not None and max_score else None,
            "items": result_items or [],
        },
    )


def emit_chapter_completed(*, student_user_id: int, subject_name: str, chapter_name: str, chapter_id: int) -> None:
    emit_event_sync(
        event_type=EVENT_CHAPTER_COMPLETED,
        student_user_id=student_user_id,
        subject_name=subject_name,
        chapter_id=chapter_id,
        chapter_name=chapter_name,
        payload={"chapter_id": chapter_id},
    )


def emit_study_session_end(
    *,
    student_user_id: int,
    subject_name: str,
    duration_seconds: int,
    mode: str,
) -> None:
    emit_event_sync(
        event_type=EVENT_STUDY_SESSION_END,
        student_user_id=student_user_id,
        subject_name=subject_name,
        payload={"duration_seconds": duration_seconds, "mode": mode, "session_minutes": duration_seconds / 60},
    )
