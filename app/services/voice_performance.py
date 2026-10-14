"""
Phase 6 — voice pipeline performance helpers.

Session pre-warm (embeddings + RAG probe), structured metrics export,
and shared warmup entry points for startup + WebSocket session_start.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol

from app.config import VOICE_EMBEDDING_WARMUP, VOICE_RAG_PREWARM, VOICE_RETRIEVAL_K

logger = logging.getLogger(__name__)

# ponytail: generic probe query — warms Chroma + embedding path, not answer-specific
_RAG_PREWARM_QUERY = "chapter introduction overview"


class _VoiceSession(Protocol):
    board: str
    subject_name: str
    chapter_ids: list[str] | None
    student_key: str
    learner_profile: Any
    collection: str

    def configure(self, msg: dict) -> None: ...


def warm_embeddings_sync() -> bool:
    """Load BGE embeddings once so first voice RAG query avoids cold-start."""
    if not VOICE_EMBEDDING_WARMUP:
        return False
    try:
        from app.services.vector_service import _get_embedding_model, is_embedding_model_loaded

        if is_embedding_model_loaded():
            return True
        _get_embedding_model()
        return is_embedding_model_loaded()
    except Exception as exc:
        logger.debug("Embedding warmup skipped: %s", exc)
        return False


def prewarm_rag_sync(collection_name: str, chapter_ids: list[str] | None) -> bool:
    """One lightweight retrieval to warm Chroma collection + embed path."""
    if not VOICE_RAG_PREWARM or not collection_name:
        return False
    try:
        from app.services.section_retrieval import retrieve_for_tutor_query

        retrieve_for_tutor_query(
            _RAG_PREWARM_QUERY,
            collection_name=collection_name,
            chapter_ids=chapter_ids,
            k=VOICE_RETRIEVAL_K,
        )
        return True
    except Exception as exc:
        logger.debug("RAG prewarm skipped for %s: %s", collection_name, exc)
        return False


async def warm_voice_stack_async() -> None:
    """Async wrapper for embedding warmup (runs in thread pool)."""
    if not VOICE_EMBEDDING_WARMUP:
        return
    await asyncio.to_thread(warm_embeddings_sync)


async def prewarm_voice_session(session: _VoiceSession) -> None:
    """
    Overlap expensive cold starts with greeting / mic permission UI.

    Runs after session_start — never blocks the listening handshake.
    """
    warmed_embed = await warm_voice_stack_async()
    warmed_rag = False
    if session.board and session.subject_name:
        warmed_rag = await asyncio.to_thread(
            prewarm_rag_sync,
            session.collection,
            session.chapter_ids,
        )
    if warmed_embed or warmed_rag:
        logger.info(
            "[voice-prewarm] collection=%s embed=%s rag=%s",
            session.collection or "?",
            warmed_embed,
            warmed_rag,
        )


def schedule_session_prewarm(session: _VoiceSession) -> None:
    """Fire-and-forget session prewarm (safe from WebSocket handler)."""
    asyncio.create_task(prewarm_voice_session(session), name="voice-session-prewarm")


async def start_learner_profile_load(session: _VoiceSession) -> None:
    """Begin Redis learner profile fetch during greeting — not on first question."""
    if not session.student_key or session.learner_profile is not None:
        return
    from app.services.learner_profile import load_learner_profile

    async def _run() -> None:
        try:
            session.learner_profile = await load_learner_profile(session.student_key)
        except Exception as exc:
            logger.debug("Learner profile preload failed: %s", exc)

    task = asyncio.create_task(_run(), name="voice-learner-preload")
    setattr(session, "_learner_load_task", task)


def schedule_learner_profile_load(session: _VoiceSession) -> None:
    if not session.student_key:
        return
    asyncio.create_task(start_learner_profile_load(session), name="voice-learner-preload-scheduler")


async def resolve_learner_profile(session: _VoiceSession, *, wait_ms: float = 50.0) -> None:
    """
    Use preloaded profile if ready; brief wait avoids blocking first token on Redis.
    """
    if session.learner_profile is not None or not session.student_key:
        return
    task = getattr(session, "_learner_load_task", None)
    if task is None:
        start_learner_profile_load(session)
        task = getattr(session, "_learner_load_task", None)
    if task is None:
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=wait_ms / 1000.0)
    except asyncio.TimeoutError:
        # ponytail: profile arrives on next turn; don't stall first token
        return
    except Exception as exc:
        logger.debug("Learner profile resolve failed: %s", exc)


def export_voice_turn_metrics(
    timing: Any,
    *,
    phase: str,
    **extra: str,
) -> None:
    """Structured log line for external metrics collectors (JSON)."""
    try:
        payload = {
            "event": "voice_turn_metrics",
            "phase": phase,
            "turn_id": getattr(timing, "turn_id", "") or "",
            **timing.summary(),
            **extra,
        }
        logger.info("voice_metrics %s", json.dumps(payload, ensure_ascii=False))
    except Exception as exc:
        logger.debug("voice metrics export skipped: %s", exc)
