"""Resolve the vector collection for a chat turn from the chapter's own upload row.

The collection was built from the client's ``subject_name`` ("Social"), but textbooks are
indexed under the uploaded subject name ("Social - Part-I" -> CBSE_CLASS_8_Social_-_Part-I).
When they differ, retrieval silently returned no passages and the tutor answered without the
textbook (Suneel QA: Science/Social chapters got 0 passages). The chapter ids identify the
upload exactly, so they win over the free-text subject.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_CACHE: dict[tuple, tuple[float, str | None]] = {}
_TTL_SEC = 300.0


def collection_name(board: str, class_level: str, subject_name: str) -> str:
    return f"{board}_{class_level}_{subject_name}".replace(" ", "_")


def _upload_collection(chapter_ids: tuple[str, ...]) -> str | None:
    ids = [int(c) for c in chapter_ids if str(c).strip().isdigit()]
    if not ids:
        return None
    from sqlalchemy import text

    from app.core.database import SessionLocal

    with SessionLocal() as db:
        rows = db.execute(
            text("SELECT board, class_level, subject_name FROM textbook_uploads WHERE id = ANY(:ids)"),
            {"ids": ids},
        ).fetchall()
    names = {collection_name(r[0], r[1], r[2]) for r in rows if r[0] and r[1] and r[2]}
    # Chapters from two different books: no single collection — keep the client's.
    return names.pop() if len(names) == 1 else None


def resolve_chat_collection(
    board: str, class_level: str, subject_name: str, chapter_ids: list[str] | None
) -> str:
    requested = collection_name(board, class_level, subject_name)
    key = tuple(sorted(str(c) for c in (chapter_ids or []) if str(c).strip()))
    if not key:
        return requested
    hit = _CACHE.get(key)
    if hit and time.monotonic() - hit[0] < _TTL_SEC:
        resolved = hit[1]
    else:
        try:
            resolved = _upload_collection(key)
        except Exception as exc:  # DB hiccup: never block the answer
            logger.warning("chapter collection lookup failed: %s", exc)
            return requested
        _CACHE[key] = (time.monotonic(), resolved)
    if resolved and resolved != requested:
        logger.info("[CHAT] collection %r -> %r from chapter ids %s", requested, resolved, list(key))
        return resolved
    return requested
