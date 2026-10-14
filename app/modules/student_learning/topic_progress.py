"""Chapter topic list + coverage progress (no agent — heading match + storage)."""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy.orm import Session

from app.modules.catalog.models import TextbookUpload
from app.modules.student_learning.models import StudentChapterProgress
from app.services.section_heading import (
    extract_headings_from_chunks,
    is_pedagogy_box_text,
    normalize_title,
    resolve_heading_scope,
    title_match_score,
)
from app.services.vector_service import fetch_chapter_chunks

logger = logging.getLogger(__name__)

_TOPICS_LEFT_RE = re.compile(
    r"(?:"
    r"topics?\s+(?:are\s+)?(?:left|remaining)|"
    r"remaining\s+topics?|"
    r"how\s+many\s+topics?|"
    r"what\s+topics?\s+(?:do\s+i\s+have\s+)?(?:left|remaining)|"
    r"topics?\s+i\s+(?:haven'?t|have\s+not)\s+(?:done|covered|studied)|"
    r"what(?:'?s|\s+is)\s+left\s+(?:in\s+(?:this\s+)?chapter)?"
    r")",
    re.I,
)

_MIN_MATCH_SCORE = 50.0  # section_heading returns 52 for single-token ⊂ multi-token title

_NOISE_TOPIC_RE = re.compile(
    r"^(?:chapter|unit|let'?s\s+(?:remember|explore)|don'?t\s+miss(?:\s+out)?|"
    r"think about|in a nutshell|alert|check your learning|give it a thought|"
    r"no warning|warning\b|watch\b|"
    r"including\b|such as\b)",
    re.I,
)


def is_topics_left_query(query: str) -> bool:
    return bool(_TOPICS_LEFT_RE.search((query or "").strip()))


def topic_key(title: str) -> str:
    return normalize_title(title)[:120]


def collection_name_for(board: str, class_level: str, subject_name: str) -> str:
    return f"{board}_{class_level}_{subject_name}".replace(" ", "_")


def _load_chapter_chunks(
    *,
    chapter_id: int,
    board: str = "",
    class_level: str = "",
    subject_name: str = "",
    chapter_name: str = "",
) -> list[Any]:
    if not subject_name:
        return []
    collection = collection_name_for(board, class_level, subject_name)
    chunks = fetch_chapter_chunks(
        collection,
        [str(chapter_id)],
        chapter_names=[chapter_name] if chapter_name else None,
    )
    if chunks:
        return chunks
    try:
        from app.services.pdf_text_layer import load_pdf_text_chunks_for_uploads

        return load_pdf_text_chunks_for_uploads([str(chapter_id)])
    except Exception:
        return []


def _is_usable_topic_title(title: str, *, chapter_name: str = "") -> bool:
    t = (title or "").strip()
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    if len(t) < 4 or len(t) > 80:
        return False
    if t.count(",") >= 2:
        return False  # place/name dumps
    if _NOISE_TOPIC_RE.match(t) or is_pedagogy_box_text(t):
        return False
    if chapter_name and normalize_title(chapter_name) == topic_key(t):
        return False
    # Require at least one letter
    if not re.search(r"[A-Za-z]", t):
        return False
    # Drop truncated / too-thin titles ("The Big", "A Note")
    words = [w for w in re.findall(r"[A-Za-z0-9']+", t) if w.lower() not in {"a", "an", "the", "of", "and", "in", "to"}]
    if len(words) < 2 and len(t) < 12:
        return False
    if len(words) == 1 and len(words[0]) < 5:
        return False
    return True


def list_chapter_topics(
    *,
    chapter_id: int,
    board: str = "",
    class_level: str = "",
    subject_name: str = "",
    chapter_name: str = "",
) -> list[dict[str, str]]:
    """Return durable topic list for a chapter from textbook headings."""
    chunks = _load_chapter_chunks(
        chapter_id=chapter_id,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
        chapter_name=chapter_name,
    )
    if not chunks:
        return []

    headings = extract_headings_from_chunks(chunks)
    # Prefer section-level numbered headings (2.1, 2.2); else all unique titles.
    sectionish = [h for h in headings if h.level >= 1 or (h.section_number and "." in h.section_number)]
    pool = sectionish if len(sectionish) >= 2 else headings

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for h in pool:
        title = (h.title or "").strip()
        if not _is_usable_topic_title(title, chapter_name=chapter_name):
            continue
        key = topic_key(title)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"key": key, "title": title})
        if len(out) >= 40:
            break
    return out


def match_topic_keys_for_query(
    query: str,
    topics: list[dict[str, str]],
    *,
    scope_title: str | None = None,
) -> list[str]:
    """Match student question to chapter topic keys."""
    matched: list[str] = []
    if scope_title:
        sk = topic_key(scope_title)
        if any(t["key"] == sk for t in topics):
            matched.append(sk)
        else:
            # Scope title may be a child of a listed topic — fuzzy attach
            best_key = ""
            best = 0.0
            for t in topics:
                score = title_match_score(scope_title, t["title"])
                if score > best:
                    best = score
                    best_key = t["key"]
            if best_key and best >= _MIN_MATCH_SCORE:
                matched.append(best_key)

    best_key = ""
    best = 0.0
    for t in topics:
        score = title_match_score(query, t["title"])
        if score > best:
            best = score
            best_key = t["key"]
    if best_key and best >= _MIN_MATCH_SCORE and best_key not in matched:
        matched.append(best_key)
    return matched


def _recompute_pct(topics_total: int, covered: list[str], *, completed: bool) -> int:
    if completed:
        return 100
    if topics_total <= 0:
        return 0
    return min(99, int(round(100 * len(set(covered)) / topics_total)))


def ensure_progress_row(
    db: Session,
    *,
    user_id: int,
    chapter_id: int,
    subject_name: str,
    chapter_name: str | None,
    scope_key: str,
    board: str = "",
    class_level: str = "",
) -> StudentChapterProgress:
    from datetime import UTC, datetime

    from sqlalchemy import select

    row = db.scalar(
        select(StudentChapterProgress).where(
            StudentChapterProgress.user_id == user_id,
            StudentChapterProgress.chapter_id == chapter_id,
        )
    )
    topics = list_chapter_topics(
        chapter_id=chapter_id,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
        chapter_name=chapter_name or "",
    )
    now = datetime.now(UTC)
    if row is None:
        row = StudentChapterProgress(
            user_id=user_id,
            chapter_id=chapter_id,
            subject_name=subject_name,
            chapter_name=chapter_name,
            status="in_progress",
            first_accessed_at=now,
            last_accessed_at=now,
            scope_key=scope_key,
            progress_pct=0,
            topics_total=len(topics),
            covered_topics=[],
        )
        db.add(row)
    else:
        row.last_accessed_at = now
        row.subject_name = subject_name
        if chapter_name:
            row.chapter_name = chapter_name
        row.scope_key = scope_key
        if topics and (not row.topics_total or row.topics_total != len(topics)):
            row.topics_total = len(topics)
        if row.status != "completed":
            row.status = "in_progress"
        row.progress_pct = _recompute_pct(
            row.topics_total or 0,
            list(row.covered_topics or []),
            completed=row.status == "completed",
        )
    db.flush()
    return row


def mark_query_topics_covered(
    db: Session,
    *,
    user_id: int,
    chapter_id: int,
    query: str,
    subject_name: str,
    chapter_name: str | None,
    scope_key: str,
    board: str = "",
    class_level: str = "",
    scope_title: str | None = None,
) -> StudentChapterProgress | None:
    """Record topics touched by this question; update progress_pct."""
    try:
        topics = list_chapter_topics(
            chapter_id=chapter_id,
            board=board,
            class_level=class_level,
            subject_name=subject_name,
            chapter_name=chapter_name or "",
        )
        if not topics:
            return ensure_progress_row(
                db,
                user_id=user_id,
                chapter_id=chapter_id,
                subject_name=subject_name,
                chapter_name=chapter_name,
                scope_key=scope_key,
                board=board,
                class_level=class_level,
            )

        keys = match_topic_keys_for_query(query, topics, scope_title=scope_title)
        row = ensure_progress_row(
            db,
            user_id=user_id,
            chapter_id=chapter_id,
            subject_name=subject_name,
            chapter_name=chapter_name,
            scope_key=scope_key,
            board=board,
            class_level=class_level,
        )
        covered = list(row.covered_topics or [])
        changed = False
        for k in keys:
            if k not in covered:
                covered.append(k)
                changed = True
        if changed or row.topics_total != len(topics):
            row.covered_topics = covered
            row.topics_total = len(topics)
            if row.status == "completed":
                row.progress_pct = 100
            else:
                row.progress_pct = _recompute_pct(len(topics), covered, completed=False)
                # Auto-complete when all topics covered
                if len(set(covered)) >= len(topics) and topics:
                    from datetime import UTC, datetime

                    row.status = "completed"
                    row.completed_at = datetime.now(UTC)
                    row.progress_pct = 100
            db.flush()
        return row
    except Exception:
        logger.exception("mark_query_topics_covered failed chapter_id=%s", chapter_id)
        return None


def chapter_topic_snapshot(
    db: Session,
    *,
    user_id: int,
    chapter_id: int,
    subject_name: str,
    chapter_name: str | None,
    board: str = "",
    class_level: str = "",
) -> dict[str, Any]:
    """Topics total/covered/remaining + progress for API or tutor answers."""
    from sqlalchemy import select

    topics = list_chapter_topics(
        chapter_id=chapter_id,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
        chapter_name=chapter_name or "",
    )
    row = db.scalar(
        select(StudentChapterProgress).where(
            StudentChapterProgress.user_id == user_id,
            StudentChapterProgress.chapter_id == chapter_id,
        )
    )
    covered_keys = set(row.covered_topics or []) if row else set()
    if row and row.status == "completed":
        covered = list(topics)
        remaining: list[dict[str, str]] = []
        pct = 100
    else:
        covered = [t for t in topics if t["key"] in covered_keys]
        remaining = [t for t in topics if t["key"] not in covered_keys]
        pct = int(row.progress_pct) if row else 0
        if topics and not row:
            pct = 0
        elif topics:
            pct = min(100 if not remaining else 99, int(round(100 * len(covered) / len(topics))))

    return {
        "topics_total": len(topics),
        "topics_covered": len(covered),
        "progress_pct": pct,
        "covered": covered,
        "remaining": remaining,
        "all_topics": topics,
        "status": (row.status if row else "not_started"),
    }


def format_topics_left_answer(snapshot: dict[str, Any], *, chapter_label: str = "this chapter") -> str:
    total = int(snapshot.get("topics_total") or 0)
    remaining = snapshot.get("remaining") or []
    covered = snapshot.get("covered") or []
    pct = int(snapshot.get("progress_pct") or 0)

    if total == 0:
        return (
            f"I could not find a topic list for {chapter_label} yet. "
            "Try asking about a specific idea from the chapter, or open the chapter again after it finishes indexing."
        )

    left = len(remaining)
    lines = [
        f"In {chapter_label} there are {total} topics. You have covered {len(covered)} ({pct}%).",
    ]
    if left == 0:
        lines.append("You have covered all topics in this chapter. Great work!")
        return "\n".join(lines)

    lines.append(f"Topics left ({left}):")
    for i, t in enumerate(remaining[:20], start=1):
        lines.append(f"{i}. {t['title']}")
    if left > 20:
        lines.append(f"…and {left - 20} more.")
    if covered:
        lines.append("Already covered:")
        for i, t in enumerate(covered[:8], start=1):
            lines.append(f"{i}. {t['title']}")
    return "\n".join(lines)


def try_topics_left_reply(
    db: Session,
    *,
    user_id: int,
    query: str,
    chapter_ids: list[str] | None,
    subject_name: str = "",
    chapter: str = "",
    board: str = "",
    class_level: str = "",
) -> str | None:
    """If the student asks how many topics are left, return a ready answer."""
    if not is_topics_left_query(query) or not chapter_ids:
        return None
    try:
        chapter_id = int(str(chapter_ids[0]).strip())
    except (TypeError, ValueError):
        return None
    meta = resolve_chapter_meta_from_upload(db, chapter_id)
    snap = chapter_topic_snapshot(
        db,
        user_id=user_id,
        chapter_id=chapter_id,
        subject_name=subject_name or meta.get("subject_name") or "",
        chapter_name=chapter or meta.get("chapter_name"),
        board=board or meta.get("board") or "",
        class_level=class_level or meta.get("class_level") or "",
    )
    label = chapter or meta.get("chapter_name") or "this chapter"
    return format_topics_left_answer(snap, chapter_label=label)


def record_turn_topic_progress(
    *,
    student_user_id: int,
    query: str,
    chapter_ids: list[str] | None,
    subject_name: str = "",
    chapter: str = "",
    board: str = "",
    class_level: str = "",
    scope_title: str | None = None,
) -> None:
    """Fail-open: mark matched chapter topics covered for this tutoring turn."""
    if not chapter_ids or not student_user_id:
        return
    try:
        chapter_id = int(str(chapter_ids[0]).strip())
    except (TypeError, ValueError):
        return
    try:
        from app.core.database import SessionLocal
        from app.modules.student_learning.service import ensure_scope_or_reset
        from app.modules.users.models import User

        db = SessionLocal()
        try:
            user = db.get(User, student_user_id)
            if user is None:
                return
            scope_key = ensure_scope_or_reset(db, user)
            meta = resolve_chapter_meta_from_upload(db, chapter_id)
            mark_query_topics_covered(
                db,
                user_id=student_user_id,
                chapter_id=chapter_id,
                query=query,
                subject_name=subject_name or meta.get("subject_name") or "",
                chapter_name=chapter or meta.get("chapter_name"),
                scope_key=scope_key,
                board=board or meta.get("board") or "",
                class_level=class_level or meta.get("class_level") or "",
                scope_title=scope_title,
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    except Exception:
        logger.exception("record_turn_topic_progress failed")


def resolve_chapter_meta_from_upload(db: Session, chapter_id: int) -> dict[str, str]:
    tb = db.get(TextbookUpload, chapter_id)
    if tb is None:
        return {}
    board = getattr(tb.board, "value", None) or str(tb.board or "")
    class_level = getattr(tb.class_level, "value", None) or str(tb.class_level or "")
    return {
        "board": board,
        "class_level": class_level,
        "subject_name": tb.subject_name or "",
        "chapter_name": tb.chapter or tb.file_name or "",
    }


if __name__ == "__main__":
    assert topic_key("  Rain Fall ") == "rain fall"
    assert is_topics_left_query("How many topics left?")
    assert is_topics_left_query("what topics are remaining in this chapter")
    assert not is_topics_left_query("what is rain")
    snap = {
        "topics_total": 3,
        "progress_pct": 33,
        "covered": [{"key": "a", "title": "Rain"}],
        "remaining": [{"key": "b", "title": "Wind"}, {"key": "c", "title": "Clouds"}],
    }
    ans = format_topics_left_answer(snap, chapter_label="Weather")
    assert "Topics left (2)" in ans and "Wind" in ans
    print("topic_progress self-check ok")
