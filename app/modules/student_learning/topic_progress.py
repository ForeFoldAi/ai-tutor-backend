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

# Full inventory (not "what's left"): "what topics are present in this chapter?"
_TOPICS_LIST_RE = re.compile(
    r"(?:"
    r"what\s+topics?\s+(?:are\s+)?(?:present|covered|there|included|in)\b|"
    r"topics?\s+(?:are\s+)?(?:present|covered|included|there)\s+in\b|"
    r"topics?\s+(?:in|of|from|covered\s+in)\s+(?:(?:this|the|current)\s+)?chapter\b|"
    r"(?:list|name|give)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?topics?\b|"
    r"(?:show|tell)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?topics?\b|"
    r"table\s+of\s+contents|chapter\s+contents?\b|"
    r"what\s+(?:is|are)\s+(?:the\s+)?(?:main\s+)?topics?\b"
    r")",
    re.I,
)

_MIN_MATCH_SCORE = 50.0  # section_heading returns 52 for single-token ⊂ multi-token title

_NOISE_TOPIC_RE = re.compile(
    r"^(?:chapter|unit|let'?s\s+(?:remember|explore)|don'?t\s+miss(?:\s+out)?|"
    r"think about|in a nutshell|alert|check your learning|give it a thought|"
    r"no warning|warning\b|watch\b|"
    r"including\b|such as\b|"
    # End-matter / non-teaching headings that must never appear in a student TOC
    r"exercises?|questions?|activities?|projects?|summary|keypoints?|"
    r"what\s+you\s+have\s+learnt|extend|group\s+discussion|index|contents?|"
    r"multiple\s+choice|fill\s+in|match\s+the|true\s+or\s+false)\b",
    re.I,
)


def is_topics_left_query(query: str) -> bool:
    return bool(_TOPICS_LEFT_RE.search((query or "").strip()))


def is_chapter_topics_list_query(query: str) -> bool:
    """True when the student wants the full chapter topic inventory (not remaining-only)."""
    q = (query or "").strip()
    if not q or is_topics_left_query(q):
        return False
    return bool(_TOPICS_LIST_RE.search(q))


def topic_key(title: str) -> str:
    return normalize_title(title)[:120]


def _section_sort_key(number: str, title: str) -> tuple:
    parts: list[int] = []
    for p in (number or "").split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return (tuple(parts) if parts else (10**9,), (title or "").lower())


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


_SENTENCE_FRAGMENT_TOPIC_RE = re.compile(
    r"^(?:if|when|now|as|for|with|from|after|before|m\.|cm\b|mm\b)|"
    r"\b(?:now you|familiar with|is placed|is located|in size|"
    r"the position,\s*size|table\b)\b",
    re.I,
)

# Formula / sign-convention dumps overwhelm a first topic list for weak learners.
_HEAVY_SUBTOPIC_RE = re.compile(
    r"\b(formula|magnification|sign\s+convention|power\s+of\s+a\s+lens|"
    r"ray\s+diagrams?)\b",
    re.I,
)


def _is_usable_section_number(num: str) -> bool:
    """Reject measurement / figure-caption numbers (3.00, 4.0) mistaken as sections."""
    n = (num or "").strip()
    if not n or "." not in n:
        return False
    parts = n.split(".")
    if not all(p.isdigit() for p in parts):
        return False
    # "3.00" / "4.0" look like measurements, not NCERT section ids.
    if any(len(p) >= 2 and p != str(int(p)) for p in parts[1:]):
        return False
    if len(parts) == 2 and parts[1] == "0":
        return False
    return True


def _is_usable_topic_title(title: str, *, chapter_name: str = "") -> bool:
    from app.services.section_heading import normalize_heading_line

    t = normalize_heading_line(title or "")
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    # Titles may still carry a leading section number from bad metadata.
    t = re.sub(r"^\d+(?:\.\d+)+\s+", "", t).strip()
    if len(t) < 4 or len(t) > 80:
        return False
    if t.count(",") >= 2:
        return False  # place/name dumps
    if _NOISE_TOPIC_RE.match(t) or is_pedagogy_box_text(t):
        return False
    if _SENTENCE_FRAGMENT_TOPIC_RE.search(t):
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


def _title_display(title: str) -> str:
    from app.services.section_heading import normalize_heading_line

    t = normalize_heading_line(title or "")
    t = re.sub(r"^\d+(?:\.\d+)+\s+", "", t).strip()
    # NCERT often prints section titles in ALL CAPS — soften for students.
    if t.isupper() and len(t) > 4:
        t = t.title()
    return t


def list_chapter_topics(
    *,
    chapter_id: int,
    board: str = "",
    class_level: str = "",
    subject_name: str = "",
    chapter_name: str = "",
) -> list[dict[str, str]]:
    """Return durable topic list for a chapter from textbook headings."""
    from app.services.section_heading import normalize_heading_line, parse_heading_line

    chunks = _load_chapter_chunks(
        chapter_id=chapter_id,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
        chapter_name=chapter_name,
    )
    if not chunks:
        return []

    headings = list(extract_headings_from_chunks(chunks))
    # Full-text scan: metadata alone often only has exercises / deep subsections.
    for doc in chunks:
        for raw in (getattr(doc, "page_content", "") or "").splitlines():
            line = normalize_heading_line(raw)
            if not line or len(line) > 160:
                continue
            parsed = parse_heading_line(line)
            if not parsed:
                continue
            number, title, level = parsed
            if not number or "." not in number:
                continue
            headings.append(
                type("H", (), {
                    "section_number": number,
                    "title": title,
                    "level": level,
                    "raw_hint": line,
                    "page": 0,
                })()
            )

    # Prefer real main sections (9.1, 9.2). Figure captions (9.7, 9.16) often look
    # numbered — keep only one-dot ids that parse as short section titles, then
    # fill thin TOCs with true subsections (9.2.1), never figure numbers.
    def _num(h: Any) -> str:
        return (getattr(h, "section_number", None) or "").strip()

    numbered = [h for h in headings if _is_usable_section_number(_num(h))]
    main = [h for h in numbered if _num(h).count(".") == 1]
    sub = [h for h in numbered if _num(h).count(".") == 2]

    def _collect(pool: list[Any], *, limit: int) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        seen_local: set[str] = set()
        for h in pool:
            num = _num(h)
            title = _title_display(getattr(h, "title", "") or "")
            # Trim trailing prose glued onto headings by OCR.
            title = re.split(r"\bNow you\b|\bTable\b", title, maxsplit=1)[0].strip(" -:,")
            if not _is_usable_topic_title(title, chapter_name=chapter_name):
                continue
            # One-dot figure captions tend to be long sentence-like titles.
            if num.count(".") == 1 and (len(title) > 48 or title.lower().startswith("ray diagrams")):
                continue
            key = topic_key(title)
            if not key or key in seen_local:
                continue
            seen_local.add(key)
            row = {"key": key, "title": title}
            if num:
                row["section_number"] = num
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows

    out = _collect(main, limit=40)
    # Prefer the dense low band (9.1–9.3); drop figure numbers (9.7+).
    if len(out) >= 2:
        try:
            low = [r for r in out if int(r["section_number"].split(".")[1]) <= 6]
            if len(low) >= 2:
                out = low
        except (ValueError, IndexError, KeyError):
            pass

    parent_nums = {r["section_number"] for r in out if r.get("section_number")}
    if len(out) < 5 and parent_nums:
        have = {r["key"] for r in out}
        child_pool = [
            h for h in sub
            if any(_num(h).startswith(p + ".") for p in parent_nums)
        ]
        # Prefer concrete idea topics (glass slab, lenses) over formula dumps.
        light = []
        heavy = []
        for row in _collect(child_pool, limit=40):
            if row["key"] in have:
                continue
            if _HEAVY_SUBTOPIC_RE.search(row.get("title") or ""):
                heavy.append(row)
            else:
                light.append(row)
        # Prefer concrete idea topics; only use formula/sign dumps if still too thin.
        for row in light if len(light) + len(out) >= 5 else (light + heavy):
            out.append(row)
            have.add(row["key"])
            if len(out) >= 9:
                break
    out.sort(key=lambda t: _section_sort_key(t.get("section_number") or "", t.get("title") or ""))
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


def format_chapter_topics_list_answer(
    topics: list[dict[str, str]],
    *,
    chapter_label: str = "this chapter",
    spoken: bool = False,
) -> str:
    """Full topic inventory for text chat (bullets) or voice (spoken numbering)."""
    clean = [t for t in (topics or []) if (t.get("title") or "").strip()]
    n = len(clean)
    if n == 0:
        if spoken:
            return (
                f"I could not find the topic list for {chapter_label} yet. "
                "Ask about one idea from the chapter, or open the chapter again after it finishes indexing."
            )
        return (
            f"I could not find a topic list for {chapter_label} yet. "
            "Try asking about a specific idea from the chapter, or open the chapter again after it finishes indexing."
        )

    def _label(t: dict[str, str]) -> str:
        title = (t.get("title") or "").strip()
        num = (t.get("section_number") or "").strip()
        return f"{num} {title}".strip() if num else title

    if spoken:
        parts = [f"This chapter, {chapter_label}, has {n} main topics."]
        for i, t in enumerate(clean[:25], start=1):
            parts.append(f"Topic {i}: {_label(t)}.")
        if n > 25:
            parts.append(f"And {n - 25} more after that.")
        parts.append("Which topic should we start with?")
        return " ".join(parts)

    lines = [f"**Topics in {chapter_label}** ({n}):", ""]
    for i, t in enumerate(clean[:40], start=1):
        lines.append(f"{i}. {_label(t)}")
    if n > 40:
        lines.append(f"…and {n - 40} more.")
    lines.append("")
    lines.append("Which topic would you like to learn first?")
    return "\n".join(lines)


def topics_from_retrieved_docs(
    docs: list[Any] | None,
    *,
    chapter_name: str = "",
) -> list[dict[str, str]]:
    """Build a topic list from retrieved textbook chunks when DB heading fetch is empty."""
    if not docs:
        return []
    from app.services.section_heading import parse_heading_line

    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(title: str, number: str = "") -> bool:
        title = (title or "").strip()
        if not _is_usable_topic_title(title, chapter_name=chapter_name):
            return len(out) >= 40
        key = topic_key(title)
        if not key or key in seen:
            return len(out) >= 40
        seen.add(key)
        row = {"key": key, "title": title}
        if number:
            row["section_number"] = number
        out.append(row)
        return len(out) >= 40

    # Metadata / early-line headings from the shared extractor.
    for h in extract_headings_from_chunks(docs):
        if h.level < 1 and not (h.section_number and "." in (h.section_number or "")):
            continue
        if _add(h.title or "", (h.section_number or "").strip()):
            return out

    # Full-text scan so later section headings are not dropped by the extractor's line peek.
    for doc in docs:
        for raw in (getattr(doc, "page_content", "") or "").splitlines():
            line = raw.strip()
            if not line or len(line) > 160:
                continue
            parsed = parse_heading_line(line)
            if not parsed:
                continue
            number, title, level = parsed
            if level < 1 and not (number and "." in number):
                continue
            if _add(title or line, (number or "").strip()):
                break
    out.sort(key=lambda t: _section_sort_key(t.get("section_number") or "", t.get("title") or ""))
    return out[:40]


def try_chapter_topics_list_reply(
    db: Session | None,
    *,
    query: str,
    chapter_ids: list[str] | None,
    subject_name: str = "",
    chapter: str = "",
    board: str = "",
    class_level: str = "",
    docs: list[Any] | None = None,
    spoken: bool = False,
) -> str | None:
    """If the student asks for the chapter topic list, return a ready inventory answer."""
    if not is_chapter_topics_list_query(query):
        return None
    topics: list[dict[str, str]] = []
    label = chapter or "this chapter"
    if chapter_ids:
        try:
            chapter_id = int(str(chapter_ids[0]).strip())
        except (TypeError, ValueError):
            chapter_id = 0
        if chapter_id and db is not None:
            meta = resolve_chapter_meta_from_upload(db, chapter_id)
            label = chapter or meta.get("chapter_name") or label
            topics = list_chapter_topics(
                chapter_id=chapter_id,
                board=board or meta.get("board") or "",
                class_level=class_level or meta.get("class_level") or "",
                subject_name=subject_name or meta.get("subject_name") or "",
                chapter_name=label,
            )
    if not topics:
        topics = topics_from_retrieved_docs(docs, chapter_name=label)
    if not topics:
        return None
    return format_chapter_topics_list_answer(topics, chapter_label=label, spoken=spoken)


def chapter_toc_context_block(
    topics: list[dict[str, str]],
    *,
    max_n: int = 40,
) -> str:
    """Inject ordered topic titles into LLM context for whole-chapter walkthroughs."""
    clean = [t for t in (topics or []) if (t.get("title") or "").strip()]
    if not clean:
        return ""
    lines = ["CHAPTER TOPICS IN ORDER (cover each one; do not skip):"]
    for i, t in enumerate(clean[:max_n], start=1):
        title = (t.get("title") or "").strip()
        num = (t.get("section_number") or "").strip()
        label = f"{num} {title}".strip() if num else title
        lines.append(f"{i}. {label}")
    return "\n".join(lines)


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
