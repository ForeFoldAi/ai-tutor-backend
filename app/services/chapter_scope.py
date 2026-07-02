"""
Chapter Awareness Mode — programmatic coverage assessment and student choice handling.

Detects when a question is fully, partially, or not covered by the selected chapter
using retrieval evidence (not prompt-only instructions).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)

_CHAPTER_NUM_RE = re.compile(r"chapter\s*(\d+)", re.I)

_TOPIC_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
    "and", "or", "to", "for", "with", "by", "from", "this", "that", "it",
    "its", "as", "be", "been", "being", "have", "has", "had", "not", "no",
    "explain", "describe", "what", "how", "why", "when", "where", "about",
    "tell", "give", "can", "you", "me", "please", "define", "meaning",
})

_SKIP_TOPIC_SCOPE_RE = re.compile(
    r"^(hi|hello|hey|hii|thanks|thank you|yes|yeah|yep|no|ok|okay|sure|"
    r"continue|go on|tell me more|explain more|simplify|summarize)\b",
    re.I,
)

_AWARENESS_MARKER = "How would you like to continue?"

_STAY_CHOICE_RE = re.compile(
    r"^(?:a\b|option\s*a\b|stay(?:\s+within)?(?:\s+the)?\s+current\s+chapter|"
    r"current\s+chapter\s+only|within\s+this\s+chapter|chapter\s+only)\b",
    re.I,
)
_SWITCH_CHOICE_RE = re.compile(
    r"^(?:b\b|option\s*b\b|switch(?:\s+chapter)?|other\s+chapter|"
    r"relevant\s+chapter|change\s+chapter)\b",
    re.I,
)
_GENERAL_CHOICE_RE = re.compile(
    r"^(?:c\b|option\s*c\b|general(?:\s+explanation)?|full\s+explanation|"
    r"explain\s+anyway|beyond\s+(?:this\s+)?chapter)\b",
    re.I,
)


class ChapterCoverageLevel(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class ChapterScopeChoice(str, Enum):
    STAY = "stay"
    SWITCH = "switch"
    GENERAL = "general"


@dataclass
class ChapterCoverageAssessment:
    level: ChapterCoverageLevel
    topic_label: str
    current_chapter_label: str
    other_chapter_label: str | None = None
    other_chapter_id: str | None = None
    selected_hits: int = 0
    required_hits: int = 1
    selected_coverage: int = 0


def chapter_number_from_name(name: str) -> int | None:
    """Extract chapter number from labels like 'Chapter 2 - Understanding the Weather'."""
    m = _CHAPTER_NUM_RE.search(name or "")
    return int(m.group(1)) if m else None


def extract_mentioned_chapter_numbers(query: str) -> set[int]:
    return {int(m.group(1)) for m in _CHAPTER_NUM_RE.finditer(query or "")}


def selected_chapter_numbers(chapter_names: list[str]) -> set[int]:
    nums: set[int] = set()
    for name in chapter_names:
        n = chapter_number_from_name(name)
        if n is not None:
            nums.add(n)
    return nums


def substantive_query_terms(query: str) -> set[str]:
    terms = set(re.findall(r"\w+", (query or "").lower()))
    return {t for t in terms if len(t) >= 3 and t not in _TOPIC_STOPWORDS}


def is_chapter_awareness_prompt(text: str) -> bool:
    return _AWARENESS_MARKER in (text or "")


def _format_chapter_list(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]}, {names[1]}, and {len(names) - 2} more"


def _current_chapter_label(chapter_names: list[str] | None) -> str:
    names = [n.strip() for n in (chapter_names or []) if n and str(n).strip()]
    if not names:
        return "your selected chapter"
    return _format_chapter_list(names)


def _chapter_focus_hint(chapter_label: str) -> str:
    """Short focus line from chapter title after the dash."""
    label = (chapter_label or "").strip()
    if " - " in label:
        return label.split(" - ", 1)[1].strip()
    return label


def build_chapter_awareness_message(assessment: ChapterCoverageAssessment) -> str:
    """Student-facing message for not-covered topics with a/b/c choices."""
    topic = assessment.topic_label or "this topic"
    current = assessment.current_chapter_label
    focus = _chapter_focus_hint(current)

    if assessment.other_chapter_label:
        opener = (
            f"You're studying **{current}**, but **{topic}** is covered in "
            f"**{assessment.other_chapter_label}**, not in this chapter."
        )
        switch_line = (
            f"b) **Switch to {assessment.other_chapter_label}** — "
            f"open **Learning Studio** and select that chapter"
        )
    else:
        opener = (
            f"**{topic}** doesn't appear to be covered in **{current}**."
        )
        switch_line = (
            "b) **Switch to the relevant chapter** — "
            "open **Learning Studio** and select the chapter that covers this topic"
        )

    return (
        f"{opener}\n\n"
        f"**{current}** focuses on **{focus}**.\n\n"
        "How would you like to continue?\n"
        f"a) **Stay within the current chapter** — I'll help with topics from **{current}**\n"
        f"{switch_line}\n"
        f"c) **Receive a general explanation** — I'll explain **{topic}** beyond this chapter\n\n"
        "Reply with **a**, **b**, or **c**."
    )


def build_stay_in_chapter_message(current_chapter_label: str) -> str:
    return (
        f"Sure — let's stay with **{current_chapter_label}**.\n\n"
        f"Ask me any question about the topics covered in this chapter, "
        f"and I'll answer using the textbook first."
    )


def build_switch_chapter_message(
    *,
    other_chapter_label: str | None,
    current_chapter_label: str,
) -> str:
    if other_chapter_label:
        return (
            f"To learn about that topic from the textbook, open **Learning Studio**, "
            f"select **{other_chapter_label}**, and start a new session.\n\n"
            f"When you're ready, you can return to **{current_chapter_label}** anytime."
        )
    return (
        "Open **Learning Studio**, select the chapter that covers this topic, "
        "and start a new session there.\n\n"
        f"You can return to **{current_chapter_label}** whenever you like."
    )


def build_partial_coverage_guidance(assessment: ChapterCoverageAssessment) -> str:
    extra = ""
    if assessment.other_chapter_label:
        extra = (
            f" Additional details on **{assessment.topic_label}** appear in "
            f"**{assessment.other_chapter_label}**."
        )
    return (
        "CHAPTER COVERAGE ASSESSMENT: PARTIAL\n"
        f"The question is only partially covered by **{assessment.current_chapter_label}**.\n"
        "- Answer ONLY the portion supported by the chapter context below.\n"
        f"- Clearly note that the full explanation may appear elsewhere.{extra}\n"
        "- Do NOT give a complete off-chapter answer from general knowledge.\n"
        "- End by asking whether they want to continue here, switch chapter, or receive a full explanation."
    )


def build_general_explanation_guidance(topic_label: str) -> str:
    return (
        "CHAPTER COVERAGE: GENERAL EXPLANATION (student chose option c)\n"
        f"The student asked for a general explanation of **{topic_label}** beyond the current chapter.\n"
        "- Begin with **Beyond this chapter...** or **Additional context...**\n"
        "- You may use accurate general educational knowledge.\n"
        "- Briefly remind them which chapter they are studying, then answer clearly."
    )


_CHOICE_LETTER_RE = re.compile(
    r"(?:^|[\s,;])(?:i(?:'ll| will)?\s+)?"
    r"(?:go with|choose|pick|take|select|want|prefer|option|letter)\s*"
    r"(?:option\s*)?['\"]?\s*([abc])\b",
    re.I,
)


def _extract_awareness_choice_letter(query: str) -> str | None:
    """Return a/b/c when the student is picking from a chapter-awareness menu."""
    q = (query or "").strip()
    if not q:
        return None
    if _STAY_CHOICE_RE.match(q):
        return "a"
    if _SWITCH_CHOICE_RE.match(q):
        return "b"
    if _GENERAL_CHOICE_RE.match(q):
        return "c"
    m = _CHOICE_LETTER_RE.search(q)
    if m:
        return m.group(1).lower()
    if re.fullmatch(r"[abc]", q, re.I):
        return q.lower()
    return None


def detect_chapter_scope_choice(
    query: str,
    conversation_history: list[dict] | None,
) -> ChapterScopeChoice | None:
    """Parse a/b/c (or phrases) after a chapter-awareness prompt."""
    q = (query or "").strip()
    if not q or not conversation_history:
        return None
    last_asst = ""
    for turn in reversed(conversation_history):
        if (turn.get("role") or "").lower() == "assistant":
            last_asst = (turn.get("content") or "").strip()
            break
    if not is_chapter_awareness_prompt(last_asst):
        return None
    letter = _extract_awareness_choice_letter(q)
    if letter == "a":
        return ChapterScopeChoice.STAY
    if letter == "b":
        return ChapterScopeChoice.SWITCH
    if letter == "c":
        return ChapterScopeChoice.GENERAL
    return None


def original_question_before_awareness(
    conversation_history: list[dict] | None,
) -> str | None:
    """User question that triggered the chapter-awareness prompt."""
    if not conversation_history:
        return None
    seen_awareness = False
    for turn in reversed(conversation_history):
        role = (turn.get("role") or "").lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant" and is_chapter_awareness_prompt(content):
            seen_awareness = True
            continue
        if seen_awareness and role == "user":
            return content
    return None


def other_chapter_from_awareness_prompt(text: str) -> str | None:
    m = re.search(
        r"\*\*Switch to (.+?)\*\*\s*—\s*open \*\*Learning Studio\*\*",
        text or "",
        re.I,
    )
    if m:
        return m.group(1).strip()
    return None


def chapter_scope_mismatch_message(
    query: str,
    chapter_names: list[str] | None,
) -> str | None:
    """Redirect when the query explicitly names a chapter outside the selection."""
    names = [n.strip() for n in (chapter_names or []) if n and str(n).strip()]
    if not names:
        return None

    mentioned = extract_mentioned_chapter_numbers(query)
    if not mentioned:
        return None

    selected_nums = selected_chapter_numbers(names)
    if not selected_nums:
        return None

    out_of_scope = mentioned - selected_nums
    if not out_of_scope:
        return None

    current_label = _format_chapter_list(names)
    wrong_labels = [f"Chapter {n}" for n in sorted(out_of_scope)]
    wrong_label = wrong_labels[0] if len(wrong_labels) == 1 else ", ".join(wrong_labels)
    topic_nums = " ".join(f"chapter {n}" for n in sorted(out_of_scope))
    assessment = ChapterCoverageAssessment(
        level=ChapterCoverageLevel.NONE,
        topic_label=topic_nums,
        current_chapter_label=current_label,
        other_chapter_label=wrong_label,
    )
    return build_chapter_awareness_message(assessment)


def _should_skip_topic_scope_check(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return True
    if len(q.split()) <= 2 and _SKIP_TOPIC_SCOPE_RE.match(q):
        return True
    return bool(_SKIP_TOPIC_SCOPE_RE.match(q))


def _term_hits_in_text(terms: set[str], text: str) -> int:
    if not terms:
        return 0
    lowered = (text or "").lower()
    return sum(1 for term in terms if term in lowered)


def _docs_with_all_terms(terms: set[str], docs: list) -> int:
    if not terms or not docs:
        return 0
    count = 0
    for doc in docs[:10]:
        if _term_hits_in_text(terms, getattr(doc, "page_content", "") or "") == len(terms):
            count += 1
    return count


def _max_term_hits(terms: set[str], docs: list) -> int:
    if not terms or not docs:
        return 0
    return max(
        _term_hits_in_text(terms, getattr(doc, "page_content", "") or "")
        for doc in docs[:10]
    )


def _upload_id_from_doc(doc) -> str:
    meta = getattr(doc, "metadata", None) or {}
    return str(meta.get("textbook_upload_id") or "")


@lru_cache(maxsize=64)
def _subject_upload_labels(board: str, class_level: str, subject_name: str) -> dict[str, str]:
    """Map textbook upload IDs to human chapter labels for a subject."""
    try:
        from sqlalchemy import select

        from app.core.database import SessionLocal
        from app.modules.catalog.models import BoardEnum, ClassEnum, ProcessingStatusEnum, TextbookUpload

        board_enum = BoardEnum(board)
        class_enum = ClassEnum(class_level)
    except ValueError:
        return {}

    try:
        with SessionLocal() as db:
            rows = list(
                db.scalars(
                    select(TextbookUpload)
                    .where(
                        TextbookUpload.board == board_enum,
                        TextbookUpload.class_level == class_enum,
                        TextbookUpload.subject_name == subject_name,
                        TextbookUpload.embedding_status == ProcessingStatusEnum.EMBEDDED,
                    )
                    .order_by(TextbookUpload.chapter)
                )
            )
    except Exception as exc:
        logger.warning("Failed to load subject uploads for topic scope: %s", exc)
        return {}

    labels: dict[str, str] = {}
    for row in rows:
        label = (row.chapter or "").strip() or row.file_name
        labels[str(row.id)] = label
    return labels


def _best_other_chapter(
    terms: set[str],
    *,
    collection_name: str,
    query: str,
    selected_set: set[str],
) -> tuple[str, int, int]:
    from app.services.vector_service import retrieve_from_collection

    cross_docs = retrieve_from_collection(
        query,
        collection_name=collection_name,
        chapter_ids=None,
        k=15,
    )
    per_upload: dict[str, list] = {}
    for doc in cross_docs:
        uid = _upload_id_from_doc(doc)
        if not uid or uid in selected_set:
            continue
        per_upload.setdefault(uid, []).append(doc)

    best_id = ""
    best_hits = 0
    best_coverage = 0
    for uid, upload_docs in per_upload.items():
        hits = _max_term_hits(terms, upload_docs)
        coverage = _docs_with_all_terms(terms, upload_docs)
        if hits > best_hits or (hits == best_hits and coverage > best_coverage):
            best_hits = hits
            best_coverage = coverage
            best_id = uid
    return best_id, best_hits, best_coverage


def assess_chapter_coverage(
    query: str,
    *,
    docs: list,
    collection_name: str,
    chapter_ids: list[str] | None,
    chapter_names: list[str] | None,
    board: str,
    class_level: str,
    subject_name: str,
) -> ChapterCoverageAssessment:
    """
  Assess whether the selected chapter fully, partially, or does not cover the question.
    """
    current_label = _current_chapter_label(chapter_names)
    terms = substantive_query_terms(query)
    topic_label = " ".join(sorted(terms)) if terms else (query or "").strip()[:80]

    if not chapter_ids:
        return ChapterCoverageAssessment(
            level=ChapterCoverageLevel.FULL,
            topic_label=topic_label,
            current_chapter_label=current_label,
        )
    if _should_skip_topic_scope_check(query):
        return ChapterCoverageAssessment(
            level=ChapterCoverageLevel.FULL,
            topic_label=topic_label,
            current_chapter_label=current_label,
        )
    if not terms:
        return ChapterCoverageAssessment(
            level=ChapterCoverageLevel.FULL,
            topic_label=topic_label,
            current_chapter_label=current_label,
        )

    selected_hits = _max_term_hits(terms, docs)
    selected_coverage = _docs_with_all_terms(terms, docs)
    required_hits = max(1, len(terms))
    selected_set = {str(cid) for cid in chapter_ids}

    other_id, other_hits, other_coverage = _best_other_chapter(
        terms,
        collection_name=collection_name,
        query=query,
        selected_set=selected_set,
    )
    labels = _subject_upload_labels(board, class_level, subject_name)
    other_label = labels.get(other_id) if other_id else None

    base = ChapterCoverageAssessment(
        level=ChapterCoverageLevel.FULL,
        topic_label=topic_label,
        current_chapter_label=current_label,
        other_chapter_label=other_label,
        other_chapter_id=other_id or None,
        selected_hits=selected_hits,
        required_hits=required_hits,
        selected_coverage=selected_coverage,
    )

    # Strong coverage in selected chapter
    if selected_hits >= required_hits and selected_coverage >= 1:
        if other_hits > selected_hits and other_coverage >= 1:
            base.level = ChapterCoverageLevel.PARTIAL
            return base
        base.level = ChapterCoverageLevel.FULL
        return base

    # Another chapter clearly owns the topic
    if other_id and other_hits >= required_hits and other_hits > selected_hits:
        base.level = ChapterCoverageLevel.NONE
        return base

    # Weak or no match in selected chapter
    if selected_hits == 0:
        base.level = ChapterCoverageLevel.NONE
        return base

    if selected_hits < required_hits:
        base.level = ChapterCoverageLevel.PARTIAL
        return base

    base.level = ChapterCoverageLevel.FULL
    return base


def topic_chapter_mismatch_message(
    query: str,
    *,
    docs: list,
    collection_name: str,
    chapter_ids: list[str] | None,
    chapter_names: list[str] | None,
    board: str,
    class_level: str,
    subject_name: str,
) -> str | None:
    """Legacy entry point — returns awareness message when coverage is NONE."""
    assessment = assess_chapter_coverage(
        query,
        docs=docs,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
    )
    if assessment.level == ChapterCoverageLevel.NONE:
        return build_chapter_awareness_message(assessment)
    return None


def resolve_chapter_scope_message(
    query: str,
    *,
    docs: list | None = None,
    collection_name: str = "",
    chapter_ids: list[str] | None = None,
    chapter_names: list[str] | None = None,
    board: str = "",
    class_level: str = "",
    subject_name: str = "",
) -> str | None:
    """Run explicit and topic-based chapter scope checks."""
    explicit = chapter_scope_mismatch_message(query, chapter_names)
    if explicit:
        return explicit
    if docs is None:
        return None
    return topic_chapter_mismatch_message(
        query,
        docs=docs,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
    )


def resolve_chapter_awareness_turn(
    query: str,
    *,
    docs: list,
    conversation_history: list[dict] | None,
    collection_name: str,
    chapter_ids: list[str] | None,
    chapter_names: list[str] | None,
    board: str,
    class_level: str,
    subject_name: str,
) -> tuple[str | None, str, ChapterCoverageAssessment | None, str]:
    """
    Handle chapter-awareness for one student turn.

    Returns:
        early_response — if set, return this to the student (no LLM call)
        effective_query — query to answer (may be the original question after option c)
        assessment — coverage assessment when applicable
        coverage_guidance — extra system-prompt block for the LLM
    """
    explicit = chapter_scope_mismatch_message(query, chapter_names)
    if explicit:
        return explicit, query, None, ""

    choice = detect_chapter_scope_choice(query, conversation_history)
    current_label = _current_chapter_label(chapter_names)

    if choice == ChapterScopeChoice.STAY:
        return build_stay_in_chapter_message(current_label), query, None, ""

    if choice == ChapterScopeChoice.SWITCH:
        last_asst = ""
        if conversation_history:
            for turn in reversed(conversation_history):
                if (turn.get("role") or "").lower() == "assistant":
                    last_asst = turn.get("content") or ""
                    break
        other = other_chapter_from_awareness_prompt(last_asst)
        return (
            build_switch_chapter_message(
                other_chapter_label=other,
                current_chapter_label=current_label,
            ),
            query,
            None,
            "",
        )

    effective_query = query
    coverage_guidance = ""

    if choice == ChapterScopeChoice.GENERAL:
        original = original_question_before_awareness(conversation_history)
        if original:
            effective_query = original
            terms = substantive_query_terms(original)
            topic = " ".join(sorted(terms)) if terms else original
            coverage_guidance = build_general_explanation_guidance(topic)
            return None, effective_query, None, coverage_guidance

    assessment = assess_chapter_coverage(
        effective_query,
        docs=docs,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
    )

    if assessment.level == ChapterCoverageLevel.NONE:
        return build_chapter_awareness_message(assessment), effective_query, assessment, ""

    if assessment.level == ChapterCoverageLevel.PARTIAL:
        coverage_guidance = build_partial_coverage_guidance(assessment)

    return None, effective_query, assessment, coverage_guidance


def resolve_chapter_scope_with_retrieval(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None,
    chapter_names: list[str] | None,
    board: str,
    class_level: str,
    subject_name: str,
    retrieval_query: str | None = None,
) -> str | None:
    """Explicit + topic scope checks using a fresh retrieval pass."""
    explicit = chapter_scope_mismatch_message(query, chapter_names)
    if explicit:
        return explicit
    if not chapter_ids:
        return None

    from app.services.section_retrieval import retrieve_for_tutor_query

    docs, _, _ = retrieve_for_tutor_query(
        retrieval_query or query,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
    )
    return topic_chapter_mismatch_message(
        query,
        docs=docs,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
    )
