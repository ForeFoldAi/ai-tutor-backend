"""
Detect when a student's question references a chapter outside the current session scope.
"""

from __future__ import annotations

import logging
import re
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


def _format_chapter_list(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]}, {names[1]}, and {len(names) - 2} more"


def chapter_scope_mismatch_message(
    query: str,
    chapter_names: list[str] | None,
) -> str | None:
    """
    Return a friendly redirect when the query names a chapter not in the selection.
    """
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
    if len(wrong_labels) == 1:
        wrong_label = wrong_labels[0]
    else:
        wrong_label = ", ".join(wrong_labels[:-1]) + f", and {wrong_labels[-1]}"

    return (
        f"You're currently studying **{current_label}**, but your question is about **{wrong_label}**.\n\n"
        f"To get accurate answers from the textbook, go back to **Learning Studio**, select **{wrong_label}**, "
        f"and start a new session — or ask a question about **{current_label}** instead."
    )


def _should_skip_topic_scope_check(query: str) -> bool:
    q = (query or "").strip()
    if not q or len(q.split()) <= 2 and _SKIP_TOPIC_SCOPE_RE.match(q):
        return True
    if _SKIP_TOPIC_SCOPE_RE.match(q):
        return True
    return False


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
    """
    Detect when a topic clearly belongs to another chapter in the same subject
    (e.g. 'what is desert?' while studying Chapter 2 - Weather).
    """
    if not chapter_ids or len(chapter_ids) != 1:
        return None
    if _should_skip_topic_scope_check(query):
        return None

    terms = substantive_query_terms(query)
    if not terms:
        return None

    selected_coverage = _docs_with_all_terms(terms, docs)
    if selected_coverage >= 2:
        return None

    selected_hits = 0
    if docs:
        selected_hits = max(
            _term_hits_in_text(terms, getattr(doc, "page_content", "") or "")
            for doc in docs[:8]
        )

    required_hits = max(1, len(terms))
    if selected_hits >= required_hits and selected_coverage >= 1:
        return None

    from app.services.vector_service import retrieve_from_collection

    selected_set = {str(cid) for cid in chapter_ids}
    cross_docs = retrieve_from_collection(
        query,
        collection_name=collection_name,
        chapter_ids=None,
        k=15,
    )

    best_other_id = ""
    best_other_hits = 0
    best_other_coverage = 0
    per_upload_docs: dict[str, list] = {}
    for doc in cross_docs:
        uid = _upload_id_from_doc(doc)
        if not uid or uid in selected_set:
            continue
        per_upload_docs.setdefault(uid, []).append(doc)

    for uid, upload_docs in per_upload_docs.items():
        hits = max(
            _term_hits_in_text(terms, getattr(doc, "page_content", "") or "")
            for doc in upload_docs
        )
        coverage = _docs_with_all_terms(terms, upload_docs)
        if hits > best_other_hits or (hits == best_other_hits and coverage > best_other_coverage):
            best_other_hits = hits
            best_other_coverage = coverage
            best_other_id = uid

    if not best_other_id:
        return None

    if best_other_hits < required_hits:
        return None
    if best_other_hits <= selected_hits:
        return None
    if best_other_coverage < 1 and selected_coverage >= 1:
        return None

    labels = _subject_upload_labels(board, class_level, subject_name)
    other_label = labels.get(best_other_id) or "the correct chapter"

    current_names = [n.strip() for n in (chapter_names or []) if n and str(n).strip()]
    current_label = current_names[0] if current_names else "your selected chapter"
    topic_label = " ".join(sorted(terms))

    return (
        f"You're studying **{current_label}**, but **{topic_label}** is covered in **{other_label}**, "
        f"not in this chapter.\n\n"
        f"Go back to **Learning Studio**, select **{other_label}**, and ask again — "
        f"or ask a question about **{current_label}**."
    )


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
