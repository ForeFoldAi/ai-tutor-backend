"""Student-scoped general assistant (platform, profile, subjects, progress, quick-start modes)."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.modules.student_assistant.context import build_student_context, build_suggested_prompts
from app.modules.student_learning import service as learning_service
from app.modules.users.models import User
from app.config import ASSISTANT_MAX_TOKENS, ASSISTANT_RAG_CHUNK_CHARS, ASSISTANT_RAG_CHUNKS
from app.core.student_messages import ANSWER_NOT_IN_CHAPTER, CHAT_ANSWER_FAILED
from app.services.chapter_scope import extract_mentioned_chapter_numbers
from app.services.chat_service import (
    _call_mistral_async,
    _prepare_math_engine_block,
    _stream_mistral_async,
)
from app.services.conversation_context import resolve_conversation_context
from app.services.conversation_memory import format_memory_for_prompt, prepare_conversation_inputs

logger = logging.getLogger(__name__)

_AGENT_MODES = frozenset({"free", "ask", "practice", "explain"})

_CONTINUE_USER = (
    "Continue exactly where you left off. Finish the answer completely. "
    "Do not repeat what you already wrote. Keep it short."
)

# Shared by text chat and voice (same student_assistant stream).
_LENGTH_STYLE = """
Length and style (text and voice — same rules):
- Keep every reply SMALL or MEDIUM only — about 40 to 120 words.
- Use simple words a school student understands. Short sentences.
- At most 3 short steps or bullets. No long lectures, essays, or long methodology.
- One small example is enough. Stop when the question is answered.
- Always finish completely — never stop mid-sentence, mid-bullet, or mid-step.
- Never use markdown formatting.
- Never output the '*' character (asterisk). For multiplication write × (example: 2 × 3 × 4 = 24).
- If the student questions, quotes, or doubts something from your previous reply (a number, a step,
  a word), it refers to that reply: say plainly what it should be, correct any slip, then finish
  the original example. Never treat it as a brand-new topic.
"""

_FREE_PROMPT = """You are the AI Tutor assistant for {student_name}, a student on the AI Tutor learning platform.

This is the open "ask anything" assistant. You may answer:
- Platform / account help: the student (name, grade, class, progress, streak, recent lessons), enrolled subjects, school/tutor (if present), and how to use AI Tutor features
- ANY learning or general-knowledge question: school subjects, topics outside their textbook or syllabus,
  and skills like programming (Python, Java, web), computers, languages, science, history, current general knowledge
- Homework help, explanations, study tips, and everyday questions

Never refuse a learning topic:
- Enrolled subjects, board and grade only tell you how simple to make the answer. They are NOT a list of allowed topics.
- If the student asks to learn or be taught something ("teach me Python", "explain coding", "what is AI"),
  start teaching it right away: what it is in one line, one tiny example, then offer the next step.
- Never reply that you "do not have lessons" on it, "do not teach" it, that it is not in their curriculum,
  that they should ask a school admin or tutor, or that they should study one of their chapters instead.
- Only decline content that is unsafe or not appropriate for a school student.

Student facts (enrollment, progress, school, tutor):
- Use ONLY STUDENT CONTEXT below. Never invent subjects, grades, teachers, or progress not listed there.
- Only when the student asks a personal/platform fact (their marks, their teacher, their schedule) that the
  context lacks: say you do not have that information and suggest who to ask (tutor, school admin) or where in the app to look.
  Never use this rule for learning or general-knowledge questions.

Academic / out-of-book help:
- Use general knowledge, pitched at their grade. Prefer clear steps and simple examples.
- For code, show short lines of code as plain text (no code fences), each line on its own line.
- When TEXTBOOK CONTEXT is present and relevant, prefer it; if the question is outside the textbook, still answer helpfully (this mode allows out-of-book).
- Never add source labels such as "Beyond your textbook:", "Beyond this chapter" or "Additional context:" — just answer naturally.
- For deeper chapter-locked tutoring, you may mention Chapter AI Tutor or AI Voice after picking a subject/chapter.

Rules:
- Never reveal other students' data or admin-only information.
- Keep answers clear, friendly, and age-appropriate.
""" + _LENGTH_STYLE + """
STUDENT CONTEXT:
{context}
{detected}
{textbook}
"""

_ASK_PROMPT = """You are the AI Tutor for {student_name}. Mode: Ask Anything.

Purpose: Give a SHORT, direct answer to the student's question (Q&A style).
Do NOT write a full lesson, long step-by-step lecture, or a practice quiz unless they ask.

Textbook rule:
- Answer using TEXTBOOK CONTEXT below (and detected subject/chapter).
- This mode covers ANY enrolled textbook chapter, not only the last one.
- If the student asks about another chapter — by number, title, or subject + chapter (e.g. "explain me social chapter 2") — answer that chapter. They do not need to say "switch".
- Other enrolled chapters are still textbook topics. Do NOT call them out-of-book. Do NOT send the student to Ask AI Tutor just because they named a different chapter.
- DETECTED SCOPE follows their latest textbook question; do not refuse because it differs from an earlier chapter.
- Informal wording and typos are fine (e.g. "explain me", "Delhi Sultane") — infer the intended textbook topic.
- If TEXTBOOK CONTEXT is missing, empty, or the question is not a textbook topic (platform, account, general knowledge with no chapter), do NOT invent an answer from general knowledge.
- Only then reply briefly: this Quick Start mode is for textbook topics only, and ask them to open Ask AI Tutor for general / platform / out-of-book questions.

Detect subject/topic from the question when possible. Do not ask them to pick a subject or chapter first.

Rules:
- Age-appropriate for their grade.
- Never invent enrollment data.
""" + _LENGTH_STYLE + """
STUDENT CONTEXT:
{context}
{detected}
{textbook}
"""

_TEXTBOOK_MODES = frozenset({"ask", "practice", "explain"})

_PRACTICE_PROMPT = """You are the AI Tutor for {student_name}. Mode: Practice Problems.

Purpose: Act as a practice coach for textbook topics.
Do NOT dump a long explanation or open lecture.

When the student asks a concrete mathematical question / pastes a problem:
1. Answer the exact question (same numbers, conditions, quantities — never substitute a different question).
2. Explain briefly how the answer was obtained.
3. Show step-by-step mathematical reasoning when appropriate (keep it short).
4. Then ask ONE short follow-up question (check understanding or a related practice step). Do NOT lead with a different problem instead of answering.
5. Do NOT restate or repost their question.

When they only name a topic and ask to practice (no concrete problem yet):
- Give ONE practice problem from TEXTBOOK CONTEXT and wait for their attempt.
- After they answer, give brief feedback; give another problem only if they want more practice.

If they only ask a theory/topic question with no request to practice (e.g. "what is a fraction?"), answer briefly — do not invent practice problems unless they ask.

Textbook rule (STRICT):
- Problems must be grounded in TEXTBOOK CONTEXT below (and detected subject/chapter).
- If TEXTBOOK CONTEXT is missing or the request is outside the textbook, do NOT invent problems from general knowledge.
- In that case reply briefly: this mode is for textbook practice only, and ask them to open Ask AI Tutor for general / platform / out-of-book help.

Chapter stickiness:
- Stay on the SAME detected chapter for the whole conversation unless the student clearly names a different chapter.
- Treat likely typos (e.g. "preposition" when studying "disposition") as the current chapter — never switch to unrelated subjects like English grammar.
- Every problem must test material from TEXTBOOK CONTEXT, not general knowledge.

Detect subject/topic from the request when possible. Do not ask them to pick a subject or chapter first.

Rules:
- Match difficulty to their grade.
- Prefer short problems; one at a time when generating practice.
""" + _LENGTH_STYLE + """
STUDENT CONTEXT:
{context}
{detected}
{textbook}
"""

_EXPLAIN_PROMPT = """You are the AI Tutor for {student_name}. Mode: Explain a Topic.

Purpose: Give a short, simple mini-explanation of a textbook topic (not a full lecture).
Do NOT start a quiz or practice problems unless the student explicitly asks.
Do NOT give only a one-line definition — use a few short steps and one simple example when useful.

Textbook rule (STRICT):
- Explain ONLY using TEXTBOOK CONTEXT below (and detected subject/chapter).
- If TEXTBOOK CONTEXT is missing or the topic is outside the textbook, do NOT invent a lesson from general knowledge.
- In that case reply briefly: this mode is for textbook explanations only, and ask them to open Ask AI Tutor for general / platform / out-of-book questions.

Detect subject/topic from the request when possible. Do not ask them to pick a subject or chapter first.

Rules:
- Use short steps and simple wording for their grade.
""" + _LENGTH_STYLE + """
STUDENT CONTEXT:
{context}
{detected}
{textbook}
"""

_MODE_PROMPTS = {
    "free": _FREE_PROMPT,
    "ask": _ASK_PROMPT,
    "practice": _PRACTICE_PROMPT,
    "explain": _EXPLAIN_PROMPT,
}


def normalize_agent_mode(mode: str | None) -> str:
    m = (mode or "free").strip().lower()
    return m if m in _AGENT_MODES else "free"


def _looks_incomplete(text: str) -> bool:
    """True when the model likely stopped mid-answer (token cut / stream drop)."""
    t = (text or "").rstrip()
    if len(t) < 40:
        return False
    if t.endswith((":", "—", "–", "-", ",", ";", "/", "(")):
        return True
    last = t.rsplit("\n", 1)[-1].strip()
    if re.match(r"^(?:[-•]|\d+[.)])\s+\S+", last) and not re.search(r"[.!?…][\"')\]]*$", last):
        return True
    if re.search(r"\b(?:step|example|for example|like this|as follows)\s*\d*\s*:?\s*$", t, re.I):
        return True
    return False


def _with_continue(messages: list[dict[str, str]], partial: str) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": partial},
        {"role": "user", "content": _CONTINUE_USER},
    ]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


# Enough of the continuation to tell a real continuation from a restart.
_CONTINUE_PROBE_CHARS = 80


def _continuation_tail(partial: str, more: str) -> str | None:
    """Text to append after *partial*, or None when *more* restarts the answer from scratch."""
    more = (more or "").lstrip()
    if not more:
        return ""
    head = _squash(more[:40])
    if len(head) >= 30 and head in _squash(partial):
        return None
    # Drop an echoed overlap ("...write them vertically:" -> "Write them vertically: 12 + 5").
    tail = partial.rstrip()
    for k in range(min(len(tail), len(more)), 9, -1):
        if _squash(tail[-k:]) == _squash(more[:k]):
            more = more[k:].lstrip()
            break
    if not more:
        return ""
    sep = " " if tail.endswith((",", ";", "—", "–", "-", "/", "(")) else "\n"
    return sep + more


def _fallback_text(mode: str, textbook: str) -> str:
    if mode in _TEXTBOOK_MODES and not textbook.strip():
        return ANSWER_NOT_IN_CHAPTER
    return CHAT_ANSWER_FAILED


def _first_name(full_name: str | None) -> str:
    name = (full_name or "").strip()
    if not name:
        return "there"
    return name.split()[0]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# Generic words ignored when matching query ↔ chapter/subject titles.
# len>=3 content words like "wit" must match (old >3 skipped them).
_SCOPE_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "how",
        "why",
        "who",
        "are",
        "was",
        "were",
        "about",
        "into",
        "unit",
        "chapter",
        "lesson",
        "part",
        "know",
        "important",
        "quick",
        "question",
        "explain",
        "practice",
    }
)


def _scope_tokens(s: str) -> list[str]:
    return [t for t in _norm(s).split() if len(t) >= 3 and t not in _SCOPE_STOP]


def _stem_forms(token: str) -> set[str]:
    """Tiny inflection set so nature↔natural and resource↔resources can match."""
    t = (token or "").lower()
    if len(t) < 3:
        return {t} if t else set()
    forms = {t}
    if len(t) >= 5:
        forms.add(t[:5])
    if t.endswith("ies") and len(t) > 4:
        forms.add(t[:-3] + "y")
    if t.endswith("es") and len(t) > 4:
        forms.add(t[:-2])
    if t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
        forms.add(t[:-1])
    if t.endswith("al") and len(t) > 5:
        forms.add(t[:-2])
    if t.endswith("e") and len(t) > 4:
        forms.add(t[:-1])
    return forms


def _token_in(token: str, haystack: str) -> bool:
    """True if token appears in haystack, allowing light stem overlap."""
    if not token or not haystack:
        return False
    if token in haystack:
        return True
    # ponytail: O(words) stem overlap; ceiling = shared 5-letter prefixes colliding
    # across chapters — upgrade to a real stemmer if false chapter picks rise.
    h_forms: set[str] = set()
    for w in haystack.split():
        if len(w) >= 3:
            h_forms |= _stem_forms(w)
    return bool(_stem_forms(token) & h_forms)


def _scope_query_text(query: str, conversation_history: list[dict[str, str]] | None) -> str:
    """Current question plus recent user turns (follow-ups inherit chapter names)."""
    parts = [query or ""]
    if conversation_history:
        for turn in conversation_history[-6:]:
            if (turn.get("role") or "").strip() != "user":
                continue
            content = (turn.get("content") or "").strip()
            if content:
                parts.append(content)
    return _norm(" ".join(parts))


def _resolve_turn_context(
    query: str,
    conversation_history: list[dict[str, str]] | None,
    *,
    chapter: str = "",
):
    recent_hist, mem = prepare_conversation_inputs(
        conversation_history,
        chapter=chapter,
    )
    conv = resolve_conversation_context(
        query,
        conversation_history=recent_hist,
        chapter=chapter or None,
        memory=mem,
    )
    return conv, recent_hist, mem


# Not "unit": NCERT units group several chapters, so "Unit 2" is not "chapter 2".
_TITLE_NUM_RE = re.compile(r"\b(?:chapter|lesson)\s*(\d+)", re.I)
# NCERT file codes: hepr103.pdf = class h, English p·r book 1, chapter 03.
_NCERT_CODE_RE = re.compile(r"(?:^|[^a-z])[a-z]{4}\d(\d{2})\.pdf$", re.I)


def _title_number(title: str, file_name: str = "") -> int | None:
    """Chapter number from 'Chapter 2 - …', else from the NCERT file code."""
    m = _TITLE_NUM_RE.search(title or "") or _NCERT_CODE_RE.search(file_name or "")
    return int(m.group(1)) if m else None


def _title_match_score(q: str, title: str, file_name: str = "") -> int:
    """Score how well a normalized query matches a subject/chapter title."""
    label = _norm(title)
    if not label or not q:
        return 0
    if label in q:
        return 8
    score = sum(2 for token in _scope_tokens(label) if _token_in(token, q))
    # "chapter 2" is an explicit pick — same weight as a full title hit
    t_num = _title_number(title, file_name)
    if t_num is not None and t_num in extract_mentioned_chapter_numbers(q):
        score += 8
    return score


def _named_subjects(subjects, q: str) -> list:
    """Subjects the query names outright ("chapter 1 in english"); all subjects when none is named."""
    padded = f" {q} "
    named = [s for s in subjects if _norm(s.subject_name) and f" {_norm(s.subject_name)} " in padded]
    return named or list(subjects)


def _scope_dict(subject, chapter, score: int) -> dict:
    if score < 2 or subject is None:
        return {}
    out: dict = {
        "subject_name": subject.subject_name,
        "board": getattr(subject.board, "value", str(subject.board)),
        "class_level": getattr(subject.class_level, "value", str(subject.class_level)),
    }
    if chapter is not None:
        out["chapter_id"] = chapter.id
        out["chapter_name"] = chapter.chapter or chapter.file_name
    return out


def _best_learning_match(overview, q: str) -> tuple[int, object | None, object | None]:
    """Return (score, best_subject, best_chapter|None) for normalized query text."""
    if not q or not overview.subjects:
        return 0, None, None

    best_subject = None
    best_chapter = None
    best_score = 0

    for subj in _named_subjects(overview.subjects, q):
        subj_name = _norm(subj.subject_name)
        score = 0
        if subj_name and subj_name in q:
            score += 5
        score += sum(1 for token in _scope_tokens(subj_name) if _token_in(token, q))
        for ch in subj.chapters:
            label = ch.chapter or ch.file_name or ""
            title_score = _title_match_score(q, label, ch.file_name or "")
            # Naming only the subject must not lock onto whichever chapter sorts first.
            if title_score == 0:
                continue
            ch_score = score + title_score
            if ch_score > best_score:
                best_score = ch_score
                best_subject = subj
                best_chapter = ch

        if score > best_score:
            best_score = score
            best_subject = subj
            best_chapter = None

    return best_score, best_subject, best_chapter


def _chapter_from_doc_meta(subj, meta: dict):
    """Map retrieval metadata back to an enrolled chapter on *subj*."""
    uid = str(meta.get("textbook_upload_id") or meta.get("chapter_id") or "").strip()
    if uid:
        for ch in subj.chapters:
            if str(ch.id) == uid:
                return ch
    label = _norm(meta.get("content_label") or meta.get("chapter") or "")
    if not label:
        return None
    for ch in subj.chapters:
        ch_label = _norm(ch.chapter or ch.file_name or "")
        if ch_label and (ch_label == label or label in ch_label or ch_label in label):
            return ch
    return None


def _scope_from_enrolled_content(overview, q: str) -> dict:
    """
    When title match fails, search enrolled subject collections and pick the
    chapter whose chunks best match the question (any subject / Quick Start mode).
    """
    if not q or not overview.subjects:
        return {}
    try:
        from app.services.query_match import keyword_match_score
        from app.services.vector_service import retrieve_from_collection
    except Exception:
        return {}

    best_score = 0
    best_subj = None
    best_ch = None

    def _search(subj):
        board = getattr(subj.board, "value", str(subj.board))
        class_level = getattr(subj.class_level, "value", str(subj.class_level))
        collection = f"{board}_{class_level}_{subj.subject_name}".replace(" ", "_")
        try:
            return retrieve_from_collection(q, collection_name=collection, chapter_ids=None, k=6)
        except Exception:
            logger.warning("assistant scope search failed for %s", collection, exc_info=True)
            return []

    subjects = [s for s in _named_subjects(overview.subjects, q) if s.chapters]
    if not subjects:
        return {}
    with ThreadPoolExecutor(max_workers=min(4, len(subjects))) as pool:
        results = list(pool.map(_search, subjects))

    for subj, docs in zip(subjects, results):
        for doc in docs or []:
            meta = getattr(doc, "metadata", None) or {}
            ch = _chapter_from_doc_meta(subj, meta)
            if ch is None:
                continue
            text = getattr(doc, "page_content", None) or str(doc) or ""
            score = keyword_match_score(q, text)
            score += _title_match_score(q, ch.chapter or ch.file_name or "", ch.file_name or "")
            if score > best_score:
                best_score = score
                best_subj = subj
                best_ch = ch

    # ponytail: need at least one real keyword hit so empty/noisy collections don't pick randomly
    if best_score < 2 or best_subj is None or best_ch is None:
        return {}
    return _scope_dict(best_subj, best_ch, best_score)


def detect_learning_scope(
    db: Session,
    user: User,
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
    *,
    agent_mode: str = "free",
    overview=None,
    turn: tuple | None = None,
) -> dict:
    """Match enrolled subject/chapter from the free-text prompt (no picker required).

    *overview* and *turn* (from ``_resolve_turn_context``) let the caller share work
    it already did this turn.
    """
    if overview is None:
        overview = learning_service.get_overview(db, user)
    if not overview.subjects:
        return {}

    mode = normalize_agent_mode(agent_mode)
    conv, recent_hist, mem = turn or _resolve_turn_context(query, conversation_history)
    q_resolved = conv.retrieval_query or query
    q_full = _scope_query_text(q_resolved, recent_hist)
    full_score, full_subj, full_ch = _best_learning_match(overview, q_full)

    # ponytail: practice/ask/explain follow-ups keep the chapter from earlier turns
    # so a typo like "preposition" does not jump to an English grammar chapter.
    if mode in _TEXTBOOK_MODES and recent_hist:
        q_hist = _scope_query_text("", recent_hist)
        if mem.topics_discussed:
            q_hist = _norm(f"{q_hist} {' '.join(mem.topics_discussed[:4])}")
        hist_score, hist_subj, hist_ch = _best_learning_match(overview, q_hist)
        if hist_ch is not None and hist_score >= 4:
            q_cur = _norm(query or "")
            cur_score, cur_subj, cur_ch = _best_learning_match(overview, q_cur)
            cur_label = (cur_ch.chapter or cur_ch.file_name or "") if cur_ch else ""
            explicit_switch = (
                cur_ch is not None
                and cur_ch.id != hist_ch.id
                and (
                    cur_score >= hist_score + 2
                    or _title_match_score(q_cur, cur_label, cur_ch.file_name or "") >= 8
                )
            )
            if explicit_switch:
                return _scope_dict(cur_subj, cur_ch, cur_score)
            named_now = _named_subjects(overview.subjects, q_cur)
            if len(named_now) < len(overview.subjects) and hist_subj not in named_now:
                # They named another subject ("now chapter 1 in english") — leave the old chapter.
                q_full = q_cur
                full_score, full_subj, full_ch = cur_score, cur_subj, cur_ch
            else:
                return _scope_dict(hist_subj, hist_ch, hist_score)

    scoped = _scope_dict(full_subj, full_ch, full_score)
    # If title match missed (any subject), fall back to content search across enrolled
    # textbooks so section headings still resolve — free mode too, so its academic answers
    # stay grounded in the student's book when it covers the topic.
    # A "chapter N" that matched nothing stays unresolved: guessing another chapter
    # is how "chapter 1 in english" got answered from Science.
    if (
        not scoped.get("chapter_id")
        and not extract_mentioned_chapter_numbers(q_full)
    ):
        content_scope = _scope_from_enrolled_content(overview, q_full)
        if content_scope.get("chapter_id"):
            return content_scope
    return scoped


def _format_detected(scope: dict) -> str:
    if not scope:
        return "DETECTED SCOPE:\n- None yet — infer topic from the student question and enrolled subjects."
    lines = ["DETECTED SCOPE:"]
    if scope.get("subject_name"):
        lines.append(f"- Subject: {scope['subject_name']}")
    if scope.get("chapter_name"):
        lines.append(f"- Chapter: {scope['chapter_name']}")
    if scope.get("board") or scope.get("class_level"):
        lines.append(f"- Board/class: {scope.get('board', '')} {scope.get('class_level', '')}".strip())
    return "\n".join(lines)


def _textbook_snippets(
    scope: dict,
    query: str,
    *,
    agent_mode: str = "ask",
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Best-effort RAG snippets from the detected chapter; fail open."""
    chapter_id = scope.get("chapter_id")
    board = scope.get("board") or ""
    class_level = scope.get("class_level") or ""
    subject_name = scope.get("subject_name") or ""
    if not chapter_id or not subject_name:
        return ""
    mode = normalize_agent_mode(agent_mode)
    conv, recent_hist, mem = _resolve_turn_context(
        query,
        conversation_history,
        chapter=(scope.get("chapter_name") or ""),
    )
    rag_query = conv.retrieval_query or query
    if mode in _TEXTBOOK_MODES:
        chapter = (scope.get("chapter_name") or "").strip()
        context_q = _scope_query_text(rag_query, recent_hist)
        rag_query = f"{chapter} {context_q}".strip() if chapter else context_q
    try:
        from app.services.section_retrieval import retrieve_for_tutor_query

        collection = f"{board}_{class_level}_{subject_name}".replace(" ", "_")
        docs, _scope, _instr = retrieve_for_tutor_query(
            rag_query,
            collection_name=collection,
            chapter_ids=[str(chapter_id)],
            chapter_names=[scope.get("chapter_name") or ""],
        )
        if not docs:
            return ""
        bits: list[str] = []
        for doc in docs[:ASSISTANT_RAG_CHUNKS]:
            text = (getattr(doc, "page_content", None) or str(doc) or "").strip()
            if text:
                bits.append(text[:ASSISTANT_RAG_CHUNK_CHARS])
        if not bits:
            return ""
        if mode == "free":
            header = (
                "TEXTBOOK CONTEXT (prefer when relevant; still answer if the question "
                "is outside this material):\n"
            )
        else:
            header = (
                "TEXTBOOK CONTEXT (use ONLY this material; if missing, redirect to Ask AI Tutor):\n"
            )
        return header + "\n---\n".join(bits)
    except Exception:
        logger.warning(
            "assistant textbook retrieval failed (chapter=%s, subject=%s)",
            chapter_id,
            subject_name,
            exc_info=True,
        )
        return ""


def _build_messages(
    *,
    context: str,
    student_name: str,
    query: str,
    conversation_history: list[dict[str, str]] | None,
    agent_mode: str,
    detected: str,
    textbook: str,
    conversation_memory: str = "",
    image_understanding_block: str = "",
    math_prompt_block: str = "",
) -> list[dict[str, str]]:
    mode = normalize_agent_mode(agent_mode)
    template = _MODE_PROMPTS[mode]
    if textbook.strip():
        textbook_block = textbook
    elif mode == "free":
        textbook_block = (
            "TEXTBOOK CONTEXT:\n- None for this turn. "
            "Answer from general knowledge at their grade level, even if the topic is not one of their subjects."
        )
    else:
        textbook_block = (
            "TEXTBOOK CONTEXT:\n- None retrieved for this turn. "
            "Do not invent textbook content, and never answer from a different subject. "
            "If DETECTED SCOPE names a subject but no chapter, say you couldn't find that "
            "chapter in their book for that subject and list its chapter titles from STUDENT CONTEXT "
            "so they can pick one. Otherwise tell the student to use Ask AI Tutor "
            "for general or out-of-book questions."
        )
    system = template.format(
        student_name=_first_name(student_name),
        context=context,
        detected=detected,
        textbook=textbook_block,
    )
    if conversation_memory.strip():
        system = system + f"\n\n{conversation_memory}"
    if image_understanding_block.strip():
        system = system + f"\n\n{image_understanding_block.strip()}"
    if math_prompt_block.strip():
        system = system + f"\n\n{math_prompt_block.strip()}"
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if conversation_history:
        for turn in conversation_history[-8:]:
            role = (turn.get("role") or "").strip()
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    user_q = (query or "").strip() or "Please help me with the attached image."
    messages.append({"role": "user", "content": user_q})
    return messages


def build_greeting(user: User) -> str:
    # ponytail: no auto greeting bubble — chat starts empty with suggestion chips
    return ""


def mode_suggested_prompts(db: Session, user: User, agent_mode: str, *, overview=None) -> list[str]:
    mode = normalize_agent_mode(agent_mode)
    if mode == "free":
        return build_suggested_prompts(db, user, overview=overview)
    if overview is None:
        overview = learning_service.get_overview(db, user)
    subject = overview.subjects[0].subject_name if overview.subjects else "your subject"
    chapter = None
    if overview.continue_learning and overview.continue_learning.chapter_name:
        chapter = overview.continue_learning.chapter_name
    elif overview.subjects and overview.subjects[0].chapters:
        ch = overview.subjects[0].chapters[0]
        chapter = ch.chapter or ch.file_name
    topic = chapter or subject
    if mode == "ask":
        return [
            f"What is important to know about {topic}?",
            f"What does {subject} say about this chapter?",
            f"Quick question about {topic}",
        ][:6]
    if mode == "practice":
        return [
            f"Give me a practice problem on {topic}",
            f"Quiz me on {subject}",
            "Give me an easy problem first",
        ][:6]
    return [
        f"Explain {topic} step by step",
        f"Explain the main ideas of {subject}",
        "Explain with a simple example",
    ][:6]


@dataclass
class _Turn:
    query: str
    messages: list[dict[str, str]]
    overview: object
    fallback: str


async def _understand_images_for_turn(user: User, overview, query: str, image_ids: list[str]):
    from app.modules.image_understanding.service import understand as understand_images

    class_level = str(getattr(overview, "class_level", "") or "")
    return await understand_images(
        image_ids,
        query,
        user_id=int(user.id),
        class_level=class_level,
    )


def _prepare_turn(
    db: Session,
    user: User,
    *,
    overview,
    query: str,
    retrieval_query: str,
    conversation_history: list[dict[str, str]] | None,
    mode: str,
    image_unclear: bool,
    image_block: str,
    math_block: str,
) -> tuple[list[dict[str, str]], str]:
    """Blocking turn setup (DB, embeddings, vector search, SymPy) — run in a worker thread.

    Returns the LLM messages and the student-facing text to show if the model returns nothing.
    """
    context = build_student_context(db, user, overview=overview)
    turn = _resolve_turn_context(query, conversation_history)
    _conv, recent_hist, mem = turn
    scope_q = retrieval_query or query
    # Unreadable image: the tutor must ask for a clearer photo, so don't fetch unrelated textbook text.
    scope = {} if image_unclear else detect_learning_scope(
        db,
        user,
        scope_q,
        recent_hist,
        agent_mode=mode,
        overview=overview,
        turn=turn if scope_q == query else None,
    )
    textbook = (
        _textbook_snippets(
            scope,
            scope_q,
            agent_mode=mode,
            conversation_history=recent_hist,
        )
        if scope
        else ""
    )
    math_block = math_block or _prepare_math_engine_block(
        query,
        subject_name=scope.get("subject_name", ""),
        class_level="",
        context=textbook,
        conversation_history=recent_hist,
    )
    messages = _build_messages(
        context=context,
        student_name=user.full_name,
        query=query,
        conversation_history=recent_hist,
        agent_mode=mode,
        detected=_format_detected(scope),
        textbook=textbook,
        conversation_memory=format_memory_for_prompt(mem),
        image_understanding_block=image_block,
        math_prompt_block=math_block,
    )
    return messages, _fallback_text(mode, textbook)


async def _start_turn(
    db: Session,
    user: User,
    *,
    query: str,
    conversation_history: list[dict[str, str]] | None,
    mode: str,
    image_ids: list[str] | None,
    on_image_status=None,
) -> _Turn:
    overview = await asyncio.to_thread(learning_service.get_overview, db, user)
    image_block = ""
    math_block = ""
    retrieval_query = query
    image_unclear = False
    if image_ids:
        bundle = await _understand_images_for_turn(user, overview, query, image_ids)
        if bundle:
            image_unclear = bundle.result.status == "unclear"
            image_block = bundle.tutor_prompt_block
            math_block = bundle.math_prompt_block
            if bundle.retrieval_query:
                retrieval_query = bundle.retrieval_query
            if not (query or "").strip():
                query = bundle.retrieval_query or "Please help me with the attached image."
            if on_image_status:
                await on_image_status(
                    "\0STATUS:image_understood:"
                    f"{bundle.result.image_type}:{bundle.intent}:{bundle.result.confidence}"
                )
    messages, fallback = await asyncio.to_thread(
        _prepare_turn,
        db,
        user,
        overview=overview,
        query=query,
        retrieval_query=retrieval_query,
        conversation_history=conversation_history,
        mode=mode,
        image_unclear=image_unclear,
        image_block=image_block,
        math_block=math_block,
    )
    return _Turn(query=query, messages=messages, overview=overview, fallback=fallback)


async def chat(
    db: Session,
    user: User,
    *,
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
    agent_mode: str = "free",
    image_ids: list[str] | None = None,
) -> tuple[str, list[str]]:
    mode = normalize_agent_mode(agent_mode)
    turn = await _start_turn(
        db,
        user,
        query=query,
        conversation_history=conversation_history,
        mode=mode,
        image_ids=image_ids,
    )
    messages = turn.messages
    answer = (
        await _call_mistral_async(
            messages,
            max_tokens=ASSISTANT_MAX_TOKENS,
            feature="assistant",
            empty_fallback=turn.fallback,
        )
        or ""
    ).strip() or turn.fallback
    if answer != turn.fallback and _looks_incomplete(answer):
        more = (
            await _call_mistral_async(
                _with_continue(messages, answer),
                max_tokens=ASSISTANT_MAX_TOKENS,
                feature="assistant",
                empty_fallback="",
            )
            or ""
        ).strip()
        tail = _continuation_tail(answer, more)
        if tail is None:
            # Model restarted instead of continuing; keep the restart if it is the complete one.
            if not _looks_incomplete(more):
                answer = more
        elif tail:
            answer = (answer.rstrip() + tail).strip()
    prompts = await asyncio.to_thread(mode_suggested_prompts, db, user, mode, overview=turn.overview)
    return answer, prompts


async def chat_stream(
    db: Session,
    user: User,
    *,
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
    agent_mode: str = "free",
    image_ids: list[str] | None = None,
) -> AsyncIterator[str]:
    mode = normalize_agent_mode(agent_mode)
    # Sent before any setup so the client (and proxies) get bytes immediately.
    yield "\0STATUS:searching"
    if image_ids:
        yield "\0STATUS:image_processing"
    statuses: list[str] = []

    async def _queue_status(s: str) -> None:
        statuses.append(s)

    turn = await _start_turn(
        db,
        user,
        query=query,
        conversation_history=conversation_history,
        mode=mode,
        image_ids=image_ids,
        on_image_status=_queue_status,
    )
    for s in statuses:
        yield s
    messages = turn.messages
    parts: list[str] = []
    try:
        async for token in _stream_mistral_async(
            messages, max_tokens=ASSISTANT_MAX_TOKENS, feature="assistant"
        ):
            parts.append(token)
            yield token
    except Exception:
        # Every provider rate-limited or down: end the turn with a reply, not a broken stream.
        logger.warning("assistant LLM stream failed", exc_info=True)
        if not "".join(parts).strip():
            yield turn.fallback
        return
    answer = "".join(parts).strip()
    if not answer:
        yield turn.fallback
        return
    if not _looks_incomplete(answer):
        return
    # One continuation pass — buffer its start so an echoed overlap or a full restart
    # never gets streamed on top of what the student already sees.
    probe = ""
    checked = False
    try:
        async for token in _stream_mistral_async(
            _with_continue(messages, answer),
            max_tokens=ASSISTANT_MAX_TOKENS,
            feature="assistant",
        ):
            if checked:
                yield token
                continue
            probe += token
            if len(probe.lstrip()) < _CONTINUE_PROBE_CHARS:
                continue
            checked = True
            tail = _continuation_tail(answer, probe)
            if tail is None:
                return
            if tail:
                yield tail
    except Exception:
        logger.warning("assistant continuation stream failed", exc_info=True)
        return
    if not checked:
        tail = _continuation_tail(answer, probe)
        if tail:
            yield tail


if __name__ == "__main__":
    assert _first_name("Alex Kumar") == "Alex"
    assert normalize_agent_mode("ASK") == "ask"
    assert normalize_agent_mode("nope") == "free"
    # Modes stay distinct: free = out-of-book OK; quick-start = textbook-only redirect.
    assert "outside their uploaded textbook" in _FREE_PROMPT
    assert "SHORT, direct answer" in _ASK_PROMPT
    assert "SMALL or MEDIUM" in _LENGTH_STYLE
    assert "40 to 120 words" in _FREE_PROMPT
    assert "Always finish completely" in _FREE_PROMPT
    assert _looks_incomplete("Step 2: Add Single-Digit Numbers. Write them vertically:")
    assert not _looks_incomplete("Adding is combining groups. 4 + 7 = 11.")
    assert "ANY enrolled textbook chapter" in _ASK_PROMPT
    assert "do not need to say \"switch\"" in _ASK_PROMPT
    assert "ONE practice problem" in _PRACTICE_PROMPT
    assert "Answer the exact question" in _PRACTICE_PROMPT
    assert "ONE short follow-up question" in _PRACTICE_PROMPT
    assert "Chapter stickiness" in _PRACTICE_PROMPT
    assert "step-by-step explanation" in _EXPLAIN_PROMPT or "mini-explanation" in _EXPLAIN_PROMPT
    for prompt in (_ASK_PROMPT, _PRACTICE_PROMPT, _EXPLAIN_PROMPT):
        assert "Ask AI Tutor" in prompt
        assert "TEXTBOOK CONTEXT" in prompt
    # Cold-start: short in-book words like "wit" must hit chapter titles.
    assert _title_match_score(_norm("What is wit?"), "Unit 1 - Wit and Wisdom") >= 2
    assert _title_match_score(_norm("What is wit?"), "Photosynthesis") == 0
    # Stem overlap works across subjects (not Social-only).
    assert (
        _title_match_score(
            _norm("when does Nature becomes A resource?"),
            "Chapter 1 - Natural Resources and Their Use",
        )
        >= 2
    )
    assert _title_match_score(_norm("fraction problems"), "Chapter 2 - Fractions and Decimals") >= 2
    assert (
        _title_match_score(
            _norm("how do plants photosynthesize"),
            "Chapter 5 - Photosynthesis in Plants",
        )
        >= 2
    )
    # Chapter number in the question is an explicit title hit.
    assert (
        _title_match_score(
            _norm("explain me social chapter 2"),
            "Chapter 2 - Reshaping India's Political Map",
        )
        >= 8
    )
    assert (
        _title_match_score(
            _norm("explain me social chapter 2"),
            "Chapter 1 - Natural Resources and Their Use",
        )
        < 8
    )
    assert _TEXTBOOK_MODES == frozenset({"ask", "practice", "explain"})
    # Follow-up inherits prior user turn that named the chapter.
    q2 = _scope_query_text(
        "What is wit?",
        [{"role": "user", "content": "What is important to know about Unit 1 - Wit and Wisdom?"}],
    )
    assert "wit" in q2 and "wisdom" in q2
    # "chapter 1 in english" must stay in English even though only Science titles say "Chapter 1".
    from types import SimpleNamespace as NS

    sci1 = NS(id=1, chapter="Chapter 1: Exploring the Investigative World of Science", file_name="hecu101.pdf")
    eng_a = NS(id=2, chapter="A Tale of Two Birds", file_name="hepr101.pdf")
    eng_b = NS(id=3, chapter="The Wit that Won Hearts", file_name="hepr103.pdf")
    eng_untagged = NS(id=4, chapter="Somebody's Mother", file_name="x.pdf")
    ov = NS(
        subjects=[
            NS(subject_name="Science", chapters=[sci1]),
            NS(subject_name="English", chapters=[eng_b, eng_a, eng_untagged]),
        ]
    )
    _, subj, ch = _best_learning_match(ov, _norm("any way explain chapter 1 in english"))
    assert subj.subject_name == "English" and ch is eng_a, (subj, ch)
    _, subj, ch = _best_learning_match(ov, _norm("explain chapter 1"))
    assert ch in (sci1, eng_a)
    # Subject named, no chapter matched → subject only, never the alphabetically-first chapter.
    _, subj, ch = _best_learning_match(ov, _norm("explain chapter 7 in english"))
    assert subj.subject_name == "English" and ch is None
    assert _title_number("Unit 2 - Values and Dispositions", "unit2.pdf") is None
    assert _title_number("Somebody's Mother", "9eaddd2923_hepr103.pdf") == 3
    assert _title_number("Somebody's Mother", "notes.pdf") is None
    print("student_assistant service self-check ok")
