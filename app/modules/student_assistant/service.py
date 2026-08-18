"""Student-scoped general assistant (platform, profile, subjects, progress, quick-start modes)."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from sqlalchemy.orm import Session

from app.modules.student_assistant.context import build_student_context, build_suggested_prompts
from app.modules.student_learning import service as learning_service
from app.modules.users.models import User
from app.services.chapter_scope import (
    chapter_number_from_name,
    extract_mentioned_chapter_numbers,
)
from app.services.chat_service import _call_mistral_async, _stream_mistral_async
from app.services.conversation_context import resolve_conversation_context
from app.services.conversation_memory import format_memory_for_prompt, prepare_conversation_inputs

_AGENT_MODES = frozenset({"free", "ask", "practice", "explain"})

_FREE_PROMPT = """You are the AI Tutor assistant for {student_name}, a student on the AI Tutor learning platform.

You may answer:
- Platform / account help: the student (name, grade, class, progress, streak, recent lessons), enrolled subjects, school/tutor (if present), and how to use AI Tutor features
- School subjects and general academic questions — including topics outside their uploaded textbook
- Homework help, explanations, study tips, and everyday school questions at their grade level

Student facts (enrollment, progress, school, tutor):
- Use ONLY STUDENT CONTEXT below. Never invent subjects, grades, teachers, or progress not listed there.
- If the context lacks a personal/platform fact, say you do not have that information and suggest who to ask (tutor, school admin) or where in the app to look.

Academic / out-of-book help:
- You MAY use general knowledge appropriate for their grade. Prefer clear steps and simple examples.
- When TEXTBOOK CONTEXT is present and relevant, prefer it; if the question is outside the textbook, still answer helpfully (this mode allows out-of-book).
- For deeper chapter-locked tutoring, you may mention Chapter AI Tutor or AI Voice after picking a subject/chapter.

Rules:
- Never reveal other students' data or admin-only information.
- Keep answers clear, friendly, and age-appropriate. Use short paragraphs or bullets when helpful.
- Never use markdown formatting.
- Never output the '*' character (asterisk).

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
- Never use markdown formatting or the '*' character.
- Keep answers concise (a few short paragraphs or bullets).

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
3. Show step-by-step mathematical reasoning when appropriate.
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
- Keep explanations concise and suitable for the student.
- Never use markdown formatting or the '*' character.
- Prefer short problems; one at a time when generating practice.

STUDENT CONTEXT:
{context}
{detected}
{textbook}
"""

_EXPLAIN_PROMPT = """You are the AI Tutor for {student_name}. Mode: Explain a Topic.

Purpose: Teach with a clear step-by-step explanation of a textbook topic (mini-lesson).
Do NOT start a quiz or practice problems unless the student explicitly asks.
Do NOT give only a one-line definition — explain with short steps and a simple example when useful.

Textbook rule (STRICT):
- Explain ONLY using TEXTBOOK CONTEXT below (and detected subject/chapter).
- If TEXTBOOK CONTEXT is missing or the topic is outside the textbook, do NOT invent a lesson from general knowledge.
- In that case reply briefly: this mode is for textbook explanations only, and ask them to open Ask AI Tutor for general / platform / out-of-book questions.

Detect subject/topic from the request when possible. Do not ask them to pick a subject or chapter first.

Rules:
- Use short steps and simple wording for their grade.
- Never use markdown formatting or the '*' character.

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


def _title_match_score(q: str, title: str) -> int:
    """Score how well a normalized query matches a subject/chapter title."""
    label = _norm(title)
    if not label or not q:
        return 0
    if label in q:
        return 8
    score = sum(2 for token in _scope_tokens(label) if _token_in(token, q))
    # "chapter 2" is an explicit pick — same weight as a full title hit
    q_nums = extract_mentioned_chapter_numbers(q)
    t_num = chapter_number_from_name(title)
    if t_num is not None and t_num in q_nums:
        score += 8
    return score


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

    for subj in overview.subjects:
        subj_name = _norm(subj.subject_name)
        score = 0
        if subj_name and subj_name in q:
            score += 5
        score += sum(1 for token in _scope_tokens(subj_name) if _token_in(token, q))
        for ch in subj.chapters:
            label = ch.chapter or ch.file_name or ""
            ch_score = score + _title_match_score(q, label)
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

    for subj in overview.subjects:
        if not subj.chapters:
            continue
        board = getattr(subj.board, "value", str(subj.board))
        class_level = getattr(subj.class_level, "value", str(subj.class_level))
        collection = f"{board}_{class_level}_{subj.subject_name}".replace(" ", "_")
        try:
            docs = retrieve_from_collection(
                q,
                collection_name=collection,
                chapter_ids=None,
                k=6,
            )
        except Exception:
            continue
        for doc in docs or []:
            meta = getattr(doc, "metadata", None) or {}
            ch = _chapter_from_doc_meta(subj, meta)
            if ch is None:
                continue
            text = getattr(doc, "page_content", None) or str(doc) or ""
            score = keyword_match_score(q, text)
            score += _title_match_score(q, ch.chapter or ch.file_name or "")
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
) -> dict:
    """Match enrolled subject/chapter from the free-text prompt (no picker required)."""
    overview = learning_service.get_overview(db, user)
    if not overview.subjects:
        return {}

    mode = normalize_agent_mode(agent_mode)
    conv, recent_hist, mem = _resolve_turn_context(query, conversation_history)
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
                    or _title_match_score(q_cur, cur_label) >= 8
                )
            )
            if explicit_switch:
                return _scope_dict(cur_subj, cur_ch, cur_score)
            return _scope_dict(hist_subj, hist_ch, hist_score)

    scoped = _scope_dict(full_subj, full_ch, full_score)
    # Ask / Practice / Explain: if title match missed (any subject), fall back to
    # content search across enrolled textbooks so section headings still resolve.
    if mode in _TEXTBOOK_MODES and not scoped.get("chapter_id"):
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
        for doc in docs[:4]:
            text = (getattr(doc, "page_content", None) or str(doc) or "").strip()
            if text:
                bits.append(text[:700])
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
) -> list[dict[str, str]]:
    mode = normalize_agent_mode(agent_mode)
    template = _MODE_PROMPTS[mode]
    if textbook.strip():
        textbook_block = textbook
    elif mode == "free":
        textbook_block = (
            "TEXTBOOK CONTEXT:\n- None for this turn. "
            "Answer academic questions with general knowledge appropriate for their grade."
        )
    else:
        textbook_block = (
            "TEXTBOOK CONTEXT:\n- None retrieved for this turn. "
            "Do not invent textbook content. Tell the student to use Ask AI Tutor "
            "for general or out-of-book questions."
        )
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": template.format(
                student_name=_first_name(student_name),
                context=context,
                detected=detected,
                textbook=textbook_block,
            )
            + (f"\n\n{conversation_memory}" if conversation_memory.strip() else ""),
        }
    ]
    if conversation_history:
        for turn in conversation_history[-8:]:
            role = (turn.get("role") or "").strip()
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query.strip()})
    return messages


def build_greeting(user: User) -> str:
    # ponytail: no auto greeting bubble — chat starts empty with suggestion chips
    return ""


def mode_suggested_prompts(db: Session, user: User, agent_mode: str) -> list[str]:
    mode = normalize_agent_mode(agent_mode)
    if mode == "free":
        return build_suggested_prompts(db, user)
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


async def chat(
    db: Session,
    user: User,
    *,
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
    agent_mode: str = "free",
) -> tuple[str, list[str]]:
    mode = normalize_agent_mode(agent_mode)
    context = build_student_context(db, user)
    conv, recent_hist, mem = _resolve_turn_context(query, conversation_history)
    scope = detect_learning_scope(
        db, user, query, recent_hist, agent_mode=mode
    )
    textbook = (
        _textbook_snippets(
            scope,
            query,
            agent_mode=mode,
            conversation_history=recent_hist,
        )
        if scope
        else ""
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
    )
    answer = (await _call_mistral_async(messages, max_tokens=900, feature="assistant") or "").strip()
    return answer, mode_suggested_prompts(db, user, mode)


async def chat_stream(
    db: Session,
    user: User,
    *,
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
    agent_mode: str = "free",
) -> AsyncIterator[str]:
    mode = normalize_agent_mode(agent_mode)
    context = build_student_context(db, user)
    conv, recent_hist, mem = _resolve_turn_context(query, conversation_history)
    scope = detect_learning_scope(
        db, user, query, recent_hist, agent_mode=mode
    )
    textbook = (
        _textbook_snippets(
            scope,
            query,
            agent_mode=mode,
            conversation_history=recent_hist,
        )
        if scope
        else ""
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
    )
    async for token in _stream_mistral_async(messages, max_tokens=900, feature="assistant"):
        yield token


if __name__ == "__main__":
    assert _first_name("Alex Kumar") == "Alex"
    assert normalize_agent_mode("ASK") == "ask"
    assert normalize_agent_mode("nope") == "free"
    # Modes stay distinct: free = out-of-book OK; quick-start = textbook-only redirect.
    assert "outside their uploaded textbook" in _FREE_PROMPT
    assert "SHORT, direct answer" in _ASK_PROMPT
    assert "ANY enrolled textbook chapter" in _ASK_PROMPT
    assert "do not need to say \"switch\"" in _ASK_PROMPT
    assert "ONE practice problem" in _PRACTICE_PROMPT
    assert "Answer the exact question" in _PRACTICE_PROMPT
    assert "ONE short follow-up question" in _PRACTICE_PROMPT
    assert "Chapter stickiness" in _PRACTICE_PROMPT
    assert "step-by-step explanation" in _EXPLAIN_PROMPT
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
    print("student_assistant service self-check ok")
