"""Student-scoped general assistant (platform, profile, subjects, progress, quick-start modes)."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

from sqlalchemy.orm import Session

from app.modules.student_assistant.context import build_student_context, build_suggested_prompts
from app.modules.student_learning import service as learning_service
from app.modules.users.models import User
from app.services.chat_service import _call_mistral_async, _stream_mistral_async

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

Textbook rule (STRICT):
- Answer ONLY using TEXTBOOK CONTEXT below (and detected subject/chapter).
- If TEXTBOOK CONTEXT is missing, empty, or does not cover the question, do NOT invent an answer from general knowledge.
- In that case reply briefly: this Quick Start mode is for textbook topics only, and ask them to open Ask AI Tutor for general / platform / out-of-book questions.

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

_PRACTICE_PROMPT = """You are the AI Tutor for {student_name}. Mode: Practice Problems.

Purpose: Act as a practice coach. Give ONE practice problem at a time from the textbook topic.
After they answer, give brief feedback, then another problem if useful.
Do NOT dump a long explanation or open lecture. Do NOT only answer a theory question without a problem.

Textbook rule (STRICT):
- Problems must be grounded in TEXTBOOK CONTEXT below (and detected subject/chapter).
- If TEXTBOOK CONTEXT is missing or the request is outside the textbook, do NOT invent problems from general knowledge.
- In that case reply briefly: this mode is for textbook practice only, and ask them to open Ask AI Tutor for general / platform / out-of-book help.

Detect subject/topic from the request when possible. Do not ask them to pick a subject or chapter first.

Rules:
- Match difficulty to their grade.
- Never use markdown formatting or the '*' character.
- Prefer short problems; one at a time.

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


def detect_learning_scope(db: Session, user: User, query: str) -> dict:
    """Match enrolled subject/chapter from the free-text prompt (no picker required)."""
    overview = learning_service.get_overview(db, user)
    q = _norm(query)
    if not q or not overview.subjects:
        return {}

    best_subject = None
    best_chapter = None
    best_score = 0

    for subj in overview.subjects:
        subj_name = _norm(subj.subject_name)
        score = 0
        if subj_name and subj_name in q:
            score += 5
        for token in subj_name.split():
            if len(token) > 3 and token in q:
                score += 1
        for ch in subj.chapters:
            label = _norm(ch.chapter or ch.file_name or "")
            if not label:
                continue
            ch_score = score
            if label in q:
                ch_score += 8
            else:
                hits = sum(1 for token in label.split() if len(token) > 3 and token in q)
                ch_score += hits * 2
            if ch_score > best_score:
                best_score = ch_score
                best_subject = subj
                best_chapter = ch

        if score > best_score:
            best_score = score
            best_subject = subj
            best_chapter = None

    if best_score < 2 or best_subject is None:
        return {}

    out: dict = {
        "subject_name": best_subject.subject_name,
        "board": getattr(best_subject.board, "value", str(best_subject.board)),
        "class_level": getattr(best_subject.class_level, "value", str(best_subject.class_level)),
    }
    if best_chapter is not None:
        out["chapter_id"] = best_chapter.id
        out["chapter_name"] = best_chapter.chapter or best_chapter.file_name
    return out


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


def _textbook_snippets(scope: dict, query: str, *, agent_mode: str = "ask") -> str:
    """Best-effort RAG snippets from the detected chapter; fail open."""
    chapter_id = scope.get("chapter_id")
    board = scope.get("board") or ""
    class_level = scope.get("class_level") or ""
    subject_name = scope.get("subject_name") or ""
    if not chapter_id or not subject_name:
        return ""
    try:
        from app.services.section_retrieval import retrieve_for_tutor_query

        collection = f"{board}_{class_level}_{subject_name}".replace(" ", "_")
        docs, _scope, _instr = retrieve_for_tutor_query(
            query,
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
        mode = normalize_agent_mode(agent_mode)
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
            ),
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
    # Free mode: optional textbook snippets when topic matches enrollment; still answers out-of-book.
    scope = detect_learning_scope(db, user, query)
    textbook = _textbook_snippets(scope, query, agent_mode=mode) if scope else ""
    messages = _build_messages(
        context=context,
        student_name=user.full_name,
        query=query,
        conversation_history=conversation_history,
        agent_mode=mode,
        detected=_format_detected(scope),
        textbook=textbook,
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
    scope = detect_learning_scope(db, user, query)
    textbook = _textbook_snippets(scope, query, agent_mode=mode) if scope else ""
    messages = _build_messages(
        context=context,
        student_name=user.full_name,
        query=query,
        conversation_history=conversation_history,
        agent_mode=mode,
        detected=_format_detected(scope),
        textbook=textbook,
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
    assert "ONE practice problem" in _PRACTICE_PROMPT
    assert "step-by-step explanation" in _EXPLAIN_PROMPT
    for prompt in (_ASK_PROMPT, _PRACTICE_PROMPT, _EXPLAIN_PROMPT):
        assert "Ask AI Tutor" in prompt
        assert "TEXTBOOK CONTEXT" in prompt
    print("student_assistant service self-check ok")
