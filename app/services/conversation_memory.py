"""
Rolling conversation memory for long tutor sessions.

Keeps recent turns for resolver/RAG while folding older turns into a compact
summary so 500+ follow-ups do not blow the LLM context window.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from app.config import (
    CONVERSATION_FOLD_BATCH_SIZE,
    CONVERSATION_RECENT_TURN_WINDOW,
    CONVERSATION_SUMMARY_MAX_CHARS,
)

__all__ = [
    "ConversationMemory",
    "prepare_conversation_inputs",
    "remember_turn",
    "format_memory_for_prompt",
    "memory_from_dict",
    "memory_to_dict",
    "update_memory_after_turn",
]


@dataclass
class ConversationMemory:
    conversation_summary: str = ""
    topics_discussed: list[str] = field(default_factory=list)
    concepts_explained: list[str] = field(default_factory=list)
    student_questions: list[str] = field(default_factory=list)
    last_student_question: str = ""
    last_ai_snippet: str = ""
    current_concept: str = ""
    turn_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> ConversationMemory:
        if not data:
            return cls()
        return cls(
            conversation_summary=str(data.get("conversation_summary") or ""),
            topics_discussed=list(data.get("topics_discussed") or [])[:40],
            concepts_explained=list(data.get("concepts_explained") or [])[:40],
            student_questions=list(data.get("student_questions") or [])[:60],
            last_student_question=str(data.get("last_student_question") or ""),
            last_ai_snippet=str(data.get("last_ai_snippet") or ""),
            current_concept=str(data.get("current_concept") or ""),
            turn_count=int(data.get("turn_count") or 0),
        )


def memory_to_dict(memory: ConversationMemory | dict | None) -> dict:
    if memory is None:
        return {}
    if isinstance(memory, dict):
        return memory
    return memory.to_dict()


def memory_from_dict(data: ConversationMemory | dict | None) -> ConversationMemory:
    if isinstance(data, ConversationMemory):
        return data
    return ConversationMemory.from_dict(data)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _cap_list(items: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in reversed(items):
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    out.reverse()
    return out


def _append_summary(memory: ConversationMemory, line: str) -> None:
    line = _norm(line)
    if not line:
        return
    if memory.conversation_summary:
        memory.conversation_summary = f"{memory.conversation_summary}\n{line}"
    else:
        memory.conversation_summary = line
    if len(memory.conversation_summary) > CONVERSATION_SUMMARY_MAX_CHARS:
        memory.conversation_summary = memory.conversation_summary[
            -CONVERSATION_SUMMARY_MAX_CHARS:
        ]


def _pair_turns(turns: list[dict]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pending_user = ""
    for turn in turns:
        role = (turn.get("role") or "").lower()
        content = _norm(turn.get("content") or "")
        if not content:
            continue
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user:
            pairs.append((pending_user, content))
            pending_user = ""
    if pending_user:
        pairs.append((pending_user, ""))
    return pairs


def _fold_turn_list_into_memory(
    memory: ConversationMemory,
    turns: list[dict],
    *,
    chapter: str = "",
) -> ConversationMemory:
    for user_q, assistant_a in _pair_turns(turns):
        u_short = user_q[:120]
        a_short = assistant_a[:160] if assistant_a else ""
        if a_short:
            _append_summary(
                memory,
                f"Student: {u_short}. Tutor: {a_short}",
            )
            memory.concepts_explained.append(a_short[:100])
        else:
            _append_summary(memory, f"Student asked: {u_short}")
        memory.student_questions.append(u_short)
        topic = u_short[:80]
        if topic:
            memory.topics_discussed.append(topic)
        memory.current_concept = (assistant_a or user_q)[:120]
    if chapter and chapter not in memory.topics_discussed:
        memory.topics_discussed.insert(0, chapter[:120])
    memory.topics_discussed = _cap_list(memory.topics_discussed, 30)
    memory.concepts_explained = _cap_list(memory.concepts_explained, 30)
    memory.student_questions = _cap_list(memory.student_questions, 40)
    return memory


def update_memory_after_turn(
    memory: ConversationMemory,
    *,
    user_query: str,
    assistant_response: str,
    resolved_topic: str = "",
    followup_type: str = "",
    chapter: str = "",
) -> ConversationMemory:
    u = _norm(user_query)
    a = _norm(assistant_response)
    if u:
        memory.last_student_question = u[:200]
        memory.student_questions.append(u[:120])
    if a:
        memory.last_ai_snippet = a[:400]
        memory.concepts_explained.append(a[:100])
    concept = _norm(resolved_topic) or u
    if concept:
        memory.current_concept = concept[:120]
        memory.topics_discussed.append(concept[:80])
    if chapter:
        memory.topics_discussed.insert(0, chapter[:120])
    memory.turn_count += 1
    memory.topics_discussed = _cap_list(memory.topics_discussed, 30)
    memory.concepts_explained = _cap_list(memory.concepts_explained, 30)
    memory.student_questions = _cap_list(memory.student_questions, 40)
    return memory


def format_memory_for_prompt(memory: ConversationMemory | dict | None) -> str:
    mem = memory_from_dict(memory)
    if mem.turn_count <= 0 and not mem.conversation_summary.strip():
        return ""
    lines = ["SESSION MEMORY (older parts of this lesson):"]
    if mem.conversation_summary.strip():
        lines.append(mem.conversation_summary.strip())
    if mem.topics_discussed:
        lines.append("Topics discussed: " + "; ".join(mem.topics_discussed[-12:]))
    if mem.current_concept:
        lines.append(f"Current concept: {mem.current_concept}")
    if mem.last_student_question:
        lines.append(f"Last student question: {mem.last_student_question[:160]}")
    return "\n".join(lines)


def prepare_conversation_inputs(
    history: list[dict] | None,
    *,
    memory: ConversationMemory | dict | None = None,
    chapter: str = "",
    recent_window: int | None = None,
) -> tuple[list[dict], ConversationMemory]:
    """
    Split client/server history into recent turns (for resolver + LLM) and
    rolling memory (for older context).
    """
    window = recent_window or CONVERSATION_RECENT_TURN_WINDOW
    hist = list(history or [])
    mem = memory_from_dict(memory)

    if len(hist) > window:
        older = hist[:-window]
        if older:
            mem = _fold_turn_list_into_memory(mem, older, chapter=chapter)
        hist = hist[-window:]

    return hist, mem


def remember_turn(
    history: list[dict],
    memory: ConversationMemory,
    *,
    user_query: str,
    assistant_response: str,
    resolved_topic: str = "",
    followup_type: str = "",
    chapter: str = "",
    history_cap: int = 20,
) -> tuple[list[dict], ConversationMemory]:
    """Append a turn, update memory, fold oldest turns when history grows."""
    history = list(history)
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": assistant_response})
    mem = update_memory_after_turn(
        memory,
        user_query=user_query,
        assistant_response=assistant_response,
        resolved_topic=resolved_topic,
        followup_type=followup_type,
        chapter=chapter,
    )

    if len(history) > history_cap:
        fold_n = max(CONVERSATION_FOLD_BATCH_SIZE, len(history) - history_cap)
        fold_n = min(fold_n, len(history) - 4)
        if fold_n > 0:
            mem = _fold_turn_list_into_memory(mem, history[:fold_n], chapter=chapter)
            history = history[fold_n:]

    return history, mem
