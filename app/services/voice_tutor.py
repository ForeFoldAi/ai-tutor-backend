"""
Voice Tutor mode — state machine and understanding heuristics.

All subjects use conversational live-teaching prompts in voice mode.
Full text-chat format is reserved for explicit “full solution” or
“explain in detail” requests (see voice_wants_full_written_answer).

Prompt templates live in voice_prompts.py (Phase 5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.services.voice_prompts import (
    VOICE_SYSTEM_PROMPT,
    build_voice_system_prompt,
    build_voice_user_message,
    voice_grade_band,
)

# Re-export for tests and callers that import from voice_tutor
__all__ = [
    "VOICE_SYSTEM_PROMPT",
    "voice_grade_band",
]

VOICE_HISTORY_TURNS = 14  # 7 user/assistant pairs max injected into Mistral


class TutorState(str, Enum):
    LISTENING = "LISTENING"
    TEACHING = "TEACHING"
    CHECKING_UNDERSTANDING = "CHECKING_UNDERSTANDING"
    QUIZING = "QUIZING"
    CLARIFYING = "CLARIFYING"


_VOICE_EXPAND = re.compile(
    r"\b("
    r"explain\s+more|tell\s+me\s+more|more\s+detail|in\s+detail|"
    r"don'?t\s+understand|do\s+not\s+understand|didn'?t\s+understand|"
    r"i\s+don'?t\s+get|not\s+clear|confused|"
    r"give\s+details|teach\s+me\s+deeply|go\s+deeper|elaborate"
    r")\b",
    re.I,
)

_CONFUSION = re.compile(
    r"\b(confused|don'?t\s+understand|not\s+clear|what\s+do\s+you\s+mean|huh|"
    r"no\s+idea|i\s+don'?t\s+get|still\s+confused|too\s+hard)\b",
    re.I,
)
_AFFIRM = re.compile(
    r"^(yes|yeah|yep|yup|ok|okay|sure|right|correct|exactly|got\s+it|"
    r"i\s+understand|understood|makes\s+sense)[.!?]*$",
    re.I,
)
_QUIZ_INTENT = re.compile(
    r"\b(quiz\s+me|test\s+me|ask\s+me\s+questions?|\d+\s+questions?)\b",
    re.I,
)
_FULL_WRITTEN_ANSWER = re.compile(
    r"\b("
    r"full\s+solution|complete\s+solution|step\s+by\s+step\s+solution|"
    r"show\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?steps|solve\s+(?:it|this)\s+completely|"
    r"write\s+(?:the\s+)?(?:full|complete)\s+answer|give\s+(?:me\s+)?(?:the\s+)?full\s+answer|"
    r"detailed\s+written\s+answer|all\s+steps\s+written"
    r")\b",
    re.I,
)
_PROBLEM_ATTEMPT = re.compile(
    r"\b(i\s+got|my\s+answer|i\s+think\s+it'?s|equals?|=\s*\d|step\s+\d)\b",
    re.I,
)


@dataclass
class UnderstandingScores:
    understanding: float = 0.5
    confidence: float = 0.5
    confusion: float = 0.0
    is_affirmation: bool = False
    wants_expansion: bool = False
    wants_quiz: bool = False

    def to_hint(self) -> str:
        if self.confusion >= 0.55:
            return (
                "The student seems confused. Simplify language, use an analogy, "
                "and give one concrete example. Echo what confused them, then clarify ONE point. "
                "Do not add new topics."
            )
        if self.is_affirmation and self.understanding >= 0.65:
            return (
                "The student understood the last point. Briefly acknowledge what they got right, "
                "then teach the next small step on the same topic."
            )
        if self.wants_expansion:
            return (
                "The student asked for more detail. You may use up to 120 words, "
                "but keep 6–12 word sentences, pause every 1–2 lines, example-led."
            )
        if self.wants_quiz:
            return "Ask ONE short quiz question orally. Wait for their answer next turn."
        return (
            "Use short spoken lines (6–12 words). Open with a quick nod to what they said, "
            "then one example or 'Imagine…' — not a definition dump."
        )

    def to_student_hint(self, student_text: str = "") -> str:
        """Short student-facing line for the voice UI."""
        if self.confusion >= 0.55:
            return "Let me explain that more simply…"
        if self.wants_quiz:
            return "Quick quiz for you…"
        if self.is_affirmation:
            return "Great — let's build on that"
        if self.wants_expansion:
            return "Going a bit deeper for you…"
        q = (student_text or "").strip()
        if not q:
            return ""
        words = q.split()
        if len(words) <= 6:
            return f'I heard: "{q}"'
        return f'I heard: "{" ".join(words[:6])}…"'


@dataclass
class LearnerProfileSnapshot:
    student_key: str = ""
    strong_topics: list[str] = field(default_factory=list)
    weak_topics: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)
    quiz_scores: list[float] = field(default_factory=list)

    def to_prompt_hint(self) -> str:
        parts: list[str] = []
        if self.weak_topics:
            parts.append(f"Student struggled before with: {', '.join(self.weak_topics[-5:])}.")
        if self.strong_topics:
            parts.append(f"Student is confident with: {', '.join(self.strong_topics[-5:])}.")
        return " ".join(parts)


def voice_expand_requested(query: str) -> bool:
    return bool(_VOICE_EXPAND.search(query or ""))


def voice_wants_full_written_answer(query: str) -> bool:
    """True when the student explicitly wants a full written/step-by-step solution."""
    return bool(_FULL_WRITTEN_ANSWER.search(query or ""))


def build_acknowledgment_guidance(student_text: str, scores: UnderstandingScores) -> str:
    """Prompt injection so the model echoes the student before teaching."""
    q = (student_text or "").strip()
    if not q:
        return ""
    parts = [f'Student\'s exact words: "{q[:200]}"']
    if _PROBLEM_ATTEMPT.search(q):
        parts.append(
            "They shared a step or answer — respond to their attempt first before teaching anything new."
        )
    if scores.confusion >= 0.55:
        parts.append("Acknowledge that this seems confusing for them.")
    return " ".join(parts)


def evaluate_student_response(
    student_text: str,
    *,
    last_assistant: str = "",
    tutor_state: TutorState = TutorState.LISTENING,
) -> UnderstandingScores:
    """Lightweight heuristic understanding estimate (no extra LLM call)."""
    q = (student_text or "").strip()
    if not q:
        return UnderstandingScores()

    words = len(q.split())
    wants_expansion = voice_expand_requested(q)
    wants_quiz = bool(_QUIZ_INTENT.search(q))
    is_affirm = bool(_AFFIRM.match(q))
    confusion_hit = bool(_CONFUSION.search(q))

    confusion = 0.75 if confusion_hit else 0.15
    if words <= 3 and q.lower() in ("no", "nope", "nah"):
        confusion = 0.6
    if tutor_state == TutorState.CHECKING_UNDERSTANDING and "?" in q and not is_affirm:
        confusion = max(confusion, 0.45)

    understanding = 0.35
    if is_affirm:
        understanding = 0.85
    elif words >= 8 and not confusion_hit:
        understanding = 0.7
    elif words >= 4:
        understanding = 0.55

    confidence = min(0.95, understanding + (0.1 if words >= 6 else 0))

    return UnderstandingScores(
        understanding=understanding,
        confidence=confidence,
        confusion=confusion,
        is_affirmation=is_affirm,
        wants_expansion=wants_expansion,
        wants_quiz=wants_quiz,
    )


def next_tutor_state(
    *,
    current: TutorState,
    scores: UnderstandingScores,
    assistant_reply: str = "",
) -> TutorState:
    if scores.wants_quiz:
        return TutorState.QUIZING
    if scores.confusion >= 0.55 or scores.wants_expansion:
        return TutorState.CLARIFYING
    if current == TutorState.QUIZING:
        return TutorState.CHECKING_UNDERSTANDING
    reply = (assistant_reply or "").strip()
    if reply.endswith("?"):
        return TutorState.CHECKING_UNDERSTANDING
    return TutorState.TEACHING


def _state_guidance(state: TutorState) -> str:
    mapping = {
        TutorState.LISTENING: "Student may speak next. One warm short line — don't teach yet.",
        TutorState.TEACHING: (
            "Teach one small idea. 6–12 word sentences, contractions, example before definition. "
            "Pause after every 1–2 sentences. Optional check-in only if it helps."
        ),
        TutorState.CHECKING_UNDERSTANDING: (
            "You asked a question last turn. Echo their answer briefly, "
            "say if they're on track, then ONE short hint or follow-up."
        ),
        TutorState.QUIZING: "One short oral quiz question. Don't explain at length.",
        TutorState.CLARIFYING: (
            "They're stuck. Simpler words, one 'Imagine…' example, two short sentences max."
        ),
    }
    return mapping.get(state, mapping[TutorState.TEACHING])


def _normalize_history(history: list[dict] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for turn in history or []:
        role = (turn.get("role") or "").lower()
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out[-VOICE_HISTORY_TURNS:]


def _build_voice_messages(
    query: str,
    context: str,
    *,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    student_name: str = "",
    conversation_history: list[dict] | None = None,
    tutor_state: TutorState = TutorState.TEACHING,
    understanding: UnderstandingScores | None = None,
    learner: LearnerProfileSnapshot | None = None,
    expand_deep: bool = False,
) -> list[dict[str, str]]:
    scores = understanding or UnderstandingScores()
    history = _normalize_history(conversation_history)
    last_assistant = ""
    for turn in reversed(history):
        if turn.get("role") == "assistant":
            last_assistant = turn.get("content") or ""
            break

    system = build_voice_system_prompt(
        query,
        class_level=class_level,
        board=board,
        subject_name=subject_name,
        chapter=chapter,
        student_name=student_name,
        conversation_history=history,
        tutor_state=tutor_state,
        understanding=scores,
        learner=learner,
        expand_deep=expand_deep,
        last_assistant=last_assistant,
        state_guidance=_state_guidance(tutor_state),
        understanding_guidance=scores.to_hint(),
        acknowledgment_guidance=build_acknowledgment_guidance(query, scores),
    )
    user = build_voice_user_message(query, context, expand_deep=expand_deep)

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in history:
        messages.append(turn)
    messages.append({"role": "user", "content": user})
    return messages


def build_voice_mistral_messages(
    query: str,
    context: str,
    **kwargs: Any,
) -> list[dict[str, str]]:
    """Public entry: build [system, history..., user] for voice mode."""
    expand_deep = kwargs.pop("expand_deep", False) or voice_expand_requested(query)
    return _build_voice_messages(query, context, expand_deep=expand_deep, **kwargs)
