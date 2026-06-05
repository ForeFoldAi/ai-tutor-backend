"""
Voice Tutor mode — conversational prompts, state machine, and lightweight understanding.

Separate from text chat tutoring (chat_service._SYSTEM_PROMPT_TEMPLATE).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.services.chat_service import (
    _GRADE_COMPLEXITY,
    _GRADE_LABELS,
    _class_band,
    _CLASS_BAND_RULES,
    _student_first_name,
)

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


VOICE_SYSTEM_PROMPT = """\
You are a friendly live AI Tutor having a real voice conversation with a school student.
Speak naturally — like a caring teacher on a phone call, NOT like a textbook or chatbot.

STUDENT CONTEXT:
- Name: {student_name}
- Class: {grade_label}
- Board: {board}
- Subject: {subject}
- Chapter: {chapter}

LANGUAGE:
{complexity_rule}
{class_band_rule}

VOICE RULES (CRITICAL):
- Maximum {max_words} words this turn (target 30-80 words, hard cap {max_words}).
- Teach ONE small idea per turn — never a full lecture.
- Use 2-4 short spoken sentences. Conversational tone.
- Use a simple everyday example when it helps.
- NEVER use section headers, emoji labels, bullet lists, or numbered lists.
- NEVER say "Concept Overview", "Key Points", "Quick Check", or similar.
- Do NOT try to be complete — prioritize dialogue over coverage.
- End with exactly ONE short follow-up question when teaching (not when quizzing).
- Wait for the student — do not continue the lesson in the same turn.

SESSION STATE: {tutor_state}
{state_guidance}
{understanding_guidance}
{learner_guidance}

TEXTBOOK CONTEXT:
Use the chapter excerpt in the user message first. If missing, use accurate general knowledge briefly.
Never invent textbook page numbers or figure names.

{expand_policy}"""


VOICE_USER_TEMPLATE = """\
CHAPTER EXCERPT (use for accuracy):
{context}

STUDENT SAID:
{question}

Reply in spoken conversational prose now ({max_words} words max):"""


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
                "and give one concrete example. Do not add new topics."
            )
        if self.is_affirmation and self.understanding >= 0.65:
            return (
                "The student understood the last point. Briefly acknowledge, then "
                "teach the next small step on the same topic."
            )
        if self.wants_expansion:
            return "The student asked for more detail. You may use up to 120 words this turn."
        if self.wants_quiz:
            return "Ask ONE short quiz question orally. Wait for their answer next turn."
        return ""


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
        TutorState.LISTENING: "Student may speak next. Keep your reply short and welcoming.",
        TutorState.TEACHING: "Introduce one small concept, then ask one check question.",
        TutorState.CHECKING_UNDERSTANDING: (
            "You asked a question last turn. Now evaluate their answer briefly and continue."
        ),
        TutorState.QUIZING: "Ask one oral quiz question only. Do not explain at length.",
        TutorState.CLARIFYING: "Student needs simpler explanation. Use analogy + one example.",
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
    max_words = 120 if expand_deep else 80
    grade_label = _GRADE_LABELS.get(class_level, class_level or "School student")
    complexity = _GRADE_COMPLEXITY.get(class_level, "Use clear, age-appropriate spoken language.")
    class_band = _CLASS_BAND_RULES.get(_class_band(class_level), _CLASS_BAND_RULES["6-8"])
    display_name = _student_first_name(student_name)
    scores = understanding or UnderstandingScores()
    learner_hint = learner.to_prompt_hint() if learner else ""

    expand_policy = (
        "The student asked for more detail — you may use up to 120 words."
        if expand_deep
        else "Keep this turn under 80 words unless the student explicitly asked for more."
    )

    system = VOICE_SYSTEM_PROMPT.format(
        student_name=display_name,
        grade_label=grade_label,
        board=board or "General",
        subject=subject_name or "General",
        chapter=chapter or "Current chapter",
        complexity_rule=complexity,
        class_band_rule=class_band,
        max_words=max_words,
        tutor_state=tutor_state.value,
        state_guidance=_state_guidance(tutor_state),
        understanding_guidance=scores.to_hint(),
        learner_guidance=learner_hint,
        expand_policy=expand_policy,
    )
    user = VOICE_USER_TEMPLATE.format(
        context=context or "(No chapter excerpt — use accurate general knowledge briefly.)",
        question=query,
        max_words=max_words,
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in _normalize_history(conversation_history):
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
