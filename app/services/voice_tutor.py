"""
Voice Tutor mode — state machine and understanding heuristics.

All subjects use conversational live-teaching prompts in voice mode.
Full text-chat format is reserved for explicit “full solution” or
“explain in detail” requests (see voice_wants_full_written_answer).

Prompt templates live in voice_prompts.py (Phase 5).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.services.voice_prompts import (
    VOICE_SYSTEM_PROMPT,
    _WORD_RE,
    build_voice_system_prompt,
    build_voice_user_message,
    voice_grade_band,
)

logger = logging.getLogger(__name__)

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


class ReplyIntent(str, Enum):
    """Classification of the student's last turn (conversation state rules)."""

    NEW_QUESTION = "NEW_QUESTION"
    WRONG_ANSWER = "WRONG_ANSWER"
    DONT_KNOW = "DONT_KNOW"
    CLOSING = "CLOSING"
    UNCLEAR = "UNCLEAR"


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
    r"no\s+idea|no\s+clue|(?:don'?t|dont)\s+know|not\s+sure|"
    r"i\s+don'?t\s+get|still\s+confused|too\s+hard)\b",
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
_CLOSING = re.compile(
    r"\b(thank\s*you|thanks|thank\s*u|okay?\s*,?\s*got\s+it|i\s+understand\s+now|"
    r"that'?s\s+all|i'?m\s+done|no\s+more\s+questions?|bye|goodbye|see\s+you)\b",
    re.I,
)

# Semantic fallback for confusion/affirmation the regex lists above miss —
# same local BGE stack (and score thresholds) as conversation_intent_classifier's
# BGE fallback, so no LLM call and no new dependency. Only runs when the
# embedding model is already warm (never forces a cold load just for this).
_UNDERSTANDING_BGE_MIN_SCORE = 38.0
_UNDERSTANDING_BGE_MARGIN = 2.5
_CONFUSION_PROTOTYPES = [
    "I don't know",
    "I'm not sure",
    "I have no idea",
    "I'm confused",
    "I don't understand this",
    "I have no clue",
    "I'm lost",
    "not sure about that",
    "can you explain that again",
    "I don't get it",
]
_AFFIRM_PROTOTYPES = [
    "yes",
    "I understand",
    "that makes sense",
    "got it",
    "okay I see",
    "sure, that's clear",
    "understood",
    "makes sense now",
    "I get it now",
    "that's correct",
]
_understanding_prototype_cache: dict[str, list[Any]] | None = None


def _load_understanding_prototype_embeddings() -> dict[str, list[Any]] | None:
    global _understanding_prototype_cache
    if _understanding_prototype_cache is not None:
        return _understanding_prototype_cache

    from app.services.image_service.figure_context_bge import embed_texts

    cache: dict[str, list[Any]] = {}
    for label, phrases in (("confusion", _CONFUSION_PROTOTYPES), ("affirm", _AFFIRM_PROTOTYPES)):
        vectors = [v for v in embed_texts(phrases) if v is not None]
        if vectors:
            cache[label] = vectors
    _understanding_prototype_cache = cache if cache else {}
    return _understanding_prototype_cache or None


def _classify_understanding_by_bge(text: str) -> str | None:
    """Return 'confusion' / 'affirm' via semantic similarity, or None when
    unavailable/ambiguous. Regex stays authoritative — this only runs on
    short replies neither regex list matched (see evaluate_student_response)."""
    from app.services.image_service.figure_context_bge import cosine_100, embed_query
    from app.services.vector_service import is_embedding_model_loaded

    if not is_embedding_model_loaded():
        return None
    prototypes = _load_understanding_prototype_embeddings()
    if not prototypes:
        return None
    qvec = embed_query(text[:200])
    if qvec is None:
        return None

    best_label: str | None = None
    best_score = 0.0
    second_score = 0.0
    for label, vectors in prototypes.items():
        score = max(cosine_100(qvec, pvec) for pvec in vectors)
        if score > best_score:
            best_label, second_score, best_score = label, best_score, score
        elif score > second_score:
            second_score = score

    if best_label is None or best_score < _UNDERSTANDING_BGE_MIN_SCORE:
        return None
    if (best_score - second_score) < _UNDERSTANDING_BGE_MARGIN:
        return None
    return best_label
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
                "then teach the next small step on the same topic. "
                "Do NOT quote or recap their earlier questions (for example 'your last question was…') "
                "unless they explicitly asked you to repeat their question."
            )
        if self.wants_expansion:
            return (
                "The student asked for more detail. You may use up to 120 words, "
                "but keep 6–12 word sentences, pause every 1–2 lines, example-led."
            )
        if self.wants_quiz:
            return "Ask ONE short quiz question orally. Wait for their answer next turn."
        return (
            "Use short spoken lines (6–12 words). Answer naturally like a helpful voice AI — "
            "no scripted labels. Do not use the student's name."
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


def topic_key(text: str, *, max_terms: int = 6) -> str:
    """Short normalized label for a topic taught this session (explained_points).
    Stopword-filtered so "can you tell me about X" and "about Y" don't look
    alike just because they share the same question scaffolding."""
    from app.services.chapter_scope import substantive_query_terms

    return " ".join(sorted(substantive_query_terms(text))[:max_terms])


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

    # Regex only catches phrasings someone thought to list — for a short
    # reply that matched neither list, fall back to semantic similarity
    # against the same prototypes ("I don't know", "makes sense", ...) so
    # "I have zero idea" / "beats me" aren't silently scored as neutral.
    if not is_affirm and not confusion_hit and words <= 10:
        bge_label = _classify_understanding_by_bge(q)
        if bge_label == "confusion":
            confusion_hit = True
        elif bge_label == "affirm":
            is_affirm = True

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


def classify_reply_intent(student_text: str, *, quiz_pending: bool) -> ReplyIntent:
    """Zero-latency heuristic classification of the student's last turn —
    NEW_QUESTION / WRONG_ANSWER / DONT_KNOW / CLOSING / UNCLEAR.

    Deliberately regex-only, no BGE semantic fallback: this gates a discrete
    behavioral fork (never say "wrong" vs. hand over a hint), and the BGE
    prototypes don't cleanly separate a hedged wrong guess ("I think it's a
    bird") from real uncertainty ("I don't know") — both share hedging
    language and score close together. evaluate_student_response's softer,
    continuous confusion nudge can afford that fuzziness; this branch can't."""
    q = (student_text or "").strip()
    if not q or not re.search(r"[a-zA-Z0-9]", q):
        return ReplyIntent.UNCLEAR
    if _CLOSING.search(q):
        return ReplyIntent.CLOSING

    if _CONFUSION.search(q):
        return ReplyIntent.DONT_KNOW

    if quiz_pending and not _AFFIRM.match(q):
        return ReplyIntent.WRONG_ANSWER

    return ReplyIntent.NEW_QUESTION


def update_quiz_state(
    *,
    quiz_pending: bool,
    quiz_attempts: int,
    reply_intent: ReplyIntent,
    assistant_reply: str,
) -> tuple[bool, int]:
    """Rules 4 & 6: a pending quiz question is only cleared by a correct
    answer or an explicit reveal — the reveal is forced (via
    reply_intent_guidance in voice_prompts.py) once attempts reach 2, so
    reaching that count here always means the answer was just revealed.
    Returns (new_quiz_pending, new_quiz_attempts)."""
    reply = (assistant_reply or "").strip()
    if quiz_pending:
        if reply_intent in (ReplyIntent.WRONG_ANSWER, ReplyIntent.DONT_KNOW):
            attempts = quiz_attempts + 1
            if attempts >= 2:
                return False, 0  # forced reveal happened this turn — resolved
            return True, attempts
        if not reply.endswith("?"):
            return False, 0  # assistant moved on — treat as resolved
        return True, quiz_attempts  # still probing the same question
    # No quiz was pending — did this turn's reply pose a new one?
    return (True, 0) if reply.endswith("?") else (False, 0)


_UNDERSTANDING_LLM_TIMEOUT_SEC = 6.0
_UNDERSTANDING_LLM_SYSTEM_PROMPT = (
    "You classify a student's spoken reply during a live tutoring session. "
    "Read the tutor's last message and the student's reply, then reply with "
    "ONLY compact JSON — no prose, no markdown fences — in exactly this shape: "
    '{"understanding": 0.0-1.0, "confusion": 0.0-1.0, "is_affirmation": true|false, '
    '"wants_expansion": true|false, "wants_quiz": true|false}\n'
    "understanding: how well they seem to grasp the material, judged from what "
    "they actually said (not reply length).\n"
    "confusion: how lost/unsure/stuck they seem — include indirect ways of "
    "saying they don't know (deflection, guessing, sarcasm, trailing off).\n"
    "is_affirmation: true only if they are confirming understanding or "
    "agreement, not just being polite.\n"
    "wants_expansion: true if they are asking for more depth or detail.\n"
    "wants_quiz: true if they are asking to be tested or quizzed."
)


def _parse_understanding_json(raw: str) -> UnderstandingScores | None:
    import json

    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None

    def _clamp01(v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    understanding = _clamp01(data.get("understanding", 0.5))
    confusion = _clamp01(data.get("confusion", 0.15))
    return UnderstandingScores(
        understanding=understanding,
        confidence=min(0.95, understanding + 0.1),
        confusion=confusion,
        is_affirmation=bool(data.get("is_affirmation", False)),
        wants_expansion=bool(data.get("wants_expansion", False)),
        wants_quiz=bool(data.get("wants_quiz", False)),
    )


async def classify_understanding_llm(
    student_text: str,
    *,
    last_assistant: str = "",
) -> UnderstandingScores | None:
    """LLM-based understanding classification — the accurate-but-slower
    counterpart to evaluate_student_response's zero-latency heuristic.

    Meant to run as a background task started alongside the main answer
    pipeline (see voice_ws._stream_answer): by the time the full answer has
    been generated and spoken, this is almost always done, so the tutor
    state machine can use it for *this* turn's transition — with zero added
    time-to-first-token, since nothing on the critical path ever awaits it.
    Returns None on any failure so callers always have the heuristic to
    fall back to.
    """
    import asyncio

    from app.services import llm_client

    q = (student_text or "").strip()
    if not q:
        return None
    messages = [
        {"role": "system", "content": _UNDERSTANDING_LLM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Tutor's last message: {last_assistant.strip() or '(none — start of session)'}\n"
                f"Student's reply: {q}"
            ),
        },
    ]
    try:
        raw = await asyncio.wait_for(
            llm_client.complete(messages, feature="voice", max_tokens=80),
            timeout=_UNDERSTANDING_LLM_TIMEOUT_SEC,
        )
    except Exception as exc:
        logger.debug("Understanding LLM classification failed: %s", exc)
        return None
    return _parse_understanding_json(raw)


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
            "Teach one small idea in natural spoken English — like a helpful voice AI. "
            "6–12 word sentences, contractions. Do not say the student's name. "
            "Pause after every 1–2 sentences. Optional check-in only if it helps."
        ),
        TutorState.CHECKING_UNDERSTANDING: (
            "You asked a question last turn. Echo their answer briefly, "
            "say if they're on track, then ONE short hint or follow-up. "
            "Use their name only if celebrating a correct answer."
        ),
        TutorState.QUIZING: "One short oral quiz question. Don't explain at length. No name.",
        TutorState.CLARIFYING: (
            "They're stuck — you MAY say their first name once for reassurance. "
            "Simpler words, one 'Imagine…' example, two short sentences max."
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
    quiz_pending: bool = False,
    quiz_question: str = "",
    quiz_attempts: int = 0,
    explained_points: list[str] | None = None,
    reply_intent: ReplyIntent | None = None,
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
        reply_intent=reply_intent or classify_reply_intent(query, quiz_pending=quiz_pending),
        quiz_pending=quiz_pending,
        quiz_question=quiz_question,
        quiz_attempts=quiz_attempts,
        explained_points=explained_points,
    )
    user = build_voice_user_message(query, context, expand_deep=expand_deep)
    from app.services.chat_service import _prepare_math_engine_block

    math_block = _prepare_math_engine_block(
        query,
        subject_name=subject_name,
        class_level=class_level,
        context=context,
    )
    if math_block:
        user = math_block + "\n\n" + user

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
