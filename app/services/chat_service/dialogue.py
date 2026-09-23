"""Dialogue-act helpers for greetings, affirmations, and continue offers."""
from __future__ import annotations

import re
from typing import Any

logger = __import__("logging").getLogger(__name__)


_AFFIRMATION_RE = re.compile(
    r"^(?:yes|yeah|yep|yup|ok|okay|sure|right|correct|exactly|got\s+it|"
    r"(?:i\s+)?(?:understand|understood)|makes\s+sense|that\s+helps|"
    r"clear\s+now|sounds\s+good|"
    # "Good thanks" / "thanks" / "thank you" mid-lesson = understood, not a fresh greeting
    r"(?:good|great|cool|nice|awesome)?\s*(?:thanks|thank\s+you|thx)|"
    r"appreciate\s+(?:it|that)|that'?s\s+(?:helpful|great|good)"
    r")(?:\s+.*)?[.!?]*$",
    re.I,
)


_EXPAND_REQUEST_RE = re.compile(
    r"\b(?:"
    r"explain(?:\s+more|\s+please)?|please\s+explain|explore|elaborate|"
    r"tell\s+me(?:\s+more)?|go\s+(?:on|ahead)|continue|carry\s+on|"
    r"next(?:\s+part)?|deeper|more\s+(?:about|detail|on)|"
    r"how\s+(?:it|they|one)\s+affects?"
    r")\b",
    re.I,
)


_TUTOR_CONTINUE_OFFER_RE = re.compile(
    r"would you like to\s+(?:explore|learn|look|see|try|continue|explain|"
    r"dive|understand|know|hear|discuss|go)|"
    r"want (?:me )?to\s+(?:explain|explore|show|teach|continue)|"
    r"shall (?:we|i)\s+(?:explore|look|continue|explain)",
    re.I,
)


_MULTI_CHOICE_MENU_RE = re.compile(
    r"(?:real[- ]?life\s+example|quick\s+quiz|explore\s+other\s+topics|"
    r"example,\s*(?:a\s+)?(?:quick\s+)?quiz|,\s*or\s+(?:to\s+learn|explore))",
    re.I,
)


_PERSONAL_EXAMPLE_RE = re.compile(
    r"\b(i was|it was|when i|where i|one day|one time|suddenly|yesterday|last\s+(?:week|month|year)|"
    r"started\s+(?:rain|snow)|raining|sunny|cloudy|snowing|storm)\b",
    re.I,
)


_SESSION_GREETING_RE = re.compile(
    r"(?:"
    r"\b(?:welcome back|i(?:'m| am) your ai tutor)\b.*\bwhat would you like to learn\b"
    r"|"
    r"\bhere are a few things you can ask me\b"
    r"|"
    r"\bpick one, or type your own question\b"
    r")",
    re.I | re.S,
)


_ACADEMIC_QUESTION_RE = re.compile(
    r"\b("
    r"how (?:far|long|many|much)|can (?:you|we|i) reach|what is|what are|explain|define|"
    r"solve|calculate|find|prove|distance|travel|moon|speed|rate|per day|per hour|"
    r"equation|formula|\d+\s*(?:km|m|cm|mm|years?|days?|hours?|minutes?|%)"
    r")\b",
    re.I,
)


def _is_session_greeting_message(text: str) -> bool:
    return bool(_SESSION_GREETING_RE.search((text or "").strip()))


def _last_assistant_text(conversation_history: list[dict] | None) -> str:
    if not conversation_history:
        return ""
    for turn in reversed(conversation_history):
        if (turn.get("role") or "").lower() == "assistant":
            return (turn.get("content") or "").strip()
    return ""


def _is_personal_dialogue_response(
    query: str, conversation_history: list[dict] | None
) -> bool:
    """Student answered the tutor's question with an example or personal experience."""
    q = (query or "").strip()
    if not q or not conversation_history:
        return False
    if _ACADEMIC_QUESTION_RE.search(q):
        return False
    last_asst = _last_assistant_text(conversation_history)
    if not last_asst or "?" not in last_asst:
        return False
    if _is_session_greeting_message(last_asst):
        return False
    if _PERSONAL_EXAMPLE_RE.search(q):
        return True
    words = q.split()
    if "," in q and len(words) >= 5 and _PERSONAL_EXAMPLE_RE.search(q):
        return True
    return False


def _last_assistant_offered_to_continue(
    conversation_history: list[dict] | None,
) -> bool:
    """True when the tutor asked a single yes/no-style 'shall we explore…?' offer."""
    last = _last_assistant_text(conversation_history)
    if not last or "?" not in last:
        return False
    if _MULTI_CHOICE_MENU_RE.search(last):
        return False
    return bool(_TUTOR_CONTINUE_OFFER_RE.search(last))


def _is_accepting_tutor_continue_offer(
    query: str, conversation_history: list[dict] | None
) -> bool:
    """Student accepted the tutor's offer to keep explaining (e.g. 'Yes please explain')."""
    q = (query or "").strip()
    if not q or not conversation_history:
        return False
    if _EXPAND_REQUEST_RE.search(q):
        return bool(_last_assistant_text(conversation_history))
    if not _last_assistant_offered_to_continue(conversation_history):
        return False
    if not _AFFIRMATION_RE.match(q):
        return False
    return len(q.split()) <= 7


def _is_affirmation_followup(query: str, conversation_history: list[dict] | None) -> bool:
    """True when the student gives a short bare acknowledgment (not a story or answer)."""
    if not conversation_history:
        return False
    q = (query or "").strip()
    if not q or not _AFFIRMATION_RE.match(q):
        return False
    if not _last_assistant_text(conversation_history):
        return False
    # "Yes please explain" / accepting an explore offer → keep teaching
    if _EXPAND_REQUEST_RE.search(q):
        return False
    if _is_accepting_tutor_continue_offer(q, conversation_history):
        return False
    words = q.split()
    if len(words) > 7:
        return False
    if "," in q and len(words) > 4:
        return False
    if _PERSONAL_EXAMPLE_RE.search(q):
        return False
    return True


def _skip_answer_expansion(
    query: str, conversation_history: list[dict] | None
) -> bool:
    from app.services.conversation_context import is_clarification_followup

    return (
        _is_affirmation_followup(query, conversation_history)
        or _is_personal_dialogue_response(query, conversation_history)
        or is_clarification_followup(query, conversation_history)
    )

