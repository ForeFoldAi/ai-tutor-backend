"""Dialogue-act helpers for voice — routing only; LLM writes the reply.

Bare acks / closing / intro must not retrieve textbook chunks.
Keep in sync with Voice/backend reply-intent.ts.
"""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[()[\]{}.,!?\"'`~]+")
_SPACE = re.compile(r"\s+")

_ACK = re.compile(
    r"^(?:"
    r"laughing|laughs|laughter|haha+|ha\s+ha|hehe+|lol|lmao|"
    r"okay|ok|yes|yeah|yep|yup|sure|right|"
    r"hmm+|mm+|mhm+|uh-?huh|"
    r"wow|whoa|interesting|nice|cool|"
    r"i\s+see|got\s+it|makes\s+sense|i\s+understand|understood"
    r")$",
    re.I,
)

_EDU = re.compile(
    r"\b("
    r"tell\s+me\s+more|explain|why|what|who|where|when|how|"
    r"continue|go\s+on|more\s+about|did\s+it|does\s+it|"
    r"quiz|simplify|"
    r"(?:give|another|an|more|some)\s+examples?|examples?\s+(?:of|please|from)"
    r")\b",
    re.I,
)

_EXAMPLE_PRAISE = re.compile(
    r"^(?:(?:ok|okay|yes|yeah|yep|sure)\s+)?(?:a\s+)?"
    r"(?:nice|good|great|cool|lovely)\s+example"
    r"(?:\s+(?:thanks|thank\s+you))?$",
    re.I,
)

# Student understood / liked the explanation — keep ≤16 words (Nest side).
_UNDERSTANDING_PRAISE = re.compile(
    r"\b(?:"
    r"(?:really\s+|very\s+|so\s+)?(?:good|great|nice|clear|helpful|excellent|awesome|wonderful|perfect)"
    r"\s+(?:explanation|example|teaching|job|one)|"
    r"well\s+explained|explained\s+(?:it\s+)?well|"
    r"(?:i\s+)?(?:really\s+)?(?:understood?|got\s+it|understand)"
    r"(?:\s+(?:it|that|this|everything))?(?:\s+(?:very\s+well|clearly|now|fully|better))?|"
    r"(?:it\s+)?(?:was\s+|is\s+)?(?:very\s+|really\s+)?(?:clear|helpful)(?:\s+(?:for\s+me|to\s+me))?|"
    r"makes\s+(?:perfect\s+|more\s+)?sense|"
    r"that\s+helped(?:\s+(?:a\s+lot|me|a\s+ton))?|"
    r"crystal\s+clear|now\s+i\s+(?:get|understand)\s+it|i\s+get\s+it\s+now"
    r")\b",
    re.I,
)

_CLOSING = re.compile(
    r"\b(thank\s*you|thanks|thank\s*u|okay?\s*,?\s*got\s+it|i\s+understand\s+now|"
    r"that'?s\s+all|i'?m\s+done|no\s+more\s+questions?|bye|goodbye|see\s+you|"
    r"have\s+(?:some\s+)?work|gotta\s+go|got\s+to\s+go)\b",
    re.I,
)

_PERSONAL_INTRO = re.compile(
    r"^(?:(?:hi|hello|hey)[,!]?\s+)?"
    r"(?:i(?:'m|\s+am)|my\s+name\s+is|this\s+is)\s+"
    r".{1,80}$",
    re.I,
)

_DIALOGUE_ACTS = frozenset({"closing", "ack", "intro"})


def _norm(text: str) -> str:
    t = _PUNCT.sub(" ", (text or "").strip().lower())
    return _SPACE.sub(" ", t).strip()


def is_understanding_praise(text: str) -> bool:
    q = _norm(text)
    if not q or _EDU.search(q):
        return False
    if len(q.split()) > 16:
        return False
    return bool(_UNDERSTANDING_PRAISE.search(q))


def is_voice_acknowledgement(text: str) -> bool:
    """True when the student is reacting, not asking to be taught."""
    q = _norm(text)
    if not q:
        return False
    if _EXAMPLE_PRAISE.match(q):
        return True
    if is_understanding_praise(q):
        return True
    if _EDU.search(q):
        return False
    return bool(_ACK.match(q))


def is_personal_intro(text: str) -> bool:
    q = (text or "").strip()
    return bool(q) and len(q) <= 80 and bool(_PERSONAL_INTRO.match(q))


def resolve_dialogue_act(
    query: str,
    *,
    dialogue_act: str | None = None,
    quiz_pending: bool = False,
) -> str | None:
    """Return closing|ack|intro when chapter RAG should be skipped."""
    raw = (dialogue_act or "").strip().lower()
    if raw in _DIALOGUE_ACTS:
        return raw
    q = (query or "").strip()
    if not q:
        return None
    if _CLOSING.search(q):
        return "closing"
    if is_personal_intro(q):
        return "intro"
    if is_voice_acknowledgement(q):
        # Bare yes/ok while a check is open → Nest EVALUATE path; still retrieve.
        if quiz_pending and re.fullmatch(
            r"(?:yes|yeah|yep|yup|ok|okay|sure|right|correct|exactly)",
            _norm(q),
            re.I,
        ):
            return None
        return "ack"
    return None


def voice_ack_reply(text: str) -> str:
    """Fallback only — LLM-first path should not use this for live voice."""
    q = _norm(text)
    if re.search(r"laugh|haha|hehe|lol", q):
        return "Glad you're enjoying it!"
    if re.search(r"wow|whoa|interesting|nice|cool|good\s+explanation|clear", q):
        return "Glad you found that helpful."
    return "Alright."
