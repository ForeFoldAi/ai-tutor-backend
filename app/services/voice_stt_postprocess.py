"""
Lightweight speech-to-text cleanup for voice tutoring.

Browser Web Speech API often mishears subject terms (especially math/science).
This normalizes common mistakes before the LLM sees the transcript.
"""

from __future__ import annotations

import re

_PHRASE_FIXES: list[tuple[str, str]] = [
    (r"\bphoto\s+synthesis\b", "photosynthesis"),
    (r"\bphoto\s+synesis\b", "photosynthesis"),
    (r"\bchloro\s+plast(s)?\b", r"chloroplast\1"),
    (r"\bmulti\s+plication\b", "multiplication"),
    (r"\bmultiply\s+cation\b", "multiplication"),
    (r"\bdivi\s+sion\b", "division"),
    (r"\bex\s+squared\b", "x squared"),
    (r"\bwhy\s+squared\b", "y squared"),
    (r"\bex\s+cubed\b", "x cubed"),
    (r"\bintegrate?\s+al\b", "integral"),
    (r"\bderiva\s+tive\b", "derivative"),
    (r"\bmito\s+chondria\b", "mitochondria"),
    (r"\bnu\s+cleus\b", "nucleus"),
    (r"\borgan\s+elle\b", "organelle"),
]

_WORD_FIXES: dict[str, str] = {
    "photosynthisis": "photosynthesis",
    "photosynthsis": "photosynthesis",
    "cloroplast": "chloroplast",
    "multipication": "multiplication",
    "eqation": "equation",
    "denomenator": "denominator",
    "numirator": "numerator",
    "mitocondria": "mitochondria",
    "mitachondria": "mitochondria",
}

_WHITESPACE = re.compile(r"\s+")
_FILLER = re.compile(
    r"\b(um|uh|uhh|umm|like|you know|i mean|so|well|actually|basically|right|okay|ok|yeah|yes|hmm|ah|oh)\b",
    re.I,
)


def _normalize_echo_text(text: str) -> str:
    t = (text or "").lower()
    # ponytail: normalize a few common contractions so overlap scoring
    # doesn't miss "you're" vs "you are" speaker-echo cases.
    t = re.sub(r"\byou'?re\b", "you are", t)
    t = re.sub(r"\bi'm\b", "i am", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = _FILLER.sub(" ", t)
    return _WHITESPACE.sub(" ", t).strip()


def is_sentence_prefix_echo(user_text: str, assistant_text: str) -> bool:
    """True when the transcript is the start of an AI sentence (greeting bleed)."""
    user = _normalize_echo_text(user_text)
    words = user.split()
    if not (2 <= len(words) <= 6):
        return False
    for raw in re.split(r"[.!?]+", assistant_text or ""):
        sent = _normalize_echo_text(raw)
        if sent == user or sent.startswith(user + " "):
            return True
    return False


def echo_similarity(user_text: str, assistant_text: str) -> float:
    """
    Combined containment + word Jaccard/overlap for echo rejection.
    Biased to catch AI self-echo without rejecting short student questions.
    """
    user = _normalize_echo_text(user_text)
    assistant = _normalize_echo_text(assistant_text)
    if len(user) < 3 or not assistant:
        return 0.0
    if is_sentence_prefix_echo(user_text, assistant_text):
        return 1.0
    u_words = [w for w in user.split() if len(w) > 2]
    # ponytail: follow-up questions often appear inside the tutor's last answer;
    # lowered from 6/48 → 4/28 so partial TTS echoes are caught.
    if assistant.find(user) >= 0 and (len(u_words) >= 4 or len(user) >= 28):
        return 1.0
    # Sliding window containment of user inside assistant (AI bleed)
    if len(user) >= 28:
        step = max(4, len(user) // 6)
        for i in range(0, max(1, len(assistant) - len(user) + 1), step):
            window = assistant[i : i + len(user) + 8]
            if user in window:
                return 0.98

    probe = assistant[: min(120, len(assistant))]
    if len(probe) >= 24 and user.find(probe) >= 0:
        return 1.0

    if not u_words:
        return 0.0
    # ponytail: brief questions share topic words with the tutor; substring match only
    # lowered from 4 → 3 so 4-5 word echoes are also caught by overlap scoring.
    if len(u_words) <= 3:
        return 0.0
    a_list = [w for w in assistant.split() if len(w) > 2]
    # Use prefix matching (len≥5 stem) to catch "dynasty"↔"dynasties", "shape"↔"shaping"
    a_set = set(a_list)
    a_stems = {w[:6] for w in a_list if len(w) >= 6}
    overlap = sum(1 for w in u_words if w in a_set or (len(w) >= 6 and w[:6] in a_stems))
    recall = overlap / len(u_words)
    # Soft Jaccard against recent assistant word set
    union = len(set(u_words) | a_set) or 1
    jaccard = len(set(u_words) & a_set) / union
    # Sequence-ish: fraction of consecutive bigrams shared
    u_bi = {f"{u_words[i]} {u_words[i + 1]}" for i in range(len(u_words) - 1)}
    a_bi = {f"{a_list[i]} {a_list[i + 1]}" for i in range(len(a_list) - 1)}
    bi = (len(u_bi & a_bi) / len(u_bi)) if u_bi else 0.0
    return max(recall, 0.55 * recall + 0.25 * jaccard + 0.20 * bi)


# Cut-off STT: "what would", "can you tell" — not a complete question.
_INCOMPLETE_QUESTION = re.compile(
    r"^(?:"
    r"what\s+(?:would|is|are|was|were|will|do|does|did|can|could|should)|"
    r"who\s+(?:is|are|was|were)|"
    r"where\s+(?:is|are|was|did)|"
    r"when\s+(?:is|did|was)|"
    r"how\s+(?:did|does|do|is|are|was|can)|"
    r"can you(?:\s+tell(?:\s+me(?:\s+about)?)?)?|"
    r"could you|"
    r"would you|"
    r"tell me(?:\s+about)?|"
    r"do you know"
    r")\s*[.?!]*$",
    re.I,
)

INCOMPLETE_UTTERANCE_REPLY = (
    "Sorry, I only caught part of that. Can you say the full question?"
)


def is_incomplete_voice_utterance(text: str) -> bool:
    """True when STT likely cut off a question before the topic."""
    q = (text or "").strip()
    return bool(q) and bool(_INCOMPLETE_QUESTION.match(q))


def transcript_likely_echo(
    user_text: str,
    assistant_text: str,
    *,
    threshold: float = 0.7,
) -> bool:
    """True when user transcript mostly repeats recent tutor speech (mic echo)."""
    if not (user_text or "").strip() or not (assistant_text or "").strip():
        return False
    return echo_similarity(user_text, assistant_text) > threshold


def postprocess_voice_transcript(text: str, *, subject_name: str = "") -> str:
    """Normalize a browser STT transcript for tutoring."""
    raw = (text or "").strip()
    if not raw:
        return ""

    out = raw
    for pattern, repl in _PHRASE_FIXES:
        out = re.sub(pattern, repl, out, flags=re.I)

    subj = (subject_name or "").lower()
    if "math" in subj:
        out = re.sub(r"\binto\b", "times", out, flags=re.I)

    words = out.split()
    fixed: list[str] = []
    for w in words:
        core = re.sub(r"^[^\w]+|[^\w]+$", "", w)
        lower = core.lower()
        if lower in _WORD_FIXES:
            fixed.append(w.lower().replace(lower, _WORD_FIXES[lower], 1))
        else:
            fixed.append(w)

    return _WHITESPACE.sub(" ", " ".join(fixed)).strip()
