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
    t = re.sub(r"[^\w\s]", " ", t)
    t = _FILLER.sub(" ", t)
    return _WHITESPACE.sub(" ", t).strip()


def echo_similarity(user_text: str, assistant_text: str) -> float:
    """
    Combined containment + word Jaccard/overlap for echo rejection.
    Biased to catch AI self-echo without rejecting short student questions.
    """
    user = _normalize_echo_text(user_text)
    assistant = _normalize_echo_text(assistant_text)
    if len(user) < 3 or not assistant:
        return 0.0
    if assistant.find(user) >= 0:
        return 1.0
    # Sliding window containment of user inside assistant (AI bleed)
    if len(user) >= 12:
        step = max(4, len(user) // 6)
        for i in range(0, max(1, len(assistant) - len(user) + 1), step):
            window = assistant[i : i + len(user) + 8]
            if user in window:
                return 0.98

    probe = assistant[: min(120, len(assistant))]
    if len(probe) >= 12 and user.find(probe) >= 0:
        return 1.0

    u_words = [w for w in user.split() if len(w) > 2]
    if not u_words:
        return 0.0
    # ponytail: brief questions share topic words with the tutor; substring match only
    if len(u_words) <= 4:
        return 0.0
    a_list = [w for w in assistant.split() if len(w) > 2]
    a_set = set(a_list)
    overlap = sum(1 for w in u_words if w in a_set)
    recall = overlap / len(u_words)
    # Soft Jaccard against recent assistant word set
    union = len(set(u_words) | a_set) or 1
    jaccard = len(set(u_words) & a_set) / union
    # Sequence-ish: fraction of consecutive bigrams shared
    u_bi = {f"{u_words[i]} {u_words[i + 1]}" for i in range(len(u_words) - 1)}
    a_bi = {f"{a_list[i]} {a_list[i + 1]}" for i in range(len(a_list) - 1)}
    bi = (len(u_bi & a_bi) / len(u_bi)) if u_bi else 0.0
    return max(recall, 0.55 * recall + 0.25 * jaccard + 0.20 * bi)


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
