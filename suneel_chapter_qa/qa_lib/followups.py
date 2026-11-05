"""Build one student-style follow-up from the tutor's previous answer."""
from __future__ import annotations

import re

_TEMPLATES = (
    "hold on, what do you mean by {x}?",
    "ok but can you explain {x} in simpler words?",
    "is {x} actually in this chapter or are you guessing?",
    "quiz me with one easy question just about {x}",
    "show me a figure or example related to {x}",
    "wait why is {x} important here?",
    "can you give one more real-life example of {x}?",
    "i still don't get {x} — say it again slowly",
    "ok so what should i remember about {x} for a test?",
    "that part about {x} sounded confusing — did i hear it right?",
)

_GENERIC = (
    "can you explain that again more simply?",
    "can you give me one more example?",
    "quiz me on that with one easy question",
)

_BOLD = re.compile(r"\*\*([^*\n]{1,40})\*\*")
_NUMBER_EXPR = re.compile(r"\b\d+(?:\s*[×x*+\-/=]\s*\d+)+(?:\s*=\s*\d+)?|\b\d+\b")
_CAP_WORD = re.compile(r"\b[A-Z][a-z]{3,}\b")
_SKIP = frozenset({"This", "That", "There", "These", "Those", "When", "What", "Here", "Think", "Like"})


def _hooks(answer: str) -> list[str]:
    text = answer or ""
    out: list[str] = []
    for m in _BOLD.finditer(text):
        h = m.group(1).strip(" :.")
        if h and h not in out:
            out.append(h)
    for m in _NUMBER_EXPR.finditer(text):
        h = m.group(0).strip()
        if h and h not in out:
            out.append(h)
    for m in _CAP_WORD.finditer(text):
        h = m.group(0)
        if h not in _SKIP and h not in out:
            out.append(h)
    return out


def followup_from_answer(answer: str, index: int, label: str = "normal") -> tuple[str, bool]:
    """Return (follow-up text, derived_from_answer)."""
    hooks = _hooks(answer)
    if not hooks:
        return _GENERIC[index % len(_GENERIC)], False
    hook = hooks[index % len(hooks)]
    return _TEMPLATES[index % len(_TEMPLATES)].format(x=hook), True
