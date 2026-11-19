"""
Lightweight interrupt-intent detector (phrase match + keyword score).

Phrases that should barge-in immediately without waiting for a full question.
"""

from __future__ import annotations

import re

# Explicit interrupt / attention phrases (matched after normalize).
INTERRUPT_PHRASES: tuple[str, ...] = (
    "stop",
    "wait",
    "hold on",
    "one second",
    "excuse me",
    "i have a doubt",
    "i have doubt",
    "hey",
    "hi",
    "hello",
)

WAKE_PHRASES_DEFAULT: tuple[str, ...] = (
    "hey tutor",
    "teacher",
)

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_intent_text(text: str) -> str:
    t = (text or "").lower()
    t = _PUNCT.sub(" ", t)
    return _WHITESPACE.sub(" ", t).strip()


def detect_interrupt_intent(
    transcript: str,
    *,
    wake_word_enabled: bool = False,
    wake_words: list[str] | None = None,
) -> dict[str, object]:
    """
    Returns:
      is_interrupt_intent: bool
      matched_phrase: str | None
      kind: "interrupt" | "wake" | None
      confidence: float
    """
    norm = normalize_intent_text(transcript)
    if not norm:
        return {
            "is_interrupt_intent": False,
            "matched_phrase": None,
            "kind": None,
            "confidence": 0.0,
        }

    # Multi-word wake phrases before short greets ("hey" vs "hey tutor").
    if wake_word_enabled:
        words = wake_words or list(WAKE_PHRASES_DEFAULT)
        for phrase in sorted(words, key=lambda w: -len(w)):
            p = normalize_intent_text(phrase)
            if not p:
                continue
            if norm == p or norm.startswith(p + " ") or f" {p} " in f" {norm} ":
                return {
                    "is_interrupt_intent": True,
                    "matched_phrase": p,
                    "kind": "wake",
                    "confidence": 0.9,
                }

    for phrase in sorted(INTERRUPT_PHRASES, key=lambda w: -len(w)):
        if norm == phrase or norm.startswith(phrase + " ") or f" {phrase} " in f" {norm} ":
            if norm == phrase or len(norm.split()) <= 6 or norm.startswith(phrase):
                return {
                    "is_interrupt_intent": True,
                    "matched_phrase": phrase,
                    "kind": "interrupt",
                    "confidence": 0.95 if norm == phrase else 0.85,
                }

    return {
        "is_interrupt_intent": False,
        "matched_phrase": None,
        "kind": None,
        "confidence": 0.0,
    }
