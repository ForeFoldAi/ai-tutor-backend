"""
TTS-only pronunciation (Communicate path only).

Why not SSML <phoneme>: edge-tts escapes text, and injecting tags inside its
<prosody> wrapper returns no audio from the Microsoft service.

Why not "whind": Neerja reads it like "whined" (/waɪnd/) — a different word —
in AI Voice and text-chat read-aloud.

Air /wɪnd/: respell as "wihnd" (ih = short i, not a real English word).
Verb /waɪnd/: leave "wind" (already the clock/rope sense).

Student UI / RAG / LLM text are never changed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.config import VOICE_DEBUG_LOGGING

logger = logging.getLogger(__name__)

_AIR = {"wind": "wihnd", "winds": "wihnds"}


@dataclass(frozen=True)
class PronunciationRule:
    word: str
    meaning: str
    spoken: str


PRONUNCIATION_RULES: tuple[PronunciationRule, ...] = (
    PronunciationRule("wind", "air", "wihnd"),
    PronunciationRule("wind", "verb", "wind"),
    PronunciationRule("winds", "air", "wihnds"),
)

_WIND_RE = re.compile(r"\bwinds?\b", re.I)

_AIR_CUE = re.compile(
    r"\b("
    r"blow|blowing|blew|blown|breeze|air|airflow|storm|weather|strong|"
    r"gentle|gust|howl|howling|whistle|chill|cold|speed|direction|"
    r"turbine|turbines|energy|farm|farms|mill|mills|power|renewable|"
    r"pressure|current|currents|monsoon|cyclone|hurricane"
    r")\b",
    re.I,
)
_VERB_CUE = re.compile(
    r"\bwinds?\s+(?:the|it|this|that|a|an|your|my|his|her|their|up|down|around|"
    r"through|into|onto|back|tight|tighter)\b"
    r"|\b(?:to|please|now)\s+winds?\b"
    r"|\bwinds?\s+\w+\s+(?:around|up|onto|into)\b",
    re.I,
)


def _match_case(src: str, dest: str) -> str:
    if src.isupper():
        return dest.upper()
    if src[0].isupper():
        return dest[0].upper() + dest[1:]
    return dest


def _classify_wind(sentence: str) -> str:
    if _VERB_CUE.search(sentence or ""):
        return "verb"
    return "air"


def prepare_text_for_tts(text: str) -> str:
    """Internal TTS string only. Do not use for chat / speech_unit / RAG."""
    raw = text or ""
    if "wind" not in raw.lower():
        return raw

    meaning = _classify_wind(raw)
    hits = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal hits
        src = m.group(0)
        if meaning == "verb":
            return src
        dest = _AIR.get(src.lower())
        if not dest:
            return src
        hits += 1
        return _match_case(src, dest)

    out = _WIND_RE.sub(_sub, raw)
    if VOICE_DEBUG_LOGGING and (hits or meaning == "verb"):
        logger.info(
            "[TTS_PRONUNCIATION] word=wind meaning=%s pronunciation=%s sentence=%r",
            meaning,
            "wind-verb" if meaning == "verb" else "wind-air",
            raw[:120],
        )
    return out


if __name__ == "__main__":
    assert prepare_text_for_tts("The wind is blowing.") == "The wihnd is blowing."
    assert prepare_text_for_tts("Wind the clock.") == "Wind the clock."
    assert prepare_text_for_tts("Open the window.") == "Open the window."
    print("voice_pronunciation self-check ok")
