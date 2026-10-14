"""
Speech prosody layer — intent classification + TTS delivery params.

Sits between LLM text and edge_tts.Communicate (rate/pitch/volume, not inline SSML).
ponytail: edge_tts XML-escapes the utterance then wraps ONE <prosody> tag, so
inline SSML cannot vary pitch mid-sentence. Meaning → one subtle Communicate
triple per chunk; the neural voice does the rest (especially "?" / "." / ",").

Do NOT re-add period/comma/vocab catch-alls — they made every explanation
chunk the same slightly-slow, slightly-flat contour.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from app.config import VOICE_DEBUG_LOGGING, VOICE_SSML_PROSODY, VOICE_TTS_PITCH, VOICE_TTS_RATE, VOICE_TTS_VOLUME
from app.services.tts_sanitize import sanitize_chunk_for_tts
from app.services.voice_pronunciation import prepare_text_for_tts

logger = logging.getLogger(__name__)

# Narrow math-step cue only — not general textbook nouns (resource/weather/energy).
_MATH_STEP = re.compile(
    r"\b(step\s+\d+|equation|multiply both|divided by|both sides)\b",
    re.I,
)
_GREETING = re.compile(
    r"^(hi|hello|hey|hii|namaste|good\s+(morning|afternoon|evening))\b"
    r"|^how are you(?:\s+today)?\s*[?!.]*$",
    re.I,
)
_SHORT_ACK = re.compile(
    r"^(good|great|nice|exactly|right|yes|okay|ok|correct|perfect|awesome)\b",
    re.I,
)
_ENCOURAGE = re.compile(
    r"\b(well done|nice work|you'?re doing|you'?ve got|proud of you|"
    r"keep going|almost there|that'?s it|good question|that'?s a good)\b",
    re.I,
)
_CHECK_IN = re.compile(
    r"\b(does that make sense|got it|shall we|ready to|want to try|"
    r"what do you think|can you tell|try this|why do you think)\b",
    re.I,
)
_IMPORTANT = re.compile(
    r"\b(most important|main (idea|reason|point|thing)|key point|remember (this|that)|"
    r"the important|critical|essential|what matters)\b",
    re.I,
)
_INTRO = re.compile(
    r"^(let'?s|okay|so,?\s+(?:let'?s|we)|first|today|now,?\s+let|"
    r"good question|sure|of course)\b",
    re.I,
)
_TRANSITION = re.compile(
    r"^(now|next|okay|so,|and then|moving on|let'?s look|here'?s)\b",
    re.I,
)
_EXAMPLE = re.compile(
    r"\b(for example|imagine|picture this|here'?s an example|here'?s a simple|"
    r"like when|think about this|simple example)\b",
    re.I,
)
_SUMMARY = re.compile(
    r"\b(so remember|in short|to sum up|in simple words|these (three|main)|"
    r"key takeaways|the main ideas|what we covered|so,\s+in simple)\b",
    re.I,
)
_CLARIFICATION = re.compile(
    r"\b(in simpler|another way|let me explain(?:\s+that)?(?:\s+again)?|"
    r"simpler words|break that down|step by step)\b",
    re.I,
)
_COMPARISON = re.compile(r"\b(compared to|unlike|similar to|on the other hand)\b", re.I)
_CORRECTION = re.compile(
    r"\b(not quite|not exactly|actually,|let me correct|that'?s not|"
    r"the correct (?:answer|idea)|almost[,.]? but)\b",
    re.I,
)
_WARNING = re.compile(
    r"\b(be careful|watch out|don'?t\s+(?:forget|confuse)|do not |"
    r"never\s+(?:mix|confuse)|warning)\b",
    re.I,
)
_DEFINITION = re.compile(
    r"(?:"
    r"\b(?:is|are)\s+(?:a|an)\b|"
    r"\b(?:means|refers to|is called|are called|is known as|are known as)\b|"
    r"\bis defined as\b"
    r")",
    re.I,
)
_CLOSING = re.compile(
    r"\b(that'?s the (?:idea|main idea)|we'?ll stop there|more next time|"
    r"see you|goodbye)\b",
    re.I,
)
_ENDS_QUESTION = re.compile(r"\?\s*$")
_ENDS_EXCLAIM = re.compile(r"!\s*$")

_RATE_RE = re.compile(r"^([+-]?\d+)\s*%$")
_PITCH_RE = re.compile(r"^([+-]?\d+)\s*Hz$", re.I)
_VOLUME_RE = re.compile(r"^([+-]?\d+)\s*%$")

# Clamp env overrides; intent deltas stay tiny (±3 rate, ±1 pitch).
_RATE_MIN, _RATE_MAX = -18, 12
_PITCH_MIN, _PITCH_MAX = -4, 8
_VOLUME_MIN, _VOLUME_MAX = -6, 12

# Intent → (rate_delta_pct, pitch_delta_hz, volume_delta_pct)
# Applied on top of VOICE_TTS_RATE/PITCH/VOLUME. Keep tiny.
_INTENT_DELTA: dict[str, tuple[int, int, int]] = {
    "GREETING": (0, 0, 0),
    "INTRODUCTION": (-1, 0, 0),
    "EXPLANATION": (0, 0, 0),
    "NORMAL_EXPLANATION": (0, 0, 0),
    "DEFINITION": (-2, 0, 0),
    "IMPORTANT_POINT": (-3, 0, 1),
    "EXAMPLE": (0, 0, 0),
    "TRANSITION": (0, 0, 0),
    "QUESTION": (0, 1, 0),  # "?" already rises; +1Hz only
    "ANSWER": (0, 0, 0),
    "CORRECTION": (-1, 0, 0),
    "WARNING": (-2, 0, 1),
    "SUMMARY": (-2, 0, 0),
    "ENCOURAGEMENT": (1, 1, 1),
    "CLARIFICATION": (-2, 0, 0),
    "COMPARISON": (0, 0, 0),
    "CONCLUSION": (-2, 0, 0),
    "CLOSING": (-2, 0, 0),
}


class SpeechIntent(str, Enum):
    GREETING = "GREETING"
    INTRODUCTION = "INTRODUCTION"
    EXPLANATION = "EXPLANATION"
    IMPORTANT_POINT = "IMPORTANT_POINT"
    DEFINITION = "DEFINITION"
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"
    EXAMPLE = "EXAMPLE"
    COMPARISON = "COMPARISON"
    TRANSITION = "TRANSITION"
    SUMMARY = "SUMMARY"
    ENCOURAGEMENT = "ENCOURAGEMENT"
    CORRECTION = "CORRECTION"
    WARNING = "WARNING"
    CLARIFICATION = "CLARIFICATION"
    CONCLUSION = "CONCLUSION"
    CLOSING = "CLOSING"


@dataclass(frozen=True)
class SpeechDelivery:
    text: str
    speech_intent: SpeechIntent
    rate: str
    pitch: str
    volume: str
    pause_before_ms: int = 0
    pause_after_ms: int = 0
    boundary: str = "auto"


def _parse_signed(value: str, pattern: re.Pattern[str]) -> int:
    m = pattern.match((value or "").strip())
    return int(m.group(1)) if m else 0


def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _adjust_rate(base: str, delta_pct: int) -> str:
    return f"{_clamp(_parse_signed(base, _RATE_RE) + delta_pct, _RATE_MIN, _RATE_MAX):+d}%"


def _adjust_pitch(base: str, delta_hz: int) -> str:
    return f"{_clamp(_parse_signed(base, _PITCH_RE) + delta_hz, _PITCH_MIN, _PITCH_MAX):+d}Hz"


def _adjust_volume(base: str, delta_pct: int) -> str:
    return f"{_clamp(_parse_signed(base, _VOLUME_RE) + delta_pct, _VOLUME_MIN, _VOLUME_MAX):+d}%"


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w])


def classify_speech_intent(text: str) -> SpeechIntent:
    """Heuristic speech-role tag for one TTS chunk (no LLM call)."""
    stripped = (text or "").strip()
    if not stripped:
        return SpeechIntent.EXPLANATION

    # Greeting before "?" — "Hi! How are you today?" is not a quiz.
    if _GREETING.search(stripped) and _word_count(stripped) <= 12:
        return SpeechIntent.GREETING
    if _ENDS_QUESTION.search(stripped) or _CHECK_IN.search(stripped):
        return SpeechIntent.QUESTION
    if _WARNING.search(stripped):
        return SpeechIntent.WARNING
    if _CORRECTION.search(stripped):
        return SpeechIntent.CORRECTION
    if _SUMMARY.search(stripped):
        return SpeechIntent.SUMMARY
    if _IMPORTANT.search(stripped):
        return SpeechIntent.IMPORTANT_POINT
    if _EXAMPLE.search(stripped):
        return SpeechIntent.EXAMPLE
    if _CLARIFICATION.search(stripped):
        return SpeechIntent.CLARIFICATION
    if _COMPARISON.search(stripped):
        return SpeechIntent.COMPARISON
    if _ENCOURAGE.search(stripped) or (_SHORT_ACK.match(stripped) and _word_count(stripped) <= 6):
        return SpeechIntent.ENCOURAGEMENT
    if _DEFINITION.search(stripped) and not _ENDS_QUESTION.search(stripped):
        return SpeechIntent.DEFINITION
    if _CLOSING.search(stripped):
        return SpeechIntent.CLOSING
    if _TRANSITION.match(stripped):
        return SpeechIntent.TRANSITION
    if _INTRO.match(stripped):
        return SpeechIntent.INTRODUCTION
    if stripped.lower().startswith(("so ", "that's why", "the reason")):
        return SpeechIntent.ANSWER
    if _ENDS_EXCLAIM.search(stripped) and _word_count(stripped) <= 8:
        return SpeechIntent.ENCOURAGEMENT
    if stripped.lower().startswith(("so,", "that's all", "all right")) and _word_count(stripped) <= 10:
        return SpeechIntent.CLOSING
    return SpeechIntent.EXPLANATION


def prepare_speech_delivery(
    text: str,
    *,
    chunk_index: int = 0,
    boundary: str = "auto",
) -> SpeechDelivery:
    """Build TTS delivery plan for one semantic speech chunk."""
    stripped = sanitize_chunk_for_tts((text or "").strip())
    intent = classify_speech_intent(stripped)
    if not stripped:
        return SpeechDelivery(
            text="",
            speech_intent=intent,
            rate=VOICE_TTS_RATE,
            pitch=VOICE_TTS_PITCH,
            volume=VOICE_TTS_VOLUME,
            boundary=boundary,
        )

    if not VOICE_SSML_PROSODY:
        return SpeechDelivery(
            text=stripped,
            speech_intent=intent,
            rate=VOICE_TTS_RATE,
            pitch=VOICE_TTS_PITCH,
            volume=VOICE_TTS_VOLUME,
            boundary=boundary,
        )

    rate_d, pitch_d, vol_d = _INTENT_DELTA.get(intent.value, (0, 0, 0))
    # Opening warmth only for greet/intro — not a pitch reset on every first chunk.
    if chunk_index == 0 and intent in (
        SpeechIntent.GREETING,
        SpeechIntent.INTRODUCTION,
        SpeechIntent.ENCOURAGEMENT,
    ):
        pitch_d += 1
        vol_d += 1
    if intent == SpeechIntent.EXPLANATION and _MATH_STEP.search(stripped):
        rate_d -= 2

    rate = _adjust_rate(VOICE_TTS_RATE, rate_d)
    pitch = _adjust_pitch(VOICE_TTS_PITCH, pitch_d)
    volume = _adjust_volume(VOICE_TTS_VOLUME, vol_d)

    delivery = SpeechDelivery(
        text=stripped,
        speech_intent=intent,
        rate=rate,
        pitch=pitch,
        volume=volume,
        pause_before_ms=0,
        pause_after_ms=0,
        boundary=boundary,
    )
    if VOICE_DEBUG_LOGGING:
        logger.info(
            "[TTS] intent=%s rate=%s pitch=%s volume=%s text_length=%s chunk_id=%s",
            delivery.speech_intent.value,
            rate,
            pitch,
            volume,
            len(stripped),
            chunk_index,
        )
    return delivery


def prepare_voice_tts(
    text: str,
    *,
    chunk_index: int = 0,
    boundary: str = "auto",
) -> tuple[str, str, str, str]:
    """Return (sanitized_text, rate, pitch, volume) for edge_tts.Communicate."""
    d = prepare_speech_delivery(text, chunk_index=chunk_index, boundary=boundary)
    spoken = prepare_text_for_tts(d.text)
    return spoken, d.rate, d.pitch, d.volume


def apply_voice_prosody(text: str, *, chunk_index: int = 0) -> str:
    """Sanitize text for TTS; prosody is applied via Communicate rate/pitch/volume params."""
    return prepare_speech_delivery(text, chunk_index=chunk_index).text
