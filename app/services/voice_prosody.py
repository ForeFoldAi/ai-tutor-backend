"""
Dynamic TTS prosody for tutor speech — rate/pitch via edge_tts params (not inline SSML).

ponytail: edge_tts.Communicate already wraps plain text in SSML; passing our own
<speak> tags gets HTML-escaped and read aloud as literal markup.
"""

from __future__ import annotations

import re

from app.config import VOICE_SSML_PROSODY, VOICE_TTS_PITCH, VOICE_TTS_RATE
from app.services.tts_sanitize import sanitize_chunk_for_tts

_MATH_HINT = re.compile(
    r"\b(\d+|step\s+\d+|equation|formula|fraction|multiply|divide|equals?|percent)\b",
    re.I,
)
_QUESTION = re.compile(r"\?\s*$")
_SHORT_ACK = re.compile(r"^(good|great|nice|exactly|right|yes|okay|ok|correct)\b", re.I)


def prepare_voice_tts(text: str, *, chunk_index: int = 0) -> tuple[str, str, str]:
    """Return (sanitized_text, rate, pitch) for edge_tts.Communicate."""
    stripped = sanitize_chunk_for_tts((text or "").strip())
    if not stripped:
        return "", VOICE_TTS_RATE, VOICE_TTS_PITCH

    if not VOICE_SSML_PROSODY:
        return stripped, VOICE_TTS_RATE, VOICE_TTS_PITCH

    rate = VOICE_TTS_RATE
    pitch = VOICE_TTS_PITCH

    if chunk_index == 0:
        rate = "+0%"
    elif _SHORT_ACK.match(stripped) and len(stripped.split()) <= 6:
        rate = "+4%"
    elif _MATH_HINT.search(stripped):
        rate = "-6%"
        pitch = "+1Hz"
    elif _QUESTION.search(stripped):
        rate = "-2%"
        pitch = "+2Hz"

    return stripped, rate, pitch


def apply_voice_prosody(text: str, *, chunk_index: int = 0) -> str:
    """Sanitize text for TTS; prosody is applied via Communicate rate/pitch params."""
    spoken, _, _ = prepare_voice_tts(text, chunk_index=chunk_index)
    return spoken
