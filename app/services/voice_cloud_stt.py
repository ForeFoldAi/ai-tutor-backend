"""
Free cloud STT via Groq's Whisper (OpenAI-compatible /audio/transcriptions).

Much more accurate than browser SpeechRecognition or local `base.en`, and fast
(~0.3-0.6 s for a short utterance). Falls back to local faster-whisper, then to
the browser transcript, so the voice tutor never depends on it.
"""

from __future__ import annotations

import logging
import os

import httpx

from app.config import provider_keys
from app.services import key_pool

from app.services.voice_stt_postprocess import (
    is_meaningful_voice_transcript,
    postprocess_voice_transcript,
)

logger = logging.getLogger(__name__)

GROQ_STT_URL = os.environ.get("GROQ_STT_URL", "https://api.groq.com/openai/v1/audio/transcriptions")
GROQ_WHISPER_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
_PROVIDER = os.environ.get("VOICE_STT_PROVIDER", "").strip().lower()
_TIMEOUT = float(os.environ.get("VOICE_CLOUD_STT_TIMEOUT_SEC", "6"))


_SCOPE = "groq-stt"  # Whisper limits are separate from chat limits
_MAX_KEY_TRIES = 3


def _keys() -> list[str]:
    return provider_keys("groq")


def cloud_stt_enabled() -> bool:
    if _PROVIDER == "local":
        return False
    return bool(_keys())


def _suffix(audio: bytes) -> tuple[str, str]:
    if len(audio) >= 12 and audio[4:8] == b"ftyp":
        return "audio.mp4", "audio/mp4"
    if audio.startswith(b"OggS"):
        return "audio.ogg", "audio/ogg"
    if audio.startswith(b"RIFF"):
        return "audio.wav", "audio/wav"
    return "audio.webm", "audio/webm"


async def transcribe_cloud(
    audio: bytes, *, prompt: str = "", language: str = "en", subject_name: str = ""
) -> dict | None:
    """Return {"transcript", "confidence", "provider"} or None on any failure."""
    if not cloud_stt_enabled() or len(audio) < 512:
        return None
    name, mime = _suffix(audio)
    data = {
        "model": GROQ_WHISPER_MODEL,
        "language": language or "en",
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if prompt:
        data["prompt"] = prompt[-400:]
    payload = None
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            # Spread load across keys; a rate-limited key is parked and the next one tried.
            for key in key_pool.ordered(_SCOPE, _keys())[:_MAX_KEY_TRIES]:
                resp = await client.post(
                    GROQ_STT_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    data=data,
                    files={"file": (name, audio, mime)},
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    break
                logger.warning(
                    "Groq STT key=%s status=%s body=%s", key_pool.label(key), resp.status_code, resp.text[:160]
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    ra = (resp.headers.get("Retry-After") or "").strip()
                    key_pool.cool(_SCOPE, key, float(ra) if ra.replace(".", "", 1).isdigit() else 30.0)
                    continue
                return None  # 400/401 etc: another key won't help
    except Exception as exc:
        logger.warning("Groq STT failed: %s", exc)
        return None
    if payload is None:
        return None

    segs = payload.get("segments") or []
    kept, probs = [], []
    for seg in segs:
        if float(seg.get("no_speech_prob", 0.0)) > 0.6:
            continue
        t = (seg.get("text") or "").strip()
        if t:
            kept.append(t)
            probs.append(float(seg.get("avg_logprob", -0.5)))
    text = " ".join(kept) if segs else (payload.get("text") or "").strip()
    text = postprocess_voice_transcript(text.strip(), subject_name=subject_name)
    if not text or not is_meaningful_voice_transcript(text):
        return {"transcript": "", "confidence": -2.0, "provider": "groq"}
    conf = sum(probs) / len(probs) if probs else -0.4
    return {"transcript": text, "confidence": conf, "provider": "groq"}
