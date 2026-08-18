"""
Server-side speech-to-text using faster-whisper (optional dependency).

Lazy-loads the model on first request. Set VOICE_WHISPER_ENABLED=false to disable.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import threading
from typing import Any

from app.config import VOICE_WHISPER_ENABLED, WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL
from app.services.voice_stt_postprocess import postprocess_voice_transcript, transcript_likely_echo

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000

_model: Any = None
_model_error: str | None = None
_model_lock = threading.Lock()


def whisper_available() -> bool:
    """True when Whisper STT is enabled and the model loaded (or can load)."""
    if not VOICE_WHISPER_ENABLED:
        return False
    if _model is not None:
        return True
    if _model_error:
        return False
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def whisper_status() -> dict[str, str | bool]:
    return {
        "enabled": VOICE_WHISPER_ENABLED,
        "available": whisper_available(),
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "error": _model_error or "",
    }


def _get_model() -> Any:
    global _model, _model_error
    if _model is not None:
        return _model
    if not VOICE_WHISPER_ENABLED:
        raise RuntimeError("Whisper STT is disabled (VOICE_WHISPER_ENABLED=false)")
    with _model_lock:
        if _model is not None:
            return _model
        if _model_error:
            raise RuntimeError(_model_error)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            _model_error = "faster-whisper is not installed"
            raise RuntimeError(_model_error) from exc
        try:
            logger.info(
                "Loading Whisper model %s (device=%s, compute=%s)",
                WHISPER_MODEL,
                WHISPER_DEVICE,
                WHISPER_COMPUTE_TYPE,
            )
            _model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            logger.info("Whisper model ready: %s", WHISPER_MODEL)
            return _model
        except Exception as exc:
            _model_error = str(exc)
            logger.exception("Failed to load Whisper model")
            raise RuntimeError(_model_error) from exc


def _math_initial_prompt(subject_name: str) -> str:
    if "math" not in (subject_name or "").lower():
        return ""
    return (
        "School mathematics tutoring. Numbers, equations, fractions, algebra, geometry, "
        "x squared, divided by, equals, step one, step two."
    )


def _decode_webm_to_pcm(audio_bytes: bytes):
    """ffmpeg fallback when PyAV cannot parse concatenated browser WebM chunks."""
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                str(_SAMPLE_RATE),
                "pipe:1",
            ],
            input=audio_bytes,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("ffmpeg decode unavailable: %s", exc)
        return None
    if proc.returncode != 0 or len(proc.stdout) < 320:
        return None
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _audio_suffix(audio_bytes: bytes) -> str:
    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        return ".mp4"
    if audio_bytes.startswith(b"OggS"):
        return ".ogg"
    return ".webm"


def _run_whisper(model: Any, audio: Any, *, language: str, subject_name: str):
    return model.transcribe(
        audio,
        language=language or "en",
        beam_size=1,
        best_of=1,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 450,
            "speech_pad_ms": 350,
            "threshold": 0.35,
        },
        compression_ratio_threshold=2.6,
        log_prob_threshold=-1.2,
        no_speech_threshold=0.5,
        condition_on_previous_text=False,
        initial_prompt=_math_initial_prompt(subject_name),
    )


def _transcribe_sync(
    audio_bytes: bytes,
    *,
    language: str,
    subject_name: str,
    reject_if_similar_to: str = "",
) -> dict[str, str | float | bool]:
    if len(audio_bytes) < 256:
        return {"transcript": "", "confidence": 0.0, "rejected": False}
    model = _get_model()
    transcribe_kwargs = {"language": language or "en", "subject_name": subject_name}
    segments = None
    with tempfile.NamedTemporaryFile(suffix=_audio_suffix(audio_bytes), delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        try:
            segments, _info = _run_whisper(model, tmp.name, **transcribe_kwargs)
        except Exception as exc:
            # ponytail: browser MediaRecorder chunk concat often lacks a valid WebM header for PyAV
            if "Invalid data" not in str(exc) and exc.__class__.__name__ != "InvalidDataError":
                raise
            pcm = _decode_webm_to_pcm(audio_bytes)
            if pcm is None or pcm.size < int(_SAMPLE_RATE * 0.12):
                logger.warning("Whisper STT: invalid WebM (%s bytes), decode failed", len(audio_bytes))
                return {"transcript": "", "confidence": 0.0, "rejected": False}
            segments, _info = _run_whisper(model, pcm, **transcribe_kwargs)
        kept: list[str] = []
        logprobs: list[float] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            prob = getattr(seg, "avg_logprob", None)
            if prob is not None and prob < -1.35:
                continue
            kept.append(text)
            if prob is not None:
                logprobs.append(float(prob))
        text = postprocess_voice_transcript(" ".join(kept).strip(), subject_name=subject_name)
        confidence = sum(logprobs) / len(logprobs) if logprobs else -0.5
        rejected = False
        if text and reject_if_similar_to and transcript_likely_echo(text, reject_if_similar_to):
            rejected = True
            text = ""
    return {"transcript": text, "confidence": confidence, "rejected": rejected}


async def transcribe_audio_bytes(
    audio_bytes: bytes,
    *,
    language: str = "en",
    subject_name: str = "",
    reject_if_similar_to: str = "",
) -> dict[str, str | float | bool]:
    """Transcribe uploaded audio (WebM/Opus from browser MediaRecorder)."""
    return await asyncio.to_thread(
        _transcribe_sync,
        audio_bytes,
        language=language,
        subject_name=subject_name,
        reject_if_similar_to=reject_if_similar_to,
    )
