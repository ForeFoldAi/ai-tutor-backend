"""
Silero VAD gate for mic audio (WebM/Opus or PCM).

Lazy-loads via torch.hub. If unavailable, falls back to energy+bandpass heuristic
so voice still works offline (ponytail: ceiling = heuristic; upgrade = baked ONNX).
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Any

import numpy as np

from app.config import MIN_SPEECH_MS, VAD_ENABLED, VAD_THRESHOLD
from app.services import voice_protection_metrics as metrics

logger = logging.getLogger(__name__)

_model: Any = None
_utils: Any = None
_model_error: str | None = None
_lock = threading.Lock()
_SAMPLE_RATE = 16000


def warm_vad() -> str:
    """Eager-load Silero (or confirm energy fallback). Returns backend name."""
    if not VAD_ENABLED:
        return "disabled"
    loaded = _get_silero()
    if loaded is not None:
        return "silero"
    return "energy"


def vad_status() -> dict[str, Any]:
    backend = (
        "silero"
        if _model is not None
        else ("error" if _model_error else ("disabled" if not VAD_ENABLED else "not_loaded"))
    )
    return {
        "enabled": VAD_ENABLED,
        "threshold": VAD_THRESHOLD,
        "min_speech_ms": MIN_SPEECH_MS,
        "backend": backend,
        "error": _model_error or "",
        "available": VAD_ENABLED and (_model is not None or _model_error is None),
        "loaded": _model is not None,
    }


def _get_silero() -> tuple[Any, Any] | None:
    global _model, _utils, _model_error
    if not VAD_ENABLED:
        return None
    if _model is not None:
        return _model, _utils
    if _model_error:
        return None
    with _lock:
        if _model is not None:
            return _model, _utils
        if _model_error:
            return None
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
                onnx=False,
            )
            model.eval()
            _model = model
            _utils = utils
            logger.info("Silero VAD loaded")
            return _model, _utils
        except Exception as exc:
            _model_error = str(exc)
            logger.warning("Silero VAD unavailable (%s) — using energy heuristic", exc)
            return None


def _decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Return mono float32 PCM at 16 kHz."""
    if len(audio_bytes) < 64:
        return np.zeros(0, dtype=np.float32)

    # Prefer soundfile / librosa via soundfile for webm is limited — try torchaudio first.
    try:
        import torch
        import torchaudio

        wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if wav.ndim > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != _SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, _SAMPLE_RATE)
        return wav.squeeze(0).numpy().astype(np.float32)
    except Exception:
        pass

    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        if sr != _SAMPLE_RATE and len(data):
            # Linear resample
            duration = len(data) / float(sr)
            n = int(duration * _SAMPLE_RATE)
            x_old = np.linspace(0, 1, num=len(data), endpoint=False)
            x_new = np.linspace(0, 1, num=n, endpoint=False)
            data = np.interp(x_new, x_old, data).astype(np.float32)
        return np.asarray(data, dtype=np.float32)
    except Exception as exc:
        logger.debug("Audio decode failed: %s", exc)
        return np.zeros(0, dtype=np.float32)


def _energy_vad(pcm: np.ndarray) -> dict[str, Any]:
    """Fallback when Silero missing — speech-band RMS gate."""
    if pcm.size < int(_SAMPLE_RATE * 0.05):
        return {
            "is_speech": False,
            "speech_probability": 0.0,
            "speech_duration_ms": 0.0,
            "backend": "energy",
        }
    # Rough band energy via absolute + high-pass (cheap)
    x = pcm - float(np.mean(pcm))
    rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
    # Map RMS to a pseudo-probability
    prob = float(min(1.0, max(0.0, (rms - 0.008) / 0.06)))
    # Consecutive frame estimate at 30ms
    frame = int(_SAMPLE_RATE * 0.03)
    spoken = 0
    for i in range(0, len(x) - frame, frame):
        chunk = x[i : i + frame]
        if float(np.sqrt(np.mean(chunk * chunk))) > 0.012:
            spoken += frame
    duration_ms = (spoken / _SAMPLE_RATE) * 1000.0
    is_speech = prob >= VAD_THRESHOLD and duration_ms >= MIN_SPEECH_MS
    return {
        "is_speech": is_speech,
        "speech_probability": prob,
        "speech_duration_ms": duration_ms,
        "backend": "energy",
    }


def _silero_vad(pcm: np.ndarray, model: Any, utils: Any) -> dict[str, Any]:
    import torch

    get_speech_timestamps = utils[0]
    audio = torch.from_numpy(pcm)
    # Windowed probabilities
    window = 512  # silero v4/v5 typical for 16k
    probs: list[float] = []
    with torch.no_grad():
        for i in range(0, max(0, len(audio) - window + 1), window):
            chunk = audio[i : i + window]
            if len(chunk) < window:
                break
            try:
                p = float(model(chunk, _SAMPLE_RATE).item())
            except Exception:
                p = float(model(chunk.unsqueeze(0), _SAMPLE_RATE).item())
            probs.append(p)

    if not probs:
        # Whole-clip timestamps
        stamps = get_speech_timestamps(
            audio,
            model,
            sampling_rate=_SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=MIN_SPEECH_MS,
        )
        duration_ms = sum(
            ((s["end"] - s["start"]) / _SAMPLE_RATE) * 1000.0 for s in stamps
        )
        prob = 1.0 if stamps else 0.0
        return {
            "is_speech": bool(stamps) and duration_ms >= MIN_SPEECH_MS,
            "speech_probability": prob,
            "speech_duration_ms": duration_ms,
            "backend": "silero",
        }

    avg_prob = float(sum(probs) / len(probs))
    speech_frames = sum(1 for p in probs if p >= VAD_THRESHOLD)
    duration_ms = speech_frames * (window / _SAMPLE_RATE) * 1000.0
    is_speech = avg_prob >= VAD_THRESHOLD and duration_ms >= MIN_SPEECH_MS
    return {
        "is_speech": is_speech,
        "speech_probability": avg_prob,
        "speech_duration_ms": duration_ms,
        "backend": "silero",
    }


def evaluate_vad(audio_bytes: bytes) -> dict[str, Any]:
    """
    Run VAD on an utterance blob.
    Returns is_speech, speech_probability, speech_duration_ms, backend.
    """
    if not VAD_ENABLED:
        return {
            "is_speech": True,
            "speech_probability": 1.0,
            "speech_duration_ms": float(MIN_SPEECH_MS),
            "backend": "disabled",
        }

    pcm = _decode_audio_bytes(audio_bytes)
    loaded = _get_silero()
    if loaded is None or pcm.size == 0:
        result = _energy_vad(pcm)
    else:
        try:
            result = _silero_vad(pcm, loaded[0], loaded[1])
        except Exception as exc:
            logger.warning("Silero eval failed: %s", exc)
            result = _energy_vad(pcm)

    metrics.set_gauge("speech_probability", float(result["speech_probability"]))
    metrics.set_gauge("speech_duration_ms", float(result["speech_duration_ms"]))
    if result["is_speech"]:
        metrics.incr("vad_accepted_count")
        metrics.log_event(
            "VAD_ACCEPTED",
            speech_probability=result["speech_probability"],
            speech_duration_ms=result["speech_duration_ms"],
            backend=result["backend"],
        )
    else:
        metrics.incr("vad_rejected_count")
        metrics.log_event(
            "VAD_REJECTED",
            speech_probability=result["speech_probability"],
            speech_duration_ms=result["speech_duration_ms"],
            backend=result["backend"],
        )
    return result
