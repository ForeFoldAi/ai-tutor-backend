"""
Noise suppression for barge-in audio.

Preferred: pyrnnoise (RNNoise) when NOISE_SUPPRESSION_PROVIDER=rnnoise.
Fallback: spectral gate (always available).
"""

from __future__ import annotations

import io
import logging
import threading
import time
from typing import Any

import numpy as np

from app.config import NOISE_SUPPRESSION_ENABLED, NOISE_SUPPRESSION_PROVIDER
from app.services import voice_protection_metrics as metrics

logger = logging.getLogger(__name__)
_SAMPLE_RATE = 16000
_RN_RATE = 48000

_lock = threading.Lock()
_rnnoise: Any = None
_rnnoise_error: str | None = None
_active_backend = "spectral_gate"


def warm_noise_suppress() -> str:
    """Eager-init RNNoise if configured; otherwise spectral_gate."""
    global _active_backend
    if not NOISE_SUPPRESSION_ENABLED:
        _active_backend = "disabled"
        return _active_backend
    if NOISE_SUPPRESSION_PROVIDER == "rnnoise":
        if _get_rnnoise() is not None:
            _active_backend = "rnnoise"
            return _active_backend
        _active_backend = "spectral_gate"
        return _active_backend
    _active_backend = "spectral_gate"
    return _active_backend


def noise_status() -> dict[str, Any]:
    return {
        "enabled": NOISE_SUPPRESSION_ENABLED,
        "provider_config": NOISE_SUPPRESSION_PROVIDER,
        "backend": _active_backend,
        "rnnoise_loaded": _rnnoise is not None,
        "error": _rnnoise_error or "",
    }


def _get_rnnoise() -> Any | None:
    global _rnnoise, _rnnoise_error
    if _rnnoise is not None:
        return _rnnoise
    if _rnnoise_error:
        return None
    with _lock:
        if _rnnoise is not None:
            return _rnnoise
        if _rnnoise_error:
            return None
        try:
            from pyrnnoise import RNNoise

            # Native RNNoise rate; library resamples as needed.
            _rnnoise = RNNoise(sample_rate=_RN_RATE)
            logger.info("RNNoise (pyrnnoise) loaded")
            return _rnnoise
        except Exception as exc:
            _rnnoise_error = str(exc)
            logger.warning("RNNoise unavailable (%s) — spectral gate fallback", exc)
            return None


def _to_pcm(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        import torch
        import torchaudio

        wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if wav.ndim > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).numpy().astype(np.float32), int(sr)
    except Exception:
        pass
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception:
        return np.zeros(0, dtype=np.float32), _SAMPLE_RATE


def _pcm_to_wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _resample(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or pcm.size == 0:
        return pcm.astype(np.float32)
    duration = len(pcm) / float(src)
    n = max(1, int(duration * dst))
    x_old = np.linspace(0, 1, num=len(pcm), endpoint=False)
    x_new = np.linspace(0, 1, num=n, endpoint=False)
    return np.interp(x_new, x_old, pcm).astype(np.float32)


def _spectral_gate(pcm: np.ndarray) -> tuple[np.ndarray, float]:
    """Simple spectral subtraction using first 200ms as noise estimate."""
    if pcm.size < 800:
        return pcm, 0.0
    sr = _SAMPLE_RATE
    noise_n = min(len(pcm), int(0.2 * sr))
    noise = pcm[:noise_n]
    frame = 512
    hop = 256
    window = np.hanning(frame).astype(np.float32)
    noise_spec = np.abs(np.fft.rfft(noise[:frame] * window)) + 1e-6
    out = np.zeros_like(pcm)
    norm = np.zeros_like(pcm) + 1e-6
    before = float(np.sqrt(np.mean(pcm * pcm)) + 1e-9)
    for i in range(0, len(pcm) - frame, hop):
        chunk = pcm[i : i + frame] * window
        spec = np.fft.rfft(chunk)
        mag = np.abs(spec)
        phase = np.angle(spec)
        cleaned = np.maximum(mag - 1.0 * noise_spec[: len(mag)], 0.15 * mag)
        rec = np.fft.irfft(cleaned * np.exp(1j * phase), n=frame).astype(np.float32)
        out[i : i + frame] += rec * window
        norm[i : i + frame] += window * window
    out = out / norm
    after = float(np.sqrt(np.mean(out * out)) + 1e-9)
    reduction_db = float(max(0.0, 20.0 * np.log10(before / after))) if after > 0 else 0.0
    return out.astype(np.float32), min(reduction_db, 12.0)


def _rnnoise_denoise(pcm16: np.ndarray) -> tuple[np.ndarray, float]:
    """Process mono float32 @16kHz via RNNoise @48kHz chunks."""
    den = _get_rnnoise()
    if den is None:
        return _spectral_gate(pcm16)

    before = float(np.sqrt(np.mean(pcm16 * pcm16)) + 1e-9)
    pcm48 = _resample(pcm16, _SAMPLE_RATE, _RN_RATE)
    # pyrnnoise expects [channels, samples]
    chunk = np.asarray(pcm48, dtype=np.float32).reshape(1, -1)
    frames: list[np.ndarray] = []
    try:
        for _prob, frame in den.denoise_chunk(chunk):
            if frame is None:
                continue
            arr = np.asarray(frame, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            frames.append(arr)
    except Exception as exc:
        logger.debug("RNNoise denoise_chunk failed: %s", exc)
        return _spectral_gate(pcm16)

    if not frames:
        return _spectral_gate(pcm16)

    out48 = np.concatenate(frames).astype(np.float32)
    out16 = _resample(out48, _RN_RATE, _SAMPLE_RATE)
    # Match length for fair metrics / STT alignment
    if len(out16) > len(pcm16):
        out16 = out16[: len(pcm16)]
    elif len(out16) < len(pcm16):
        pad = np.zeros(len(pcm16) - len(out16), dtype=np.float32)
        out16 = np.concatenate([out16, pad])
    after = float(np.sqrt(np.mean(out16 * out16)) + 1e-9)
    reduction_db = float(max(0.0, 20.0 * np.log10(before / max(after, 1e-9))))
    return out16, min(reduction_db, 18.0)


def suppress_noise(audio_bytes: bytes) -> tuple[bytes, float]:
    """
    Returns (possibly-suppressed audio bytes, noise_reduction_db).
    Output is WAV if processed; otherwise original container.
    """
    global _active_backend
    if not NOISE_SUPPRESSION_ENABLED:
        return audio_bytes, 0.0

    t0 = time.perf_counter()
    pcm, sr = _to_pcm(audio_bytes)
    if pcm.size == 0:
        return audio_bytes, 0.0
    if sr != _SAMPLE_RATE:
        pcm = _resample(pcm, sr, _SAMPLE_RATE)

    used = "spectral_gate"
    if NOISE_SUPPRESSION_PROVIDER == "rnnoise" and _get_rnnoise() is not None:
        cleaned, db = _rnnoise_denoise(pcm)
        used = "rnnoise"
    else:
        cleaned, db = _spectral_gate(pcm)

    _active_backend = used
    latency_ms = (time.perf_counter() - t0) * 1000.0
    metrics.set_gauge("noise_reduction_db", db)
    metrics.set_gauge("noise_processing_latency_ms", latency_ms)
    try:
        return _pcm_to_wav_bytes(cleaned, _SAMPLE_RATE), db
    except Exception as exc:
        logger.debug("Noise suppress encode failed: %s", exc)
        return audio_bytes, 0.0
