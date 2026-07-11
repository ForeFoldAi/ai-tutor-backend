"""
Speaker verification for barge-in (embedding cosine).

Enrollment utterance: "Hello, I am ready to learn."

Preferred backends (lazy):
  1. speechbrain ECAPA-TDNN (if installed)
  2. resemblyzer (if installed)
  3. MFCC+energy fingerprint — ponytail ceiling for CPU-only deploys;
     upgrade path: pip install speechbrain or resemblyzer.
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Any

import numpy as np

from app.config import SPEAKER_SIMILARITY_THRESHOLD, SPEAKER_VERIFICATION_ENABLED
from app.services import voice_protection_metrics as metrics

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_lock = threading.Lock()
_backend_name: str | None = None
_backend_error: str | None = None
_ecapa: Any = None
_resemblyzer: Any = None

# In-memory student embeddings: student_key -> float32 vector
_enrollments: dict[str, np.ndarray] = {}

ENROLLMENT_PROMPT = "Hello, I am ready to learn."


def warm_speaker() -> str:
    """Eager-load preferred speaker backend. Returns backend name."""
    return _load_backend()


def speaker_status() -> dict[str, Any]:
    return {
        "enabled": SPEAKER_VERIFICATION_ENABLED,
        "threshold": SPEAKER_SIMILARITY_THRESHOLD,
        "backend": _backend_name or ("error" if _backend_error else "not_loaded"),
        "error": _backend_error or "",
        "enrollment_prompt": ENROLLMENT_PROMPT,
        "enrolled_count": len(_enrollments),
        "loaded": _backend_name is not None,
    }


def _to_pcm16k(audio_bytes: bytes) -> np.ndarray:
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
            duration = len(data) / float(sr)
            n = int(duration * _SAMPLE_RATE)
            x_old = np.linspace(0, 1, num=len(data), endpoint=False)
            x_new = np.linspace(0, 1, num=n, endpoint=False)
            data = np.interp(x_new, x_old, data).astype(np.float32)
        return np.asarray(data, dtype=np.float32)
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _mfcc_embed(pcm: np.ndarray) -> np.ndarray:
    """Lightweight fingerprint — mean+std log-mel style features via FFT bands."""
    if pcm.size < 400:
        return np.zeros(32, dtype=np.float32)
    # Frame
    frame = 400
    hop = 160
    feats: list[np.ndarray] = []
    window = np.hanning(frame).astype(np.float32)
    for i in range(0, len(pcm) - frame, hop):
        chunk = pcm[i : i + frame] * window
        spec = np.abs(np.fft.rfft(chunk))
        # 32 mel-ish bands by log-spaced pooling
        bands = np.array_split(spec, 32)
        feats.append(np.array([float(np.log1p(np.mean(b))) for b in bands], dtype=np.float32))
    mat = np.stack(feats, axis=0)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    emb = np.concatenate([mean, std]).astype(np.float32)
    n = float(np.linalg.norm(emb) + 1e-9)
    return emb / n


def _load_backend() -> str:
    global _backend_name, _backend_error, _ecapa, _resemblyzer
    if _backend_name:
        return _backend_name
    with _lock:
        if _backend_name:
            return _backend_name
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

            _ecapa = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
            )
            _backend_name = "ecapa"
            logger.info("Speaker verification backend: ECAPA-TDNN")
            return _backend_name
        except Exception as exc:
            logger.debug("ECAPA unavailable: %s", exc)

        try:
            from resemblyzer import VoiceEncoder  # type: ignore

            _resemblyzer = VoiceEncoder()
            _backend_name = "resemblyzer"
            logger.info("Speaker verification backend: resemblyzer")
            return _backend_name
        except Exception as exc:
            logger.debug("Resemblyzer unavailable: %s", exc)

        _backend_name = "mfcc"
        _backend_error = "Using MFCC fingerprint (install speechbrain or resemblyzer for production)"
        logger.info("Speaker verification backend: mfcc fingerprint")
        return _backend_name


def embed_audio(audio_bytes: bytes) -> np.ndarray | None:
    pcm = _to_pcm16k(audio_bytes)
    if pcm.size < 1600:  # <100ms
        return None
    backend = _load_backend()
    if backend == "ecapa" and _ecapa is not None:
        import torch

        wav = torch.from_numpy(pcm).unsqueeze(0)
        with torch.no_grad():
            emb = _ecapa.encode_batch(wav).squeeze().cpu().numpy().astype(np.float32)
        n = float(np.linalg.norm(emb) + 1e-9)
        return emb / n
    if backend == "resemblyzer" and _resemblyzer is not None:
        emb = np.asarray(_resemblyzer.embed_utterance(pcm), dtype=np.float32)
        n = float(np.linalg.norm(emb) + 1e-9)
        return emb / n
    return _mfcc_embed(pcm)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))


def enroll_speaker(student_key: str, audio_bytes: bytes) -> dict[str, Any]:
    if not student_key:
        return {"ok": False, "error": "student_key required"}
    emb = embed_audio(audio_bytes)
    if emb is None:
        return {"ok": False, "error": "Could not extract voice embedding — speak longer."}
    _enrollments[student_key] = emb
    return {
        "ok": True,
        "student_key": student_key,
        "dim": int(emb.shape[0]),
        "backend": _load_backend(),
        "prompt": ENROLLMENT_PROMPT,
    }


def verify_speaker(student_key: str, audio_bytes: bytes) -> dict[str, Any]:
    """
    Returns match bool + similarity. If verification disabled or no enrollment,
    match=True (fail-open so voice still works).
    """
    if not SPEAKER_VERIFICATION_ENABLED:
        return {
            "match": True,
            "similarity": 1.0,
            "skipped": True,
            "reason": "disabled",
        }
    enrolled = _enrollments.get(student_key or "")
    if enrolled is None:
        return {
            "match": True,
            "similarity": 1.0,
            "skipped": True,
            "reason": "not_enrolled",
        }
    emb = embed_audio(audio_bytes)
    if emb is None:
        metrics.incr("speaker_rejections")
        metrics.log_event("SPEAKER_REJECTED", reason="no_embedding", similarity=0.0)
        return {"match": False, "similarity": 0.0, "skipped": False, "reason": "no_embedding"}
    sim = cosine_similarity(enrolled, emb)
    metrics.set_gauge("speaker_similarity", sim)
    match = sim >= SPEAKER_SIMILARITY_THRESHOLD
    if not match:
        metrics.incr("speaker_rejections")
        metrics.log_event("SPEAKER_REJECTED", similarity=sim, threshold=SPEAKER_SIMILARITY_THRESHOLD)
    return {
        "match": match,
        "similarity": sim,
        "skipped": False,
        "reason": "ok" if match else "below_threshold",
        "threshold": SPEAKER_SIMILARITY_THRESHOLD,
    }


def clear_enrollment(student_key: str) -> None:
    _enrollments.pop(student_key, None)
