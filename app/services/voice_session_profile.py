"""
Session-scoped voice profile — bootstrap from the student's first question audio,
verify subsequent turns, auto-delete when the voice chat session ends.

Storage: Redis (multi-replica) with in-memory fallback when Redis is unavailable.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import (
    SPEAKER_SIMILARITY_THRESHOLD,
    SPEAKER_VERIFICATION_ENABLED,
    VOICE_SESSION_PROFILE,
    VOICE_SESSION_PROFILE_MIN_MS,
    VOICE_SESSION_REDIS,
    VOICE_SESSION_TTL_SEC,
)
from app.services.voice_speaker import cosine_similarity, embed_audio

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_profiles: dict[str, "_SessionProfile"] = {}
_redis_client: object | None = None
_redis_unavailable = False


@dataclass
class _SessionProfile:
    embedding: np.ndarray
    speech_ms: float = 0.0
    sample_count: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def ready(self) -> bool:
        return self.speech_ms >= float(VOICE_SESSION_PROFILE_MIN_MS) and self.sample_count >= 1


def _redis_key(session_id: str) -> str:
    return f"voice:session:{session_id}"


def _get_redis():
    global _redis_client, _redis_unavailable
    if not VOICE_SESSION_REDIS or _redis_unavailable:
        return None
    if _redis_client is not None:
        return _redis_client
    with _lock:
        if _redis_client is not None:
            return _redis_client
        if _redis_unavailable:
            return None
        try:
            import redis

            from app.core.config import get_settings

            client = redis.from_url(get_settings().redis_url, decode_responses=True)
            client.ping()
            _redis_client = client
            logger.info("Session voice profiles using Redis backend")
            return _redis_client
        except Exception as exc:
            _redis_unavailable = True
            logger.warning("Session voice Redis unavailable (%s) — in-memory fallback", exc)
            return None


def _serialize_profile(prof: _SessionProfile) -> str:
    return json.dumps(
        {
            "embedding_b64": base64.b64encode(prof.embedding.astype(np.float32).tobytes()).decode("ascii"),
            "dim": int(prof.embedding.shape[0]),
            "speech_ms": prof.speech_ms,
            "sample_count": prof.sample_count,
            "created_at": prof.created_at,
        }
    )


def _deserialize_profile(raw: str) -> _SessionProfile | None:
    try:
        data = json.loads(raw)
        dim = int(data["dim"])
        emb = np.frombuffer(base64.b64decode(data["embedding_b64"]), dtype=np.float32)
        if emb.shape[0] != dim:
            return None
        return _SessionProfile(
            embedding=emb,
            speech_ms=float(data.get("speech_ms") or 0),
            sample_count=int(data.get("sample_count") or 0),
            created_at=float(data.get("created_at") or time.time()),
        )
    except Exception:
        return None


def _load_profile(session_id: str) -> _SessionProfile | None:
    r = _get_redis()
    if r is not None:
        try:
            raw = r.get(_redis_key(session_id))
            if raw:
                prof = _deserialize_profile(raw)
                if prof is not None:
                    with _lock:
                        _profiles[session_id] = prof
                    return prof
        except Exception as exc:
            logger.debug("Redis session voice load failed: %s", exc)
    with _lock:
        return _profiles.get(session_id)


def _save_profile(session_id: str, prof: _SessionProfile) -> None:
    r = _get_redis()
    if r is not None:
        try:
            r.setex(_redis_key(session_id), int(VOICE_SESSION_TTL_SEC), _serialize_profile(prof))
        except Exception as exc:
            logger.debug("Redis session voice save failed: %s", exc)
    with _lock:
        _profiles[session_id] = prof


def _estimate_speech_ms(audio_bytes: bytes) -> float:
    if len(audio_bytes) > 44 and audio_bytes[:4] == b"RIFF":
        return max(0.0, (len(audio_bytes) - 44) / 32000.0 * 1000.0)
    return max(0.0, len(audio_bytes) / 4000.0)


def _purge_expired_locked() -> int:
    cutoff = time.time() - float(VOICE_SESSION_TTL_SEC)
    dead = [k for k, v in _profiles.items() if v.created_at < cutoff]
    for k in dead:
        del _profiles[k]
    return len(dead)


def purge_expired_sessions() -> int:
    with _lock:
        return _purge_expired_locked()


def session_profile_status(session_id: str) -> dict:
    if not session_id:
        return {"exists": False, "ready": False, "backend": _backend_name()}
    prof = _load_profile(session_id)
    if prof is None:
        return {"exists": False, "ready": False, "backend": _backend_name()}
    return {
        "exists": True,
        "ready": prof.ready,
        "speech_ms": prof.speech_ms,
        "sample_count": prof.sample_count,
        "backend": _backend_name(),
    }


def _backend_name() -> str:
    if VOICE_SESSION_REDIS and _get_redis() is not None:
        return "redis"
    return "memory"


def bootstrap_session_voice(session_id: str, audio_bytes: bytes) -> dict:
    if not VOICE_SESSION_PROFILE or not session_id or len(audio_bytes) < 256:
        return {"ok": False, "skipped": True, "reason": "disabled_or_short"}
    emb = embed_audio(audio_bytes)
    if emb is None:
        return {"ok": False, "reason": "no_embedding"}
    ms = _estimate_speech_ms(audio_bytes)
    with _lock:
        _purge_expired_locked()
    prof = _load_profile(session_id)
    if prof is None:
        prof = _SessionProfile(embedding=emb.copy(), speech_ms=ms, sample_count=1)
    else:
        alpha = 0.35
        merged = (1.0 - alpha) * prof.embedding + alpha * emb
        n = float(np.linalg.norm(merged) + 1e-9)
        prof.embedding = (merged / n).astype(np.float32)
        prof.speech_ms += ms
        prof.sample_count += 1
    _save_profile(session_id, prof)
    logger.debug(
        "Session voice bootstrap session=%s ready=%s speech_ms=%.0f backend=%s",
        session_id[:8],
        prof.ready,
        prof.speech_ms,
        _backend_name(),
    )
    return {
        "ok": True,
        "ready": prof.ready,
        "speech_ms": prof.speech_ms,
        "sample_count": prof.sample_count,
        "backend": _backend_name(),
    }


def verify_session_speaker(session_id: str, audio_bytes: bytes) -> dict:
    if not SPEAKER_VERIFICATION_ENABLED or not VOICE_SESSION_PROFILE:
        return {"match": True, "similarity": 1.0, "skipped": True, "reason": "disabled"}
    prof = _load_profile(session_id or "")
    if prof is None or not prof.ready:
        return {"match": True, "similarity": 1.0, "skipped": True, "reason": "not_ready"}
    emb = embed_audio(audio_bytes)
    if emb is None:
        return {"match": False, "similarity": 0.0, "skipped": False, "reason": "no_embedding"}
    sim = cosine_similarity(prof.embedding, emb)
    match = sim >= SPEAKER_SIMILARITY_THRESHOLD
    return {
        "match": match,
        "similarity": sim,
        "skipped": False,
        "reason": "ok" if match else "below_threshold",
        "threshold": SPEAKER_SIMILARITY_THRESHOLD,
    }


def clear_session_voice(session_id: str) -> None:
    if not session_id:
        return
    r = _get_redis()
    if r is not None:
        try:
            r.delete(_redis_key(session_id))
        except Exception as exc:
            logger.debug("Redis session voice delete failed: %s", exc)
    with _lock:
        if _profiles.pop(session_id, None) is not None:
            logger.debug("Session voice cleared session=%s", session_id[:8])


def evaluate_session_user_audio(session_id: str, audio_bytes: bytes) -> dict:
    if not VOICE_SESSION_PROFILE or not session_id:
        return {"accept": True, "reason": "disabled"}
    status = session_profile_status(session_id)
    if not status.get("ready"):
        return {"accept": True, "reason": "profile_not_ready"}
    speaker = verify_session_speaker(session_id, audio_bytes)
    if speaker.get("skipped") or speaker.get("match", True):
        return {"accept": True, "reason": "ok", "speaker_similarity": speaker.get("similarity", 1.0)}
    return {
        "accept": False,
        "reason": "speaker_rejected",
        "speaker_similarity": speaker.get("similarity", 0.0),
    }
