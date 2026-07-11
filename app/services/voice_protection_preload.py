"""
Startup preload for voice-protection ML (VAD / speaker / noise).

Avoids first-barge-in latency and surfaces readiness via /health/voice.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_LOAD: dict[str, Any] = {
    "started_at": None,
    "finished_at": None,
    "vad_ms": None,
    "speaker_ms": None,
    "noise_ms": None,
    "vad_ok": False,
    "speaker_ok": False,
    "noise_ok": False,
    "errors": [],
}


def preload_status() -> dict[str, Any]:
    return dict(_LOAD)


def warm_voice_protection_models() -> dict[str, Any]:
    """Load Silero VAD, speaker backend, and noise suppress models eagerly."""
    from app.config import VOICE_PROTECTION_PRELOAD
    from app.services import voice_protection_metrics as metrics

    if not VOICE_PROTECTION_PRELOAD:
        _LOAD["errors"] = ["preload_disabled"]
        return preload_status()

    _LOAD["started_at"] = time.time()
    t_all = time.perf_counter()

    # VAD
    t0 = time.perf_counter()
    try:
        from app.services.voice_vad import warm_vad

        backend = warm_vad()
        _LOAD["vad_ok"] = True
        _LOAD["vad_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        metrics.set_gauge("model_load_vad_ms", float(_LOAD["vad_ms"]))
        logger.info("Voice protection: VAD ready (%s, %.0fms)", backend, _LOAD["vad_ms"])
    except Exception as exc:
        _LOAD["vad_ok"] = False
        _LOAD["vad_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        _LOAD["errors"].append(f"vad:{exc}")
        logger.warning("Voice protection: VAD preload failed (non-fatal): %s", exc)

    # Speaker
    t0 = time.perf_counter()
    try:
        from app.services.voice_speaker import warm_speaker

        backend = warm_speaker()
        _LOAD["speaker_ok"] = True
        _LOAD["speaker_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        metrics.set_gauge("model_load_speaker_ms", float(_LOAD["speaker_ms"]))
        logger.info("Voice protection: speaker ready (%s, %.0fms)", backend, _LOAD["speaker_ms"])
    except Exception as exc:
        _LOAD["speaker_ok"] = False
        _LOAD["speaker_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        _LOAD["errors"].append(f"speaker:{exc}")
        logger.warning("Voice protection: speaker preload failed (non-fatal): %s", exc)

    # Noise
    t0 = time.perf_counter()
    try:
        from app.services.voice_noise_suppress import warm_noise_suppress

        backend = warm_noise_suppress()
        _LOAD["noise_ok"] = True
        _LOAD["noise_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        metrics.set_gauge("model_load_noise_ms", float(_LOAD["noise_ms"]))
        logger.info("Voice protection: noise ready (%s, %.0fms)", backend, _LOAD["noise_ms"])
    except Exception as exc:
        _LOAD["noise_ok"] = False
        _LOAD["noise_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        _LOAD["errors"].append(f"noise:{exc}")
        logger.warning("Voice protection: noise preload failed (non-fatal): %s", exc)

    total = round((time.perf_counter() - t_all) * 1000, 1)
    metrics.set_gauge("model_load_total_ms", total)
    _LOAD["finished_at"] = time.time()
    _LOAD["total_ms"] = total
    logger.info("Voice protection preload finished in %.0fms", total)
    return preload_status()
