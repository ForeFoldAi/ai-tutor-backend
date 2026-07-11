"""
Production voice-protection + streaming metrics.

Logs: [VAD_*] [ECHO_*] [SPEAKER_*] [INTERRUPT_*]
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_counters: dict[str, int | float] = {
    "speech_probability": 0.0,
    "speech_duration_ms": 0.0,
    "speaker_similarity": 0.0,
    "echo_similarity_score": 0.0,
    "noise_reduction_db": 0.0,
    "noise_processing_latency_ms": 0.0,
    "interrupt_latency_ms": 0.0,
    "interrupt_score": 0.0,
    "echo_rejected_count": 0,
    "speaker_rejections": 0,
    "vad_rejected_count": 0,
    "vad_accepted_count": 0,
    "false_interrupt_count": 0,
    "interrupt_accepted_count": 0,
    "interrupt_rejected_count": 0,
    "intent_interrupt_count": 0,
    "model_load_vad_ms": 0.0,
    "model_load_speaker_ms": 0.0,
    "model_load_noise_ms": 0.0,
    "model_load_total_ms": 0.0,
    "tts_queue_depth": 0.0,
    "tts_wait_time_ms": 0.0,
    "playback_gap_ms": 0.0,
    "first_tts_latency_ms": 0.0,
    "first_token_latency_ms": 0.0,
    "first_audio_latency_ms": 0.0,
    "chunk_size": 0.0,
    "chunk_generation_latency_ms": 0.0,
    "avg_playback_gap_ms": 0.0,
    "playback_gap_samples": 0,
}


def snapshot() -> dict[str, int | float]:
    with _lock:
        data = dict(_counters)
    accepted = float(data.get("interrupt_accepted_count", 0) or 0)
    rejected = float(data.get("interrupt_rejected_count", 0) or 0)
    total = accepted + rejected
    echo_rej = float(data.get("echo_rejected_count", 0) or 0)
    speaker_rej = float(data.get("speaker_rejections", 0) or 0)
    false_n = float(data.get("false_interrupt_count", 0) or 0)
    data["false_interrupt_rate"] = (false_n / total) if total else 0.0
    data["barge_in_success_rate"] = (accepted / total) if total else 0.0
    data["echo_false_positive_rate"] = (echo_rej / total) if total else 0.0
    data["speaker_false_positive_rate"] = (speaker_rej / total) if total else 0.0
    return data


def set_gauge(name: str, value: float) -> None:
    with _lock:
        _counters[name] = float(value)


def incr(name: str, amount: int = 1) -> None:
    with _lock:
        cur = _counters.get(name, 0)
        _counters[name] = int(cur) + amount


def record_playback_gap(ms: float) -> None:
    with _lock:
        n = int(_counters.get("playback_gap_samples", 0) or 0) + 1
        prev = float(_counters.get("avg_playback_gap_ms", 0) or 0)
        avg = prev + (ms - prev) / n
        _counters["playback_gap_samples"] = n
        _counters["avg_playback_gap_ms"] = avg
        _counters["playback_gap_ms"] = float(ms)


def log_event(tag: str, **fields: Any) -> None:
    logger.info("[%s] %s", tag, fields)
