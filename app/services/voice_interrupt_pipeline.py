"""
Interrupt decision pipeline with confidence scoring.

Volume spike (client) → denoise → VAD → echo → score(VAD, speaker, intent) → accept
"""

from __future__ import annotations

import time
from typing import Any

from app.config import (
    ECHO_SIMILARITY_THRESHOLD,
    INTERRUPT_SCORE_THRESHOLD,
    INTERRUPT_WEIGHT_INTENT,
    INTERRUPT_WEIGHT_SPEAKER,
    INTERRUPT_WEIGHT_VAD,
    MIN_SPEECH_MS,
    SPEAKER_SIMILARITY_THRESHOLD,
    VOICE_PROTECTION_ENABLED,
    WAKE_WORD_ENABLED,
    WAKE_WORDS,
)
from app.services import voice_protection_metrics as metrics
from app.services.voice_interrupt_intent import detect_interrupt_intent
from app.services.voice_noise_suppress import suppress_noise
from app.services.voice_speaker import verify_speaker
from app.services.voice_stt_postprocess import echo_similarity, transcript_likely_echo
from app.services.voice_vad import evaluate_vad


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_interrupt_score(
    *,
    vad_prob: float,
    speech_duration_ms: float,
    speaker_similarity: float,
    speaker_skipped: bool,
    intent_confidence: float,
    is_intent: bool,
) -> tuple[float, dict[str, float]]:
    """Weighted interrupt_score = w_v*vad + w_s*speaker + w_i*intent."""
    duration_factor = _clamp01(speech_duration_ms / max(float(MIN_SPEECH_MS), 1.0))
    vad_score = _clamp01(vad_prob) * (0.65 + 0.35 * duration_factor)

    if speaker_skipped:
        speaker_score = 0.85  # not enrolled / disabled — neutral-positive
    else:
        # Map cosine around SPEAKER_SIMILARITY_THRESHOLD into 0..1
        thr = SPEAKER_SIMILARITY_THRESHOLD
        speaker_score = _clamp01((speaker_similarity - (thr - 0.2)) / 0.4)

    intent_score = 1.0 if is_intent else _clamp01(intent_confidence)

    w_v = INTERRUPT_WEIGHT_VAD
    w_s = INTERRUPT_WEIGHT_SPEAKER
    w_i = INTERRUPT_WEIGHT_INTENT
    w_sum = w_v + w_s + w_i
    if w_sum <= 0:
        w_v, w_s, w_i, w_sum = 0.4, 0.4, 0.2, 1.0

    score = (w_v * vad_score + w_s * speaker_score + w_i * intent_score) / w_sum
    parts = {
        "vad_score": vad_score,
        "speaker_score": speaker_score,
        "intent_score": intent_score,
    }
    return _clamp01(score), parts


def evaluate_barge_in(
    audio_bytes: bytes,
    *,
    transcript: str = "",
    recent_ai_speech: str = "",
    student_key: str = "",
    voice_session_id: str = "",
    explicit_intent_only: bool = False,
) -> dict[str, Any]:
    """
    Run protection gates on a barge-in candidate.
    allow_interrupt when interrupt_score >= INTERRUPT_SCORE_THRESHOLD
    (intent phrases get a strong intent_score boost).
    """
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "allow_interrupt": False,
        "reason": "init",
        "speech_probability": 0.0,
        "speech_duration_ms": 0.0,
        "echo_similarity_score": 0.0,
        "speaker_similarity": 0.0,
        "noise_reduction_db": 0.0,
        "interrupt_score": 0.0,
        "interrupt_reason": "init",
        "score_parts": {},
        "intent": None,
        "processed_audio": audio_bytes,
    }

    if not VOICE_PROTECTION_ENABLED:
        out.update(
            allow_interrupt=True,
            reason="protection_disabled",
            interrupt_reason="protection_disabled",
            interrupt_score=1.0,
        )
        return out

    # ponytail: VAD on raw mic — spectral gate/RNNoise crush levels and false-reject barge-in
    vad = evaluate_vad(audio_bytes)
    out["speech_probability"] = vad["speech_probability"]
    out["speech_duration_ms"] = vad["speech_duration_ms"]

    processed, noise_db = suppress_noise(audio_bytes)
    out["processed_audio"] = processed
    out["noise_reduction_db"] = noise_db

    # Hard reject: clearly non-speech (horn/fan bursts fail duration/prob)
    if not vad["is_speech"]:
        metrics.incr("interrupt_rejected_count")
        metrics.incr("false_interrupt_count")
        metrics.log_event(
            "INTERRUPT_REJECTED",
            reason="vad",
            interrupt_score=0.0,
            speech_probability=out["speech_probability"],
            speech_duration_ms=out["speech_duration_ms"],
        )
        out.update(
            allow_interrupt=False,
            reason="vad_rejected",
            interrupt_reason="vad_rejected",
            interrupt_score=0.0,
        )
        return out

    intent = detect_interrupt_intent(
        transcript,
        wake_word_enabled=WAKE_WORD_ENABLED,
        wake_words=WAKE_WORDS,
    )
    out["intent"] = intent
    if intent["is_interrupt_intent"]:
        metrics.incr("intent_interrupt_count")

    if recent_ai_speech and transcript.strip():
        sim = echo_similarity(transcript, recent_ai_speech)
        out["echo_similarity_score"] = sim
        metrics.set_gauge("echo_similarity_score", sim)
        if not intent["is_interrupt_intent"] and transcript_likely_echo(
            transcript, recent_ai_speech, threshold=ECHO_SIMILARITY_THRESHOLD
        ):
            metrics.incr("echo_rejected_count")
            metrics.incr("interrupt_rejected_count")
            metrics.log_event("ECHO_REJECTED", similarity=sim, text=transcript[:80])
            metrics.log_event(
                "INTERRUPT_REJECTED",
                reason="echo",
                interrupt_score=0.0,
                similarity=sim,
            )
            out.update(
                allow_interrupt=False,
                reason="echo_rejected",
                interrupt_reason="echo_rejected",
                interrupt_score=0.0,
            )
            return out

    speaker = verify_speaker(student_key, processed)
    if voice_session_id:
        from app.services.voice_session_profile import verify_session_speaker

        session_sp = verify_session_speaker(voice_session_id, processed)
        if not session_sp.get("skipped"):
            speaker = session_sp
    out["speaker_similarity"] = float(speaker.get("similarity") or 0.0)
    speaker_skipped = bool(speaker.get("skipped"))

    # Hard reject only when enrolled and clearly a different speaker (not same student, noisy clip)
    if (
        not speaker.get("match", True)
        and not speaker_skipped
        and out["speaker_similarity"] < 0.45
    ):
        metrics.incr("interrupt_rejected_count")
        metrics.log_event(
            "INTERRUPT_REJECTED",
            reason="speaker",
            interrupt_score=0.0,
            similarity=out["speaker_similarity"],
        )
        out.update(
            allow_interrupt=False,
            reason="speaker_rejected",
            interrupt_reason="speaker_rejected",
            interrupt_score=0.0,
        )
        return out

    score, parts = compute_interrupt_score(
        vad_prob=float(vad["speech_probability"]),
        speech_duration_ms=float(vad["speech_duration_ms"]),
        speaker_similarity=out["speaker_similarity"],
        speaker_skipped=speaker_skipped,
        intent_confidence=float(intent.get("confidence") or 0.0),
        is_intent=bool(intent.get("is_interrupt_intent")),
    )
    out["interrupt_score"] = score
    out["score_parts"] = parts
    metrics.set_gauge("interrupt_score", score)

    if explicit_intent_only and not intent["is_interrupt_intent"]:
        metrics.incr("interrupt_rejected_count")
        metrics.log_event("INTERRUPT_REJECTED", reason="no_intent", interrupt_score=score)
        out.update(
            allow_interrupt=False,
            reason="no_intent",
            interrupt_reason="no_intent",
        )
        return out

    latency_ms = (time.perf_counter() - t0) * 1000.0
    metrics.set_gauge("interrupt_latency_ms", latency_ms)

    if score >= INTERRUPT_SCORE_THRESHOLD:
        metrics.incr("interrupt_accepted_count")
        reason = "intent" if intent["is_interrupt_intent"] else "score"
        metrics.log_event(
            "INTERRUPT_ACCEPTED",
            interrupt_score=score,
            interrupt_reason=reason,
            intent=intent.get("matched_phrase"),
            speech_probability=out["speech_probability"],
            speaker_similarity=out["speaker_similarity"],
            interrupt_latency_ms=latency_ms,
            **parts,
        )
        out.update(
            allow_interrupt=True,
            reason="accepted",
            interrupt_reason=reason,
            interrupt_latency_ms=latency_ms,
        )
        return out

    metrics.incr("interrupt_rejected_count")
    metrics.incr("false_interrupt_count")
    metrics.log_event(
        "INTERRUPT_REJECTED",
        interrupt_score=score,
        interrupt_reason="below_threshold",
        threshold=INTERRUPT_SCORE_THRESHOLD,
        **parts,
    )
    out.update(
        allow_interrupt=False,
        reason="below_threshold",
        interrupt_reason="below_threshold",
        interrupt_latency_ms=latency_ms,
    )
    return out
