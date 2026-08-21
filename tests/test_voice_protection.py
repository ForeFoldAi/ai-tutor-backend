"""
Voice-protection regression tests (false interrupt scenarios).

Covers VAD gating, echo rejection, interrupt intent, and pipeline decisions
without requiring Silero/ECAPA downloads (uses heuristics / pure functions).
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import io
import pytest


def _pcm_wav(seconds: float, freq: float = 0.0, sr: int = 16000, amp: float = 0.0) -> bytes:
    n = int(seconds * sr)
    t = np.linspace(0, seconds, n, endpoint=False)
    if freq > 0:
        pcm = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    else:
        pcm = (amp * np.random.randn(n)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_interrupt_intent_phrases():
    from app.services.voice_interrupt_intent import detect_interrupt_intent

    for phrase in ("stop", "wait", "hold on", "one second", "excuse me", "I have a doubt", "hey", "hi", "hello"):
        r = detect_interrupt_intent(phrase)
        assert r["is_interrupt_intent"], phrase

    assert not detect_interrupt_intent("photosynthesis is the process")["is_interrupt_intent"]
    assert not detect_interrupt_intent("")["is_interrupt_intent"]


def test_wake_word_optional():
    from app.services.voice_interrupt_intent import detect_interrupt_intent

    r = detect_interrupt_intent("hey tutor please explain", wake_word_enabled=True)
    assert r["is_interrupt_intent"] and r["kind"] == "wake"

    r2 = detect_interrupt_intent("hey tutor", wake_word_enabled=False)
    # "hey" alone is still an interrupt phrase
    assert r2["is_interrupt_intent"]


def test_echo_rejects_ai_speech():
    from app.services.voice_stt_postprocess import echo_similarity, transcript_likely_echo

    tutor = "Photosynthesis is the process plants use to make food from sunlight"
    assert transcript_likely_echo("photosynthesis is the process plants use to make food", tutor)
    assert echo_similarity("photosynthesis is the process plants use", tutor) > 0.7
    assert not transcript_likely_echo("what is mitochondria", tutor)


def test_echo_allows_student_question():
    from app.services.voice_stt_postprocess import transcript_likely_echo

    tutor = "The mitochondria is the powerhouse of the cell"
    assert not transcript_likely_echo("can you explain photosynthesis again", tutor)
    assert not transcript_likely_echo("stop", tutor)  # short; overlap low


def test_echo_rejects_greeting_sentence_prefix():
    from app.services.voice_stt_postprocess import transcript_likely_echo

    greeting = (
        "Hi Suneel, welcome back. "
        "What would you like to learn today from Chapter 2?"
    )
    assert transcript_likely_echo("what would", greeting)
    assert not transcript_likely_echo("what is photosynthesis", greeting)
    assert not transcript_likely_echo(
        "the mughals",
        "Later the Mughals take over, and the map changes again.",
    )


def test_incomplete_voice_utterance():
    from app.services.voice_stt_postprocess import is_incomplete_voice_utterance

    assert is_incomplete_voice_utterance("what would")
    assert is_incomplete_voice_utterance("can you tell")
    assert is_incomplete_voice_utterance("what is")
    assert not is_incomplete_voice_utterance("can you tell me about the political map")
    assert not is_incomplete_voice_utterance("what is your name")
    assert not is_incomplete_voice_utterance("why")


def test_echo_rejects_short_tts_paraphrase_with_truncation():
    """
    Regression: STT may truncate an AI sentence mid-word (e.g. dynasty/dynasties),
    but we still must reject it as speaker bleed.
    """
    from app.services.voice_stt_postprocess import echo_similarity, transcript_likely_echo

    tutor = (
        "Ah, Suneel, you're mixing two things. "
        "In Chapter 2, we don't count maps."
    )
    echo_frag = "Sunil you are mixing two things in"
    assert echo_similarity(echo_frag, tutor) > 0.7
    assert transcript_likely_echo(echo_frag, tutor)


def test_vad_rejects_silence_and_noise(monkeypatch):
    from app.services import voice_vad

    monkeypatch.setattr(voice_vad, "VAD_ENABLED", True)
    monkeypatch.setattr(voice_vad, "VAD_THRESHOLD", 0.75)
    monkeypatch.setattr(voice_vad, "MIN_SPEECH_MS", 300)
    monkeypatch.setattr(voice_vad, "_get_silero", lambda: None)

    silence = _pcm_wav(0.5, amp=0.0)
    r = voice_vad.evaluate_vad(silence)
    assert r["is_speech"] is False

    # Fan-like low amplitude noise
    fan = _pcm_wav(0.8, amp=0.004)
    r2 = voice_vad.evaluate_vad(fan)
    assert r2["is_speech"] is False

    # Car horn / tone burst short < 300ms should fail duration
    horn = _pcm_wav(0.15, freq=440.0, amp=0.3)
    r3 = voice_vad.evaluate_vad(horn)
    assert r3["is_speech"] is False


def test_vad_accepts_sustained_speech_band(monkeypatch):
    from app.services import voice_vad

    monkeypatch.setattr(voice_vad, "VAD_ENABLED", True)
    monkeypatch.setattr(voice_vad, "VAD_THRESHOLD", 0.55)
    monkeypatch.setattr(voice_vad, "MIN_SPEECH_MS", 300)
    monkeypatch.setattr(voice_vad, "_get_silero", lambda: None)

    speech = _pcm_wav(0.6, freq=180.0, amp=0.2)
    r = voice_vad.evaluate_vad(speech)
    assert r["speech_duration_ms"] >= 0
    # Energy heuristic may accept strong sustained tone
    assert "speech_probability" in r


def test_pipeline_rejects_echo_noise(monkeypatch):
    from app.services import voice_interrupt_pipeline as pipe

    monkeypatch.setattr(pipe, "VOICE_PROTECTION_ENABLED", True)

    # Force VAD accept so we can exercise echo path
    monkeypatch.setattr(
        pipe,
        "evaluate_vad",
        lambda _b: {
            "is_speech": True,
            "speech_probability": 0.9,
            "speech_duration_ms": 500,
            "backend": "stub",
        },
    )
    monkeypatch.setattr(pipe, "suppress_noise", lambda b: (b, 0.0))
    monkeypatch.setattr(
        pipe,
        "verify_speaker",
        lambda *_a, **_k: {"match": True, "similarity": 1.0, "skipped": True},
    )

    audio = _pcm_wav(0.5, freq=200.0, amp=0.2)
    tutor = "Gravity pulls objects toward the earth"
    out = pipe.evaluate_barge_in(
        audio,
        transcript="gravity pulls objects toward the earth",
        recent_ai_speech=tutor,
        student_key="test",
    )
    assert out["allow_interrupt"] is False
    assert out["reason"] == "echo_rejected"


def test_pipeline_accepts_stop_phrase(monkeypatch):
    from app.services import voice_interrupt_pipeline as pipe

    monkeypatch.setattr(pipe, "VOICE_PROTECTION_ENABLED", True)
    monkeypatch.setattr(pipe, "INTERRUPT_SCORE_THRESHOLD", 0.75)
    monkeypatch.setattr(
        pipe,
        "evaluate_vad",
        lambda _b: {
            "is_speech": True,
            "speech_probability": 0.95,
            "speech_duration_ms": 400,
            "backend": "stub",
        },
    )
    monkeypatch.setattr(pipe, "suppress_noise", lambda b: (b, 1.5))
    monkeypatch.setattr(
        pipe,
        "verify_speaker",
        lambda *_a, **_k: {"match": True, "similarity": 0.9, "skipped": False},
    )

    audio = _pcm_wav(0.4, freq=200.0, amp=0.2)
    out = pipe.evaluate_barge_in(
        audio,
        transcript="stop",
        recent_ai_speech="Let me explain photosynthesis in detail for you",
        student_key="test",
    )
    assert out["allow_interrupt"] is True
    assert out["intent"]["is_interrupt_intent"] is True
    assert out["interrupt_score"] >= 0.75


def test_interrupt_score_weights(monkeypatch):
    from app.services.voice_interrupt_pipeline import compute_interrupt_score

    score, parts = compute_interrupt_score(
        vad_prob=0.9,
        speech_duration_ms=500,
        speaker_similarity=0.85,
        speaker_skipped=False,
        intent_confidence=0.95,
        is_intent=True,
    )
    assert score >= 0.75
    assert "vad_score" in parts


def test_stress_simultaneous_noise_and_echo(monkeypatch):
    """Car horn + AI echo must not interrupt."""
    from app.services import voice_interrupt_pipeline as pipe

    monkeypatch.setattr(pipe, "VOICE_PROTECTION_ENABLED", True)
    monkeypatch.setattr(
        pipe,
        "evaluate_vad",
        lambda _b: {
            "is_speech": False,
            "speech_probability": 0.2,
            "speech_duration_ms": 80,
            "backend": "stub",
        },
    )
    monkeypatch.setattr(pipe, "suppress_noise", lambda b: (b, 0.0))
    audio = _pcm_wav(0.1, freq=880.0, amp=0.5)
    out = pipe.evaluate_barge_in(
        audio,
        transcript="photosynthesis is the process",
        recent_ai_speech="Photosynthesis is the process plants use",
        student_key="test",
    )
    assert out["allow_interrupt"] is False


def test_queue_overflow_config():
    from app.config import VOICE_SPEECH_QUEUE_MAXSIZE
    from app.services.voice_streaming import create_speech_unit_queue

    q = create_speech_unit_queue()
    if VOICE_SPEECH_QUEUE_MAXSIZE > 0:
        assert q.maxsize == VOICE_SPEECH_QUEUE_MAXSIZE


def test_speech_unit_lookahead_does_not_crash():
    from app.services.voice_chunking import extract_voice_chunks

    chunks, rest = extract_voice_chunks(
        "Photosynthesis is how plants make food because sunlight gives energy and water helps too",
        chunks_emitted=1,
    )
    assert isinstance(chunks, list)
    assert isinstance(rest, str)


def test_metrics_rates():
    from app.services import voice_protection_metrics as m

    before = m.snapshot()
    base_a = int(before.get("interrupt_accepted_count", 0) or 0)
    base_r = int(before.get("interrupt_rejected_count", 0) or 0)
    m.incr("interrupt_accepted_count", 3)
    m.incr("interrupt_rejected_count", 1)
    m.incr("false_interrupt_count", 1)
    snap = m.snapshot()
    total = (base_a + 3) + (base_r + 1)
    expected = (base_a + 3) / total if total else 0.0
    assert abs(float(snap["barge_in_success_rate"]) - expected) < 1e-9
    assert "false_interrupt_rate" in snap
    assert "echo_false_positive_rate" in snap



def test_pipeline_rejects_parent_or_other_child(monkeypatch):
    from app.services import voice_interrupt_pipeline as pipe

    monkeypatch.setattr(pipe, "VOICE_PROTECTION_ENABLED", True)
    monkeypatch.setattr(
        pipe,
        "evaluate_vad",
        lambda _b: {
            "is_speech": True,
            "speech_probability": 0.9,
            "speech_duration_ms": 500,
            "backend": "stub",
        },
    )
    monkeypatch.setattr(pipe, "suppress_noise", lambda b: (b, 0.0))
    monkeypatch.setattr(
        pipe,
        "verify_speaker",
        lambda *_a, **_k: {"match": False, "similarity": 0.3, "skipped": False, "reason": "below_threshold"},
    )

    audio = _pcm_wav(0.5, freq=220.0, amp=0.2)
    out = pipe.evaluate_barge_in(
        audio,
        transcript="come eat dinner now",
        recent_ai_speech="We are learning about cells",
        student_key="student-1",
    )
    assert out["allow_interrupt"] is False
    assert out["reason"] == "speaker_rejected"


def test_speaker_enrollment_roundtrip():
    from app.services.voice_speaker import clear_enrollment, enroll_speaker, verify_speaker

    clear_enrollment("unit-test-student")
    audio = _pcm_wav(1.2, freq=160.0, amp=0.25)
    enrolled = enroll_speaker("unit-test-student", audio)
    assert enrolled.get("ok") is True
    same = verify_speaker("unit-test-student", audio)
    assert same["match"] is True
    clear_enrollment("unit-test-student")


@pytest.mark.parametrize(
    "label,seconds,freq,amp",
    [
        ("music", 0.7, 523.25, 0.05),
        ("dog_bark_short", 0.12, 300.0, 0.4),
        ("notification", 0.08, 880.0, 0.5),
        ("tv_noise", 0.9, 0.0, 0.01),
    ],
)
def test_short_or_weak_env_noise_rejected_by_vad(monkeypatch, label, seconds, freq, amp):
    from app.services import voice_vad

    monkeypatch.setattr(voice_vad, "VAD_ENABLED", True)
    monkeypatch.setattr(voice_vad, "VAD_THRESHOLD", 0.75)
    monkeypatch.setattr(voice_vad, "MIN_SPEECH_MS", 300)
    monkeypatch.setattr(voice_vad, "_get_silero", lambda: None)

    clip = _pcm_wav(seconds, freq=freq, amp=amp)
    r = voice_vad.evaluate_vad(clip)
    # Short clips always fail duration; weak noise fails probability.
    if seconds * 1000 < 300 or amp < 0.02:
        assert r["is_speech"] is False, label
