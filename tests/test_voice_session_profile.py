"""Session-scoped voice profile lifecycle."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf


def _wav(seconds: float, freq: float = 180.0, amp: float = 0.25, sr: int = 16000) -> bytes:
    n = int(seconds * sr)
    t = np.linspace(0, seconds, n, endpoint=False)
    pcm = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_session_profile_bootstrap_and_clear(monkeypatch):
    from app.services import voice_session_profile as vsp

    monkeypatch.setattr(vsp, "VOICE_SESSION_PROFILE", True)
    monkeypatch.setattr(vsp, "VOICE_SESSION_REDIS", False)
    monkeypatch.setattr(vsp, "VOICE_SESSION_PROFILE_MIN_MS", 800)

    sid = "test-session-abc"
    vsp.clear_session_voice(sid)
    audio = _wav(1.5)
    out = vsp.bootstrap_session_voice(sid, audio)
    assert out.get("ok") is True
    status = vsp.session_profile_status(sid)
    assert status.get("exists") is True
    same = vsp.verify_session_speaker(sid, audio)
    assert same.get("match") is True
    vsp.clear_session_voice(sid)
    assert vsp.session_profile_status(sid).get("exists") is False


def test_session_profile_fail_open_until_ready(monkeypatch):
    from app.services import voice_session_profile as vsp

    monkeypatch.setattr(vsp, "VOICE_SESSION_PROFILE", True)
    monkeypatch.setattr(vsp, "VOICE_SESSION_REDIS", False)
    monkeypatch.setattr(vsp, "VOICE_SESSION_PROFILE_MIN_MS", 5000)

    sid = "test-session-short"
    vsp.clear_session_voice(sid)
    gate = vsp.evaluate_session_user_audio(sid, _wav(0.4))
    assert gate.get("accept") is True
    assert gate.get("reason") == "profile_not_ready"
    vsp.clear_session_voice(sid)
