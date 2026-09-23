"""Tests for Nest voice server auth on /auth/voice-tts-pcm."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.modules.auth.dependencies import authorize_voice_tts


def test_voice_internal_token_allows_service_call(monkeypatch):
    monkeypatch.setenv("VOICE_INTERNAL_TOKEN", "voice-secret")
    authorize_voice_tts(None, MagicMock(), "voice-secret")


def test_voice_internal_token_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("VOICE_INTERNAL_TOKEN", "voice-secret")
    with pytest.raises(HTTPException):
        authorize_voice_tts(None, MagicMock(), "wrong")
