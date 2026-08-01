"""Production configuration guards."""

from __future__ import annotations

import os

import pytest


def test_production_checks_skip_in_development(monkeypatch):
    from app.core import production_checks as pc

    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    pc.validate_production_settings()


def test_production_checks_reject_weak_jwt(monkeypatch):
    from app.core import production_checks as pc
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "replace-with-strong-secret")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com")
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        pc.validate_production_settings()
    get_settings.cache_clear()
