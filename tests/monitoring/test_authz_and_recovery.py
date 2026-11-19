"""Authorization + alert recovery logic smoke tests (no live DB required for helpers)."""

from datetime import UTC, datetime

from app.modules.monitoring.detection.engine import should_fire_with_cooldown
from app.modules.monitoring.router import _internal_token_ok


def test_internal_token_rejects_empty(monkeypatch):
    monkeypatch.delenv("MONITORING_INGEST_TOKEN", raising=False)
    monkeypatch.delenv("VOICE_INTERNAL_TOKEN", raising=False)
    assert _internal_token_ok(None) is False
    assert _internal_token_ok("") is False
    assert _internal_token_ok("anything") is False


def test_internal_token_accepts_configured(monkeypatch):
    monkeypatch.setenv("MONITORING_INGEST_TOKEN", "secret-token")
    assert _internal_token_ok("secret-token") is True
    assert _internal_token_ok("wrong") is False


def test_recovery_notification_cooldown_gate():
    now = datetime.now(UTC)
    # After recovery we still respect cooldown for re-fire
    assert should_fire_with_cooldown(last_notified_at=now, now=now, cooldown_seconds=300) is False
