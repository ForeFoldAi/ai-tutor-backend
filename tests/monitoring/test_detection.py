"""Unit tests for monitoring detection helpers."""

from datetime import UTC, datetime, timedelta

from app.modules.monitoring.detection.engine import (
    compare_threshold,
    consecutive_breaches,
    ewma_anomaly,
    fingerprint_error,
    mad_zscore,
    normalize_error_message,
    should_fire_with_cooldown,
)
from app.modules.monitoring.redaction import preview_message, redact_obj


def test_fingerprint_groups_volatile_ids():
    a = fingerprint_error(title="timeout user 12345 on req abcdef0123456789", service="api")
    b = fingerprint_error(title="timeout user 99999 on req deadbeefdeadbeef", service="api")
    assert a == b


def test_normalize_strips_uuids():
    text = normalize_error_message("failed 550e8400-e29b-41d4-a716-446655440000 once")
    assert "<uuid>" in text
    assert "550e8400" not in text


def test_compare_threshold():
    assert compare_threshold(10, "gt", 5) is True
    assert compare_threshold(5, "gt", 5) is False
    assert compare_threshold(5, "gte", 5) is True
    assert compare_threshold(3, "lt", 5) is True


def test_consecutive_breaches():
    assert consecutive_breaches([False, True, True, True], 3) is True
    assert consecutive_breaches([True, True, False], 3) is False


def test_cooldown():
    now = datetime.now(UTC)
    assert should_fire_with_cooldown(last_notified_at=None, now=now, cooldown_seconds=60) is True
    assert (
        should_fire_with_cooldown(
            last_notified_at=now - timedelta(seconds=10),
            now=now,
            cooldown_seconds=60,
        )
        is False
    )
    assert (
        should_fire_with_cooldown(
            last_notified_at=now - timedelta(seconds=120),
            now=now,
            cooldown_seconds=60,
        )
        is True
    )


def test_ewma_and_mad_need_history():
    assert ewma_anomaly([1.0, 1.1])[0] is False
    assert mad_zscore([1.0, 1.0])[0] is False
    values = [1.0, 1.0, 1.0, 1.0, 1.0, 50.0]
    anomalous, _, _ = ewma_anomaly(values, z_threshold=3.0)
    assert anomalous is True
    mad_hit, _ = mad_zscore(values, z_threshold=3.0)
    assert mad_hit is True


def test_redaction_strips_secrets_and_conversations():
    payload = {
        "authorization": "Bearer supersecret",
        "password": "x",
        "transcript": "student said hello",
        "nested": {"api_key": "abc", "ok": 1},
        "message": "Bearer eyJhbGciOiJIUzI1NiJ9.abc",
    }
    out = redact_obj(payload)
    assert out["authorization"] == "[REDACTED]"
    assert out["password"] == "[REDACTED]"
    assert out["transcript"] == "[REDACTED]"
    assert out["nested"]["api_key"] == "[REDACTED]"
    assert out["nested"]["ok"] == 1
    assert "[REDACTED]" in out["message"]
    assert out["transcript"] == "[REDACTED]"
    assert "Bearer" in preview_message("Bearer eyJhbGciOiJIUzI1NiJ9.abc")
    assert "[REDACTED]" in preview_message("Bearer eyJhbGciOiJIUzI1NiJ9.abc")
