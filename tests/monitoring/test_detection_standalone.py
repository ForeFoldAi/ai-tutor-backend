"""Standalone detection tests runnable without pytest/fastapi if needed."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.modules.monitoring.detection.engine import (  # noqa: E402
    compare_threshold,
    fingerprint_error,
    should_fire_with_cooldown,
)
from app.modules.monitoring.redaction import redact_obj  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402


def main() -> None:
    assert fingerprint_error(title="err 1", service="api") == fingerprint_error(title="err 2", service="api")
    assert compare_threshold(9, "gt", 3)
    now = datetime.now(UTC)
    assert should_fire_with_cooldown(last_notified_at=now - timedelta(seconds=10), now=now, cooldown_seconds=60) is False
    assert redact_obj({"token": "abc", "ok": True})["token"] == "[REDACTED]"
    print("test_detection_standalone: ok")


if __name__ == "__main__":
    main()
