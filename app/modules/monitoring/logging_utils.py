"""Structured JSON logging helpers for monitoring correlation fields."""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from app.modules.monitoring.redaction import preview_message, redact_obj

current_request_id: ContextVar[str | None] = ContextVar("mon_request_id", default=None)
current_trace_id: ContextVar[str | None] = ContextVar("mon_trace_id", default=None)

_logger = logging.getLogger("learnmestro.monitoring")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def structured_log(
    severity: str,
    message: str,
    *,
    service: str = "api",
    module: str | None = None,
    duration_ms: float | None = None,
    release: str | None = None,
    environment: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "environment": environment or os.environ.get("ENVIRONMENT", "production"),
        "service": service,
        "module": module,
        "severity": severity,
        "message": preview_message(message, 2000),
        "request_id": current_request_id.get(),
        "trace_id": current_trace_id.get(),
        "release": release or os.environ.get("RELEASE_SHA") or os.environ.get("GIT_SHA"),
        "duration_ms": duration_ms,
    }
    if extra:
        payload["extra"] = redact_obj(extra)
    line = {k: v for k, v in payload.items() if v is not None}
    try:
        _logger.log(
            getattr(logging, severity.upper(), logging.INFO),
            "%s",
            json.dumps(line, default=str),
        )
    except Exception:
        pass
    return line
