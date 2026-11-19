"""Fail-open ASGI middleware: request latency + error counters for monitoring."""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.modules.monitoring.logging_utils import (
    current_request_id,
    current_trace_id,
    new_request_id,
    structured_log,
)

logger = logging.getLogger(__name__)

_SKIP_PREFIXES = ("/health", "/metrics", "/ws/", "/rtc/")


def _monitoring_enabled() -> bool:
    return os.environ.get("MONITORING_ENABLED", "false").lower() in ("1", "true", "yes")


class MonitoringMetricsMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _monitoring_enabled():
            await self.app(scope, receive, send)
            return

        path = scope.get("path") or ""
        if any(path.startswith(p) for p in _SKIP_PREFIXES):
            await self.app(scope, receive, send)
            return

        req_id = new_request_id()
        trace_id = req_id
        for key, value in scope.get("headers") or []:
            if key.lower() == b"x-request-id":
                req_id = value.decode("latin-1")[:64]
            if key.lower() == b"x-trace-id":
                trace_id = value.decode("latin-1")[:64]

        tok_r = current_request_id.set(req_id)
        tok_t = current_trace_id.set(trace_id)
        status_code = 500
        t0 = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message.get("status") or 500)
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", req_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000
            try:
                _record_request_metric(
                    path=path,
                    method=(scope.get("method") or "GET").upper(),
                    status_code=status_code,
                    duration_ms=duration_ms,
                    request_id=req_id,
                    trace_id=trace_id,
                )
            except Exception as exc:
                logger.debug("monitoring metric write skipped: %s", exc)
            current_request_id.reset(tok_r)
            current_trace_id.reset(tok_t)


def _record_request_metric(
    *,
    path: str,
    method: str,
    status_code: int,
    duration_ms: float,
    request_id: str,
    trace_id: str,
) -> None:
    """Never raises to callers — swallows all DB errors."""
    structured_log(
        "error" if status_code >= 500 else "info",
        f"{method} {path} -> {status_code}",
        service="api",
        module="http",
        duration_ms=duration_ms,
        extra={"status_code": status_code, "path": path, "method": method},
    )
    try:
        from sqlalchemy import select

        from app.core.database import SessionLocal
        from app.modules.monitoring.models import MonLogIndex, MonMetricAggregate

        bucket = datetime.now(UTC).replace(second=0, microsecond=0)
        is_error = 1 if status_code >= 500 else 0
        db = SessionLocal()
        try:
            db.add(
                MonLogIndex(
                    ts=datetime.now(UTC),
                    environment=os.environ.get("ENVIRONMENT", "production"),
                    service="api",
                    module="http",
                    severity="error" if is_error else "info",
                    message_preview=f"{method} {path} -> {status_code}"[:500],
                    request_id=request_id,
                    trace_id=trace_id,
                    duration_ms=duration_ms,
                )
            )
            _upsert_metric(db, bucket, "http.request.duration_ms", duration_ms)
            _upsert_metric(db, bucket, "http.request.count", 1.0)
            if is_error:
                _upsert_metric(db, bucket, "http.request.errors", 1.0)
            db.commit()
        finally:
            db.close()
    except Exception:
        # Fail-open: monitoring must never break requests
        pass


def _upsert_metric(db: Any, bucket: datetime, metric_key: str, value: float) -> None:
    from sqlalchemy import select

    from app.modules.monitoring.models import MonMetricAggregate

    row = db.execute(
        select(MonMetricAggregate).where(
            MonMetricAggregate.bucket_start == bucket,
            MonMetricAggregate.metric_key == metric_key,
            MonMetricAggregate.service_key == "api",
            MonMetricAggregate.environment == os.environ.get("ENVIRONMENT", "production"),
        )
    ).scalar_one_or_none()
    if row is None:
        db.add(
            MonMetricAggregate(
                bucket_start=bucket,
                bucket_seconds=60,
                metric_key=metric_key,
                service_key="api",
                environment=os.environ.get("ENVIRONMENT", "production"),
                count=1,
                sum_value=float(value),
                min_value=float(value),
                max_value=float(value),
                p50=float(value),
                p95=float(value),
                p99=float(value),
            )
        )
        return
    row.count = int(row.count or 0) + 1
    row.sum_value = float(row.sum_value or 0) + float(value)
    row.min_value = float(value) if row.min_value is None else min(float(row.min_value), float(value))
    row.max_value = float(value) if row.max_value is None else max(float(row.max_value), float(value))
    # Cheap running estimate for percentiles until a full hist job runs.
    avg = row.sum_value / max(row.count, 1)
    row.p50 = avg
    row.p95 = row.max_value
    row.p99 = row.max_value
