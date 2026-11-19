"""Helpers for normalizing Sentry webhook payloads into ingest events."""

from __future__ import annotations

from typing import Any

from app.modules.monitoring.schemas import IngestEventIn


def sentry_payload_to_ingest(body: dict[str, Any]) -> IngestEventIn:
    event = body.get("event") or body.get("data", {}).get("event") or body
    title = str(event.get("title") or event.get("message") or "Sentry event")[:500]
    tags = event.get("tags") or {}
    service = "api"
    if isinstance(tags, dict):
        service = str(tags.get("service") or "api")
    elif isinstance(tags, list):
        for item in tags:
            if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == "service":
                service = str(item[1])
    return IngestEventIn(
        source="sentry",
        service=service,
        environment=str(event.get("environment") or "production"),
        severity=str(event.get("level") or "error"),
        title=title,
        message=str(event.get("message") or title),
        stacktrace=event.get("exception") if isinstance(event.get("exception"), dict) else None,
        breadcrumbs=event.get("breadcrumbs") if isinstance(event.get("breadcrumbs"), list) else None,
        request=event.get("request") if isinstance(event.get("request"), dict) else None,
        user_ref=str((event.get("user") or {}).get("id") or "") or None,
        release=str(event.get("release") or "") or None,
        meta={"sentry_id": event.get("event_id") or body.get("id")},
    )
