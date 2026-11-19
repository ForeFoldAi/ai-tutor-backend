"""Grafana Loki query proxy (server-side only)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

import httpx

from app.modules.monitoring.redaction import redact_obj

logger = logging.getLogger(__name__)


class LokiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 8.0) -> None:
        self.base_url = (base_url or os.environ.get("LOKI_URL") or "").rstrip("/")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def query_range(
        self,
        *,
        query: str,
        start: datetime,
        end: datetime,
        limit: int = 100,
        direction: str = "BACKWARD",
    ) -> dict[str, Any]:
        if not self.configured:
            return {"unavailable": True, "result": [], "message": "LOKI_URL not configured"}
        params = {
            "query": query,
            "start": int(start.timestamp() * 1e9),
            "end": int(end.timestamp() * 1e9),
            "limit": max(1, min(limit, 1000)),
            "direction": direction,
        }
        url = f"{self.base_url}/loki/api/v1/query_range?{urlencode(params)}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
            streams = (data.get("data") or {}).get("result") or []
            entries: list[dict[str, Any]] = []
            for stream in streams:
                labels = stream.get("stream") or {}
                for ts_ns, line in stream.get("values") or []:
                    raw: dict[str, Any]
                    try:
                        raw = json.loads(line) if line.startswith("{") else {"message": line}
                    except Exception:
                        raw = {"message": line}
                    raw = redact_obj(raw) if isinstance(raw, dict) else {"message": str(raw)}
                    entries.append(
                        {
                            "ts_ns": ts_ns,
                            "labels": labels,
                            "raw": raw,
                            "message": str(raw.get("message") or line)[:2000],
                            "severity": raw.get("severity") or labels.get("level") or "info",
                            "service": raw.get("service") or labels.get("service") or labels.get("app"),
                            "module": raw.get("module"),
                            "request_id": raw.get("request_id"),
                            "trace_id": raw.get("trace_id"),
                            "release": raw.get("release"),
                            "environment": raw.get("environment") or labels.get("env"),
                            "duration_ms": raw.get("duration_ms"),
                        }
                    )
            return {"unavailable": False, "result": entries, "message": None}
        except Exception as exc:
            logger.warning("Loki query failed (non-fatal): %s", exc)
            return {"unavailable": True, "result": [], "message": str(exc)}


def get_loki_client() -> LokiClient:
    return LokiClient()
