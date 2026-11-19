"""Sensitive-data redaction for logs, issues, and alert payloads."""

from __future__ import annotations

import re
from typing import Any

_SECRET_KEYS = re.compile(
    r"(password|passwd|secret|token|authorization|api[_-]?key|cookie|credential|refresh|jwt|bearer)",
    re.I,
)
_BEARER = re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.I)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CONV_KEYS = re.compile(r"(transcript|audio|conversation|message_body|pcm|recording)", re.I)


def redact_string(value: str) -> str:
    value = _BEARER.sub(r"\1[REDACTED]", value)
    if len(value) > 2000:
        value = value[:2000] + "…[truncated]"
    return value


def redact_obj(obj: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "[MAX_DEPTH]"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return redact_string(obj)
    if isinstance(obj, list):
        return [redact_obj(x, depth=depth + 1) for x in obj[:200]]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in list(obj.items())[:200]:
            key = str(k)
            if _SECRET_KEYS.search(key) or _CONV_KEYS.search(key):
                out[key] = "[REDACTED]"
            else:
                out[key] = redact_obj(v, depth=depth + 1)
        return out
    return str(obj)[:500]


def preview_message(message: str | None, limit: int = 500) -> str:
    if not message:
        return ""
    return redact_string(message)[:limit]
