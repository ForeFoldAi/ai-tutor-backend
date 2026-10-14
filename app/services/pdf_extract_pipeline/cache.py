"""In-process cache for pipeline results (avoids double extraction per upload)."""

from __future__ import annotations

import os
from typing import Any

_cache: dict[str, Any] = {}


def _cache_key(file_path: str) -> str | None:
    if not file_path or not os.path.isfile(file_path):
        return None
    try:
        stat = os.stat(file_path)
        return f"{os.path.abspath(file_path)}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return None


def get_cached_pipeline_result(file_path: str) -> Any | None:
    key = _cache_key(file_path)
    if not key:
        return None
    return _cache.get(key)


def set_cached_pipeline_result(file_path: str, result: Any) -> None:
    key = _cache_key(file_path)
    if key:
        _cache[key] = result


def clear_pipeline_cache() -> None:
    _cache.clear()
