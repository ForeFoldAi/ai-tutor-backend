"""In-process cache for pipeline results (avoids double extraction per upload)."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

# ponytail: the text+layout pass and the image pass of one chapter share a result, so only the
# most recent few are needed; anything larger just holds RAM across a whole book.
_MAX_ENTRIES = 2
_cache: OrderedDict[str, Any] = OrderedDict()


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
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)


def clear_pipeline_cache() -> None:
    _cache.clear()
