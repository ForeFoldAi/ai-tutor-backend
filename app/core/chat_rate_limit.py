"""Per-user sliding-window limit on tutor chat turns.

Every chat turn spends shared LLM quota; one runaway client (a stuck retry loop,
a script, a held-down Enter key) must not starve every other student.
In-process only: with several API replicas each one enforces its own window,
which is still a hard ceiling per replica.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException

CHAT_RATE_LIMIT_MESSAGE = (
    "You're sending questions very quickly. Please wait a few seconds and try again."
)


def _limit_per_min() -> int:
    try:
        return int(os.environ.get("CHAT_RATE_LIMIT_PER_MIN", "20"))
    except ValueError:
        return 20


_WINDOW_SEC = 60.0
_lock = threading.Lock()
_hits: dict[str, deque[float]] = defaultdict(deque)


def check_chat_rate_limit(user_key: object) -> None:
    """Raise HTTP 429 when *user_key* exceeded CHAT_RATE_LIMIT_PER_MIN in the last minute.

    CHAT_RATE_LIMIT_PER_MIN=0 disables the check.
    """
    limit = _limit_per_min()
    if limit <= 0:
        return
    key = str(user_key)
    now = time.monotonic()
    with _lock:
        q = _hits[key]
        while q and now - q[0] >= _WINDOW_SEC:
            q.popleft()
        if len(q) >= limit:
            retry = max(1, int(_WINDOW_SEC - (now - q[0])) + 1)
            raise HTTPException(
                status_code=429,
                detail=CHAT_RATE_LIMIT_MESSAGE,
                headers={"Retry-After": str(retry)},
            )
        q.append(now)
        # Drop idle users so the map cannot grow without bound.
        if len(_hits) > 10_000:
            for k in [k for k, v in _hits.items() if not v or now - v[-1] >= _WINDOW_SEC]:
                _hits.pop(k, None)


def reset_chat_rate_limit() -> None:
    """Test helper."""
    with _lock:
        _hits.clear()
