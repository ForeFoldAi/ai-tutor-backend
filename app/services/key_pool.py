"""Rotate several API keys of one provider and skip the ones that are rate-limited.

Free tiers cap each key (Groq: tokens/minute, requests/day). With several keys,
requests are spread round-robin; a key that returns 429/5xx is parked for the
duration the provider asked for, and traffic moves on to the next key. Only when
every key is parked does the caller go to the secondary provider.
"""

from __future__ import annotations

import itertools
import threading
import time

_lock = threading.Lock()
_ready_at: dict[tuple[str, str], float] = {}  # (scope, key) -> monotonic time it is usable again
_counter = itertools.count()


def cool(scope: str, key: str, seconds: float) -> None:
    """Park ``key`` for ``seconds`` (never shortens an existing, longer park)."""
    until = time.monotonic() + max(0.0, seconds)
    with _lock:
        k = (scope, key)
        _ready_at[k] = max(_ready_at.get(k, 0.0), until)


def is_cooling(scope: str, key: str) -> bool:
    with _lock:
        return _ready_at.get((scope, key), 0.0) > time.monotonic()


def ordered(scope: str, keys: list[str]) -> list[str]:
    """Round-robin start, usable keys first, then parked keys soonest-ready first."""
    if len(keys) <= 1:
        return list(keys)
    start = next(_counter) % len(keys)
    rotated = keys[start:] + keys[:start]
    now = time.monotonic()
    with _lock:
        ready = [k for k in rotated if _ready_at.get((scope, k), 0.0) <= now]
        parked = sorted(
            (k for k in rotated if _ready_at.get((scope, k), 0.0) > now),
            key=lambda k: _ready_at[(scope, k)],
        )
    return ready + parked


def label(key: str) -> str:
    """Log-safe identifier (never the full key)."""
    return f"…{key[-4:]}" if len(key) > 4 else "…"


def reset() -> None:
    """Tests only."""
    with _lock:
        _ready_at.clear()
