"""Threshold, EWMA, MAD/z-score, fingerprinting, and cooldown helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from statistics import median

_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.I,
)
_HEX_ID = re.compile(r"\b[0-9a-f]{16,}\b", re.I)
_NUM = re.compile(r"\b\d+\b")


def normalize_error_message(message: str) -> str:
    text = message.strip().lower()
    text = _UUID.sub("<uuid>", text)
    text = _HEX_ID.sub("<id>", text)
    text = _NUM.sub("<n>", text)
    return re.sub(r"\s+", " ", text)[:400]


def fingerprint_error(
    *,
    title: str,
    culprit: str | None = None,
    service: str | None = None,
    environment: str | None = None,
) -> str:
    base = "|".join(
        [
            normalize_error_message(title),
            normalize_error_message(culprit or ""),
            (service or "").lower(),
            (environment or "production").lower(),
        ]
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:40]


def compare_threshold(value: float, comparator: str, threshold: float) -> bool:
    if comparator == "gt":
        return value > threshold
    if comparator == "gte":
        return value >= threshold
    if comparator == "lt":
        return value < threshold
    if comparator == "lte":
        return value <= threshold
    if comparator == "eq":
        return abs(value - threshold) < 1e-9
    return False


def ewma_anomaly(
    values: list[float],
    *,
    alpha: float = 0.3,
    z_threshold: float = 3.0,
) -> tuple[bool, float | None, float | None]:
    """Return (is_anomaly, ewma, last_value). Needs >= 5 points."""
    if len(values) < 5:
        return False, None, values[-1] if values else None
    ewma = values[0]
    residuals: list[float] = []
    for v in values[1:]:
        ewma = alpha * v + (1 - alpha) * ewma
        residuals.append(abs(v - ewma))
    if not residuals:
        return False, ewma, values[-1]
    mad = median(residuals) or 1e-6
    last = values[-1]
    score = abs(last - ewma) / (1.4826 * mad)
    return score >= z_threshold, ewma, last


def mad_zscore(values: list[float], *, z_threshold: float = 3.5) -> tuple[bool, float | None]:
    if len(values) < 5:
        return False, None
    med = median(values)
    deviations = [abs(v - med) for v in values]
    mad = median(deviations) or 1e-6
    last = values[-1]
    score = abs(last - med) / (1.4826 * mad)
    return score >= z_threshold, score


def should_fire_with_cooldown(
    *,
    last_notified_at: datetime | None,
    now: datetime,
    cooldown_seconds: int,
) -> bool:
    if last_notified_at is None:
        return True
    delta = (now - last_notified_at).total_seconds()
    return delta >= max(0, cooldown_seconds)


def consecutive_breaches(flags: list[bool], required: int) -> bool:
    if required <= 0:
        return False
    if len(flags) < required:
        return False
    return all(flags[-required:])
