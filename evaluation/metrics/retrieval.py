"""Retrieval IR metrics."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def precision_at_k(relevant: set[int], ranked: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    hits = sum(1 for x in top if x in relevant)
    return hits / k


def recall_at_k(relevant: set[int], ranked: Sequence[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for x in top if x in relevant)
    return hits / len(relevant)


def hit_at_k(relevant: set[int], ranked: Sequence[int], k: int) -> float:
    return 1.0 if any(x in relevant for x in ranked[:k]) else 0.0


def mrr(relevant: set[int], ranked: Sequence[int]) -> float:
    for i, x in enumerate(ranked, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    s = 0.0
    for i, g in enumerate(gains[:k], start=1):
        s += g / math.log2(i + 1)
    return s


def ndcg_at_k(relevant: set[int], ranked: Sequence[int], k: int) -> float:
    gains = [1.0 if x in relevant else 0.0 for x in ranked[:k]]
    ideal = sorted(gains, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def page_match(expected_pages: Sequence[int], actual_pages: Sequence[int | None], k: int = 5) -> bool:
    exp = {int(p) for p in expected_pages}
    act = {int(p) for p in actual_pages[:k] if p is not None}
    return bool(exp & act)
