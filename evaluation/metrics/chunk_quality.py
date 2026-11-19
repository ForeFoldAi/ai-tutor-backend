"""Chunk quality scoring."""

from __future__ import annotations

import re
from typing import Any


_BROKEN_LIST = re.compile(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$", re.M)


def token_len(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text or ""))
    except Exception:
        return max(1, len(text or "") // 4)


def score_chunk(
    text: str,
    meta: dict[str, Any] | None = None,
    *,
    min_tokens: int = 50,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    meta = meta or {}
    issues: list[str] = []
    t = text or ""
    n = token_len(t)
    if n < min_tokens:
        issues.append("too_small")
    if n > max_tokens:
        issues.append("too_large")
    # Broken paragraph: ends mid-word / trailing hyphen commonly from bad splits
    if re.search(r"[A-Za-z]-\s*$", t.strip()):
        issues.append("broken_paragraph_hyphen")
    if _BROKEN_LIST.search(t):
        issues.append("broken_list")
    # Split table heuristic: odd number of markdown table rows / single header only
    rows = _TABLE_ROW.findall(t)
    if rows and len(rows) == 1:
        issues.append("possible_split_table")
    # Figure without caption association when figure mention present
    if re.search(r"\b(?:fig(?:ure)?\.?\s*\d)", t, re.I) and "caption" not in t.lower():
        # soft signal only
        if not meta.get("figure_ids") and not meta.get("caption"):
            issues.append("figure_mention_without_association")

    score = 1.0 - 0.2 * len(issues)
    score = max(0.0, min(1.0, score))
    return {
        "tokens": n,
        "issues": issues,
        "score": score,
        "has_section_hint": bool(meta.get("section_hint")),
        "page": meta.get("page"),
    }


def corpus_chunk_quality(chunks: list[Any], **kwargs) -> dict[str, Any]:
    scored = []
    for c in chunks:
        text = getattr(c, "page_content", "") or ""
        meta = dict(getattr(c, "metadata", None) or {})
        scored.append(score_chunk(text, meta, **kwargs))
    if not scored:
        return {"mean_score": 0.0, "count": 0, "issues_total": 0}
    mean = sum(s["score"] for s in scored) / len(scored)
    issues_total = sum(len(s["issues"]) for s in scored)
    return {
        "mean_score": mean,
        "count": len(scored),
        "issues_total": issues_total,
        "too_large": sum(1 for s in scored if "too_large" in s["issues"]),
        "too_small": sum(1 for s in scored if "too_small" in s["issues"]),
        "with_section_hint": sum(1 for s in scored if s["has_section_hint"]),
        "per_chunk": scored,
    }
