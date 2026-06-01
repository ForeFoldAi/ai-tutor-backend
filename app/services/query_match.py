"""
Keyword-style scoring helpers for RAG when semantic search is unavailable
or as a supplement. Reduces false positives from page numbers (e.g. "1")
matching almost every chunk.
"""

from __future__ import annotations

import re


def query_terms_for_keyword_match(query: str) -> set[str]:
    """
    Tokenize the query and drop tokens that dominate scoring but carry little meaning:
    - single-character tokens
    - short pure-digit tokens (page numbers / list markers like "1", "2")
    """
    terms = set(re.findall(r"\w+", query.lower()))
    out: set[str] = set()
    for t in terms:
        if len(t) < 2:
            continue
        if t.isdigit() and len(t) <= 3:
            continue
        out.add(t)
    return out


def chapter_phrase_bonus(query: str, text: str) -> int:
    """
    Strong bonus when the chunk contains the same chapter reference as the query
    (e.g. query mentions chapter 1 and chunk has 'CHAPTER 1' / 'chapter 1').
    """
    q = query.lower()
    t = (text or "").lower()
    bonus = 0
    for m in re.finditer(r"chapter\s*(\d+)", q, flags=re.I):
        n = m.group(1)
        if re.search(rf"chapter\s*{n}\b", t, flags=re.I):
            bonus += 18
    return bonus


def keyword_match_score(query: str, text: str) -> int:
    """Higher is better."""
    t = (text or "").lower()
    score = sum(1 for term in query_terms_for_keyword_match(query) if term in t)
    score += chapter_phrase_bonus(query, text)
    return score


def document_page(doc) -> int:
    """Best-effort page index from LangChain Document metadata (PyPDFLoader)."""
    meta = getattr(doc, "metadata", None) or {}
    try:
        return int(meta.get("page", 0) or 0)
    except Exception:
        return 0
