from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz

_STOP = frozenset(
    "the a an and or of in on at to for with is are was were be been being this that these those".split()
)


def extract_key_concepts(text: str, *, top_k: int = 12) -> list[str]:
    """Extract candidate concepts from textbook context via n-gram frequency."""
    if not text:
        return []
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text) if w.lower() not in _STOP]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    counts = Counter(words + bigrams)
    ranked = [term for term, _ in counts.most_common(top_k * 2)]
    # De-dupe near-synonyms
    selected: list[str] = []
    for term in ranked:
        if any(fuzz.ratio(term, s) > 85 for s in selected):
            continue
        selected.append(term)
        if len(selected) >= top_k:
            break
    return selected
