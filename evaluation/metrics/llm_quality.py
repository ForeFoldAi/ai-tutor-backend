"""LLM / tutor response quality metrics (rule-based + optional judge)."""

from __future__ import annotations

import re
from typing import Any


_CITATION_RE = re.compile(r"(?:page\s*\d+|fig(?:ure)?\.?\s*\d+|chapter\s*\d+)", re.I)
_UNSAFE_RE = re.compile(
    r"\b(kill yourself|make a bomb|hack into|credit card number)\b",
    re.I,
)


def groundedness(answer: str, contexts: list[str], *, min_overlap_tokens: int = 3) -> float:
    """Lexical groundedness: fraction of answer content tokens found in context."""
    ans_tokens = _tokens(answer)
    if not ans_tokens:
        return 0.0
    ctx = " ".join(contexts or []).lower()
    if not ctx.strip():
        return 0.0
    hits = sum(1 for t in ans_tokens if t in ctx)
    return hits / len(ans_tokens)


def faithfulness_proxy(answer: str, contexts: list[str], must_include: list[str] | None = None) -> float:
    """
    Proxy faithfulness: required phrases present + low contradiction markers.
    Not a full NLI judge — use llm_judge for deeper checks when enabled.
    """
    score = 1.0
    low = (answer or "").lower()
    ctx = " ".join(contexts or []).lower()
    for phrase in must_include or []:
        if phrase.lower() not in low and phrase.lower() not in ctx:
            score -= 0.25
    # Hedging without context often signals hallucination
    if re.search(r"\b(as an ai|i don't have access to the (textbook|chapter))\b", low):
        score -= 0.2
    return max(0.0, min(1.0, score))


def hallucination_flags(answer: str, contexts: list[str], forbidden: list[str] | None = None) -> list[str]:
    flags: list[str] = []
    low = (answer or "").lower()
    ctx = " ".join(contexts or []).lower()
    for term in forbidden or []:
        if term.lower() in low and term.lower() not in ctx:
            flags.append(f"unsupported_term:{term}")
    # Numeric claims not in context
    for num in re.findall(r"\b\d{4}\b", answer or ""):
        if num not in ctx:
            flags.append(f"unsupported_year:{num}")
    return flags


def curriculum_alignment(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    low = (answer or "").lower()
    hits = sum(1 for k in expected_keywords if k.lower() in low)
    return hits / len(expected_keywords)


def grade_appropriateness(answer: str, *, max_avg_word_len: float = 8.5) -> float:
    words = re.findall(r"[A-Za-z]+", answer or "")
    if not words:
        return 0.0
    avg = sum(len(w) for w in words) / len(words)
    # Soft score around readability for middle-school
    if avg <= max_avg_word_len:
        return 1.0
    return max(0.0, 1.0 - (avg - max_avg_word_len) / 5.0)


def safety_ok(answer: str) -> bool:
    return not bool(_UNSAFE_RE.search(answer or ""))


def citation_quality(answer: str) -> float:
    if not answer:
        return 0.0
    return 1.0 if _CITATION_RE.search(answer) else 0.5


def length_ok(answer: str, *, min_chars: int = 40, max_chars: int = 8000) -> bool:
    n = len(answer or "")
    return min_chars <= n <= max_chars


def score_tutor_response(
    answer: str,
    *,
    contexts: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    must_include: list[str] | None = None,
    forbidden: list[str] | None = None,
) -> dict[str, Any]:
    contexts = contexts or []
    g = groundedness(answer, contexts)
    f = faithfulness_proxy(answer, contexts, must_include=must_include)
    hall = hallucination_flags(answer, contexts, forbidden=forbidden)
    align = curriculum_alignment(answer, expected_keywords or [])
    grade = grade_appropriateness(answer)
    safe = safety_ok(answer)
    cite = citation_quality(answer)
    length = length_ok(answer)
    # Composite
    hall_penalty = min(0.5, 0.1 * len(hall))
    overall = (
        0.25 * g
        + 0.25 * f
        + 0.2 * align
        + 0.1 * grade
        + 0.1 * cite
        + (0.1 if length else 0.0)
        - hall_penalty
    )
    if not safe:
        overall = 0.0
    overall = max(0.0, min(1.0, overall))
    return {
        "groundedness": g,
        "faithfulness": f,
        "hallucination_flags": hall,
        "hallucination_rate": 1.0 if hall else 0.0,
        "curriculum_alignment": align,
        "grade_appropriateness": grade,
        "safety_ok": safe,
        "citation_quality": cite,
        "length_ok": length,
        "tutor_quality_score": overall,
    }


def llm_judge_answer(
    question: str,
    answer: str,
    contexts: list[str],
    *,
    enabled: bool = True,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    try:
        from app.services.llm_client import complete_sync

        prompt = (
            "You are an educational QA judge. Score the tutor answer.\n"
            "Return ONLY JSON with keys: groundedness, faithfulness, correctness, "
            "hallucination (0-1), reason (short).\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{chr(10).join(contexts)[:3000]}\n\n"
            f"Answer:\n{answer[:3000]}\n"
        )
        raw = complete_sync(
            [{"role": "user", "content": prompt}],
            feature="chat",
            max_tokens=512,
        )
        import json

        m = re.search(r"\{.*\}", raw or "", re.S)
        if not m:
            return {"raw": raw, "parse_ok": False}
        data = json.loads(m.group(0))
        data["parse_ok"] = True
        return data
    except Exception as exc:  # noqa: BLE001
        return {"parse_ok": False, "error": str(exc)}


def _tokens(text: str) -> list[str]:
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are",
        "was", "were", "be", "as", "by", "with", "that", "this", "it", "from",
    }
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if t not in stop and len(t) > 2]
