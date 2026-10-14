"""Production algorithms for LIA — mastery, understanding, style, risk."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from app.modules.learning_intelligence.constants import (
    GAP_THRESHOLD,
    MASTERY_STRONG,
    MASTERY_WEAK,
    STYLE_ANIMATION,
    STYLE_CONVERSATION,
    STYLE_EXAMPLE,
    STYLE_EXPERIMENT,
    STYLE_PRACTICE,
    STYLE_READING,
    STYLE_STORY,
    STYLE_VISUAL,
    UNDERSTANDING_ANALYZE,
    UNDERSTANDING_APPLY,
    UNDERSTANDING_CREATE,
    UNDERSTANDING_EVALUATE,
    UNDERSTANDING_REMEMBER,
    UNDERSTANDING_UNDERSTAND,
)
from app.services.lesson_planner.algorithms.bloom import BloomLevel, classify_bloom

# Evidence weights for Bayesian Knowledge Tracing lite (single-parameter EMA)
_EVIDENCE_WEIGHTS: dict[str, float] = {
    "correct": 0.15,
    "incorrect": -0.12,
    "confusion": -0.08,
    "affirmation": 0.10,
    "hint_used": -0.05,
    "retry_success": 0.08,
    "assignment_correct": 0.18,
    "assignment_incorrect": -0.14,
    "chapter_complete": 0.12,
}


def concept_key_from_topic(topic: str, *, subject: str = "", chapter: str = "") -> str:
    """Stable slug for a learning topic."""
    raw = f"{subject}|{chapter}|{(topic or '').strip().lower()}"
    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if len(slug) > 200:
        slug = slug[:200]
    return slug or "general"


def topic_hash(topic: str, scope_key: str = "") -> str:
    return hashlib.sha256(f"{scope_key}:{topic}".encode()).hexdigest()[:32]


def update_mastery_ema(current: float, evidence_type: str, *, magnitude: float = 1.0) -> float:
    """EMA mastery update — production-safe, bounded [0, 1]."""
    delta = _EVIDENCE_WEIGHTS.get(evidence_type, 0.0) * magnitude
    return max(0.0, min(1.0, current + delta))


def bloom_to_understanding_level(bloom: BloomLevel | str) -> int:
    mapping = {
        BloomLevel.REMEMBER: UNDERSTANDING_REMEMBER,
        BloomLevel.UNDERSTAND: UNDERSTANDING_UNDERSTAND,
        BloomLevel.APPLY: UNDERSTANDING_APPLY,
        BloomLevel.ANALYZE: UNDERSTANDING_ANALYZE,
        BloomLevel.EVALUATE: UNDERSTANDING_EVALUATE,
        BloomLevel.CREATE: UNDERSTANDING_CREATE,
    }
    if isinstance(bloom, str):
        try:
            bloom = BloomLevel(bloom)
        except ValueError:
            return UNDERSTANDING_UNDERSTAND
    return mapping.get(bloom, UNDERSTANDING_UNDERSTAND)


def infer_understanding_level(
    *,
    mastery_score: float,
    query: str = "",
    is_correct: bool | None = None,
    response_length: int = 0,
) -> int:
    """Depth 1–6 from mastery + Bloom signals + performance."""
    base = bloom_to_understanding_level(classify_bloom(query))
    if is_correct is True and mastery_score >= MASTERY_STRONG:
        base = min(6, base + 1)
    elif is_correct is False:
        base = max(1, base - 1)
    if mastery_score < MASTERY_WEAK:
        base = min(base, UNDERSTANDING_UNDERSTAND)
    if response_length >= 12 and mastery_score >= 0.6:
        base = min(6, base + 1)
    return base


def memorized_likelihood(
    *,
    mastery_score: float,
    understanding_level: int,
    fast_correct: bool = False,
    slow_incorrect: bool = False,
) -> float:
    """High mastery + low understanding depth → likely memorization."""
    if mastery_score >= 0.7 and understanding_level <= UNDERSTANDING_REMEMBER:
        return 0.75
    if fast_correct and understanding_level <= UNDERSTANDING_UNDERSTAND:
        return 0.65
    if slow_incorrect and mastery_score < 0.4:
        return 0.2
    return max(0.0, min(1.0, mastery_score - (understanding_level - 1) * 0.1))


def infer_learning_style_signals(events: list[dict[str, Any]]) -> dict[str, float]:
    """Weighted behavioural inference — no student questionnaire."""
    scores: dict[str, float] = {
        STYLE_VISUAL: 0.0,
        STYLE_CONVERSATION: 0.0,
        STYLE_EXPERIMENT: 0.0,
        STYLE_READING: 0.0,
        STYLE_PRACTICE: 0.0,
        STYLE_STORY: 0.0,
        STYLE_EXAMPLE: 0.0,
        STYLE_ANIMATION: 0.0,
    }
    for ev in events:
        et = ev.get("event_type", "")
        payload = ev.get("payload") or {}
        if et in ("chat_user_question", "voice_user_utterance"):
            scores[STYLE_CONVERSATION] += 1.0
        if payload.get("images_viewed") or payload.get("animation_used"):
            scores[STYLE_VISUAL] += 1.5
            scores[STYLE_ANIMATION] += 1.2
        if payload.get("science_experiment"):
            scores[STYLE_EXPERIMENT] += 2.0
        if et == "assignment_submitted" or payload.get("wants_quiz"):
            scores[STYLE_PRACTICE] += 1.3
        if payload.get("reading_time_sec", 0) > 60:
            scores[STYLE_READING] += 1.4
        if payload.get("story_example"):
            scores[STYLE_STORY] += 1.0
        if payload.get("real_world_example"):
            scores[STYLE_EXAMPLE] += 1.2
    total = sum(scores.values()) or 1.0
    return {k: round(v / total, 3) for k, v in scores.items()}


def dominant_learning_style(style_scores: dict[str, float]) -> str:
    if not style_scores:
        return "balanced"
    best = max(style_scores, key=style_scores.get)
    if style_scores[best] < 0.15:
        return "balanced"
    return best


def update_confidence_ema(current: float, signal: float) -> float:
    """signal in [0,1] — confusion low, affirmation high."""
    alpha = 0.2
    return max(0.0, min(1.0, current * (1 - alpha) + signal * alpha))


def attention_score_from_signals(
    *,
    idle_ratio: float = 0.0,
    session_minutes: float = 0.0,
    topic_switches: int = 0,
) -> float:
    """Lower idle + moderate session + few switches → higher attention."""
    base = 0.7
    base -= min(0.4, idle_ratio * 0.5)
    base -= min(0.2, topic_switches * 0.05)
    if session_minutes > 90:
        base -= 0.15
    elif 10 <= session_minutes <= 45:
        base += 0.1
    return max(0.0, min(1.0, base))


def engagement_score_from_events(event_count: int, *, days: int = 7) -> float:
    if days <= 0:
        return 0.5
    daily = event_count / days
    return max(0.0, min(1.0, daily / 20.0))


def compute_risk(
    *,
    mastery_avg: float,
    confidence: float,
    engagement: float,
    regression_trend: float,
) -> tuple[str, float]:
    """Returns (risk_level, risk_score 0–1)."""
    score = (
        (1.0 - mastery_avg) * 0.35
        + (1.0 - confidence) * 0.25
        + (1.0 - engagement) * 0.25
        + regression_trend * 0.15
    )
    score = max(0.0, min(1.0, score))
    if score >= 0.65:
        return "high", score
    if score >= 0.4:
        return "medium", score
    return "low", score


def find_missing_prerequisites(
    concept_key: str,
    mastery_by_concept: dict[str, float],
    dependencies: list[tuple[str, str, float]],
    *,
    threshold: float = GAP_THRESHOLD,
) -> list[str]:
    """Traverse prerequisite DAG — weak prereqs block dependent concepts."""
    missing: list[str] = []
    for prereq, dependent, _weight in dependencies:
        if dependent != concept_key:
            continue
        m = mastery_by_concept.get(prereq, 0.0)
        if m < threshold:
            missing.append(prereq)
    return missing


def teaching_pace_from_signals(
    *,
    confusion: float,
    attention: float,
    mastery_avg: float,
) -> str:
    if confusion >= 0.55 or attention < 0.4:
        return "slow"
    if mastery_avg >= 0.75 and confusion < 0.3:
        return "fast"
    return "medium"


def difficulty_recommendation(mastery_avg: float, confidence: float) -> str:
    if mastery_avg < MASTERY_WEAK or confidence < 0.4:
        return "below_level"
    if mastery_avg >= MASTERY_STRONG and confidence >= 0.7:
        return "above_level"
    return "on_level"


def _self_check() -> None:
    assert 0.0 <= update_mastery_ema(0.5, "correct") <= 1.0
    assert update_mastery_ema(0.5, "incorrect") < 0.5
    assert bloom_to_understanding_level(BloomLevel.APPLY) == UNDERSTANDING_APPLY
    assert dominant_learning_style({STYLE_VISUAL: 0.5, STYLE_READING: 0.1}) == STYLE_VISUAL
    level, _ = compute_risk(mastery_avg=0.2, confidence=0.3, engagement=0.2, regression_trend=0.5)
    assert level == "high"
    print("lia.algorithms.mastery self-check ok")


if __name__ == "__main__":
    _self_check()
