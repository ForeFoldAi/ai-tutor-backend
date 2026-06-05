"""
Persistent learner profile for voice tutoring (Redis-backed, fail-open).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

from app.services.voice_tutor import LearnerProfileSnapshot, UnderstandingScores

logger = logging.getLogger(__name__)

_PROFILE_PREFIX = "learner:voice:"
_MAX_TOPICS = 20
_MAX_SCORES = 30


@dataclass
class LearnerProfile:
    student_key: str
    strong_topics: list[str] = field(default_factory=list)
    weak_topics: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)
    quiz_scores: list[float] = field(default_factory=list)

    def snapshot(self) -> LearnerProfileSnapshot:
        return LearnerProfileSnapshot(
            student_key=self.student_key,
            strong_topics=list(self.strong_topics),
            weak_topics=list(self.weak_topics),
            recent_topics=list(self.recent_topics),
            quiz_scores=list(self.quiz_scores),
        )

    def record_topic(self, topic: str) -> None:
        t = (topic or "").strip()[:120]
        if not t:
            return
        self.recent_topics = [t] + [x for x in self.recent_topics if x != t]
        self.recent_topics = self.recent_topics[:_MAX_TOPICS]

    def apply_understanding(self, topic: str, scores: UnderstandingScores) -> None:
        self.record_topic(topic)
        if scores.confusion >= 0.55:
            if topic and topic not in self.weak_topics:
                self.weak_topics.append(topic)
            self.weak_topics = self.weak_topics[-_MAX_TOPICS:]
        elif scores.understanding >= 0.7 and topic:
            if topic not in self.strong_topics:
                self.strong_topics.append(topic)
            self.strong_topics = self.strong_topics[-_MAX_TOPICS:]
            self.weak_topics = [w for w in self.weak_topics if w != topic]
        if scores.wants_quiz and scores.understanding >= 0.6:
            self.quiz_scores.append(scores.understanding)
            self.quiz_scores = self.quiz_scores[-_MAX_SCORES:]


async def load_learner_profile(student_key: str) -> LearnerProfile:
    if not student_key:
        return LearnerProfile(student_key="")
    try:
        from app.core.cache import get_redis

        raw = await get_redis().get(f"{_PROFILE_PREFIX}{student_key}")
        if raw:
            data = json.loads(raw)
            return LearnerProfile(
                student_key=student_key,
                strong_topics=list(data.get("strong_topics") or []),
                weak_topics=list(data.get("weak_topics") or []),
                recent_topics=list(data.get("recent_topics") or []),
                quiz_scores=[float(x) for x in (data.get("quiz_scores") or [])],
            )
    except Exception as exc:
        logger.debug("Learner profile load miss: %s", exc)
    return LearnerProfile(student_key=student_key)


async def save_learner_profile(profile: LearnerProfile) -> None:
    if not profile.student_key:
        return
    try:
        from app.core.cache import get_redis

        payload = json.dumps(asdict(profile))
        await get_redis().set(f"{_PROFILE_PREFIX}{profile.student_key}", payload, ex=86400 * 30)
    except Exception as exc:
        logger.debug("Learner profile save skipped: %s", exc)
