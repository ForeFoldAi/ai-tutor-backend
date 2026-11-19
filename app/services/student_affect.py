"""
Hybrid student affect / sentiment classification for voice tutoring.

Layer 1: regex (< 1ms)
Layer 2: BGE prototypes (only when regex inconclusive)
Layer 3: LLM JSON (only when confidence < threshold)
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_AFFECT_BGE_MIN_SCORE = 38.0
_AFFECT_BGE_MARGIN = 2.5
_AFFECT_LLM_CONFIDENCE_THRESHOLD = 0.6
_AFFECT_LLM_TIMEOUT_SEC = 6.0

_CLOSING_RE = re.compile(
    r"\b(thank\s*you|thanks|thank\s*u|okay?\s*,?\s*got\s+it|i\s+understand\s+now|"
    r"that'?s\s+all|i'?m\s+done|no\s+more\s+questions?|bye|goodbye|see\s+you)\b",
    re.I,
)
_CONFUSION_RE = re.compile(
    r"\b(confused|don'?t\s+understand|not\s+clear|what\s+do\s+you\s+mean|huh|"
    r"no\s+idea|no\s+clue|(?:don'?t|dont)\s+know|not\s+sure|"
    r"i\s+don'?t\s+get|still\s+confused|too\s+hard|lost)\b",
    re.I,
)
_FRUSTRATION_RE = re.compile(
    r"\b(frustrat|annoyed|fed\s+up|give\s+up|stupid|hate\s+this|"
    r"this\s+sucks|can'?t\s+do\s+this|so\s+hard|impossible)\b",
    re.I,
)
_BOREDOM_RE = re.compile(
    r"\b(bor(?:ed|ing)|tired\s+of|something\s+else|don'?t\s+care|"
    r"whatever|skip\s+this|move\s+on)\b",
    re.I,
)
_EXCITEMENT_RE = re.compile(
    r"\b(wow|amazing|awesome|cool|love\s+that|so\s+cool|excited|"
    r"that'?s\s+great|brilliant|fantastic)\b",
    re.I,
)
_AFFIRM_RE = re.compile(
    r"^(yes|yeah|yep|yup|ok|okay|sure|right|correct|exactly|got\s+it|"
    r"i\s+understand|understood|makes\s+sense)[.!?]*$",
    re.I,
)
_PERSONAL_RE = re.compile(
    r"\b(i\s+was|it\s+was|when\s+i|where\s+i|one\s+day|yesterday|last\s+(?:week|month)|"
    r"near\s+my|at\s+my\s+house|my\s+home)\b",
    re.I,
)
_CURIOUS_RE = re.compile(
    r"\b(how\s+does|why\s+does|tell\s+me\s+more|curious|wonder|"
    r"what\s+if|can\s+you\s+explain|what\s+is|who\s+is|where\s+is)\b",
    re.I,
)
_QUESTION_START_RE = re.compile(
    r"^(what|who|where|when|why|how|can|could|would|is|are|do|does)\b",
    re.I,
)
_SARCASM_RE = re.compile(r"\b(sure\s+whatever|yeah\s+right|if\s+you\s+say\s+so)\b", re.I)

_AFFECT_PROTOTYPES: dict[str, list[str]] = {
    "confused": [
        "I don't understand",
        "I'm confused",
        "I don't get it",
        "this is too hard",
        "I'm lost",
    ],
    "frustrated": [
        "I'm so frustrated",
        "this is annoying",
        "I'm fed up",
        "I give up",
    ],
    "bored": [
        "this is boring",
        "I'm bored",
        "can we do something else",
        "whatever",
    ],
    "excited": [
        "wow that's amazing",
        "that's so cool",
        "I love this",
        "this is awesome",
    ],
    "closing": [
        "thank you bye",
        "I'm done for today",
        "that's all thanks",
    ],
    "affirmation": [
        "got it",
        "makes sense",
        "I understand now",
        "okay I see",
    ],
}

_AFFECT_LLM_SYSTEM_PROMPT = (
    "You classify a school student's emotional tone during a live voice tutoring session. "
    "Read the tutor's last message and the student's reply. Reply with ONLY compact JSON — "
    "no prose, no markdown — exactly:\n"
    '{"primary":"confused|frustrated|bored|excited|curious|proud|affirmation|personal|closing|neutral",'
    '"valence":-1.0 to 1.0,"engagement":0.0 to 1.0,"confusion":0.0 to 1.0,'
    '"frustration":0.0 to 1.0,"excitement":0.0 to 1.0,"confidence":0.0 to 1.0}\n'
    "primary: dominant affect. valence: negative to positive mood. "
    "engagement: how interested they seem. confidence: how sure you are."
)


@dataclass
class StudentAffect:
    primary: str = "neutral"
    valence: float = 0.0
    engagement: float = 0.5
    confusion: float = 0.0
    frustration: float = 0.0
    excitement: float = 0.0
    confidence: float = 0.5
    source: str = "regex"
    is_affirmation: bool = False
    wants_expansion: bool = False
    wants_quiz: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_understanding_scores(self) -> dict[str, Any]:
        """Bridge to existing UnderstandingScores consumers."""
        understanding = 0.5
        if self.is_affirmation:
            understanding = 0.85
        elif self.confusion >= 0.55:
            understanding = 0.35
        elif self.engagement >= 0.7:
            understanding = 0.7
        return {
            "understanding": understanding,
            "confidence": min(0.95, self.confidence + 0.1),
            "confusion": self.confusion,
            "is_affirmation": self.is_affirmation,
            "wants_expansion": self.wants_expansion,
            "wants_quiz": self.wants_quiz,
        }

    def to_hint(self) -> str:
        hints = {
            "confused": (
                "The student seems confused. Simplify language, use one analogy, "
                "and teach ONE small point. Validate that it's okay to be stuck."
            ),
            "frustrated": (
                "The student seems frustrated. Slow down, acknowledge their feeling briefly, "
                "then one tiny step. You may use their first name once for reassurance."
            ),
            "bored": (
                "The student seems bored or disengaged. Keep this turn short. "
                "Offer a choice or a quick challenge — no lecture."
            ),
            "excited": (
                "The student is excited. Match their energy in tone only. "
                "Do not start a new lesson unless they asked a question."
            ),
            "curious": (
                "The student is curious. If they asked a question, answer it from the chapter. "
                "If they only reacted, acknowledge — do not add new facts."
            ),
            "affirmation": (
                "The student understood. Brief warm acknowledgment only. "
                "Do not teach a new fact unless they asked for more."
            ),
            "personal": (
                "The student shared a personal example. Acknowledge their story first, "
                "then connect it to the lesson."
            ),
            "closing": (
                "The student is wrapping up. Warm brief goodbye — do not teach new content."
            ),
            "proud": (
                "Celebrate their progress briefly, then offer a gentle next step."
            ),
        }
        return hints.get(self.primary, "")

    def to_student_hint(self) -> str:
        ui = {
            "confused": "Let me explain that more simply…",
            "frustrated": "Let's slow down together…",
            "bored": "Let's make this more interesting…",
            "excited": "Love that energy — one moment…",
            "closing": "Wrapping up…",
            "affirmation": "Glad that made sense.",
        }
        return ui.get(self.primary, "")

    def summary(self) -> str:
        return f"{self.primary} (engagement={self.engagement:.1f}, valence={self.valence:+.1f})"


def _clamp01(v: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _classify_affect_by_bge(text: str) -> tuple[str | None, float]:
    from app.services.image_service.figure_context_bge import cosine_100, embed_query
    from app.services.vector_service import is_embedding_model_loaded

    if not is_embedding_model_loaded():
        return None, 0.0
    prototypes = _load_affect_prototype_embeddings()
    if not prototypes:
        return None, 0.0
    qvec = embed_query(text[:200])
    if qvec is None:
        return None, 0.0

    best_label: str | None = None
    best_score = 0.0
    second_score = 0.0
    for label, vectors in prototypes.items():
        score = max(cosine_100(qvec, pvec) for pvec in vectors)
        if score > best_score:
            best_label, second_score, best_score = label, best_score, score
        elif score > second_score:
            second_score = score

    if best_label is None or best_score < _AFFECT_BGE_MIN_SCORE:
        return None, 0.0
    if (best_score - second_score) < _AFFECT_BGE_MARGIN:
        return None, best_score / 100.0
    return best_label, min(0.85, best_score / 100.0)


_affect_prototype_cache: dict[str, list[Any]] | None = None


def _load_affect_prototype_embeddings() -> dict[str, list[Any]] | None:
    global _affect_prototype_cache
    if _affect_prototype_cache is not None:
        return _affect_prototype_cache or None
    from app.services.image_service.figure_context_bge import embed_texts

    cache: dict[str, list[Any]] = {}
    for label, phrases in _AFFECT_PROTOTYPES.items():
        vectors = [v for v in embed_texts(phrases) if v is not None]
        if vectors:
            cache[label] = vectors
    _affect_prototype_cache = cache if cache else {}
    return _affect_prototype_cache or None


def _regex_affect(text: str) -> StudentAffect | None:
    q = (text or "").strip()
    if not q:
        return None
    from app.services.voice_tutor import voice_expand_requested

    wants_expansion = voice_expand_requested(q)
    wants_quiz = bool(re.search(r"\b(quiz\s+me|test\s+me|ask\s+me\s+questions?)\b", q, re.I))
    is_affirm = bool(_AFFIRM_RE.match(q))

    if _CLOSING_RE.search(q):
        return StudentAffect(
            primary="closing",
            valence=0.4,
            engagement=0.3,
            confidence=0.9,
            source="regex",
        )
    if _CONFUSION_RE.search(q):
        return StudentAffect(
            primary="confused",
            valence=-0.3,
            engagement=0.4,
            confusion=0.8,
            confidence=0.88,
            source="regex",
            wants_expansion=wants_expansion or True,
        )
    if _FRUSTRATION_RE.search(q):
        return StudentAffect(
            primary="frustrated",
            valence=-0.6,
            engagement=0.35,
            confusion=0.5,
            frustration=0.85,
            confidence=0.88,
            source="regex",
            wants_expansion=True,
        )
    if _BOREDOM_RE.search(q):
        return StudentAffect(
            primary="bored",
            valence=-0.4,
            engagement=0.2,
            confidence=0.85,
            source="regex",
        )
    if _EXCITEMENT_RE.search(q):
        return StudentAffect(
            primary="excited",
            valence=0.7,
            engagement=0.85,
            excitement=0.85,
            confidence=0.85,
            source="regex",
        )
    if _SARCASM_RE.search(q):
        return StudentAffect(
            primary="bored",
            valence=-0.5,
            engagement=0.25,
            frustration=0.4,
            confidence=0.75,
            source="regex",
        )
    if is_affirm:
        return StudentAffect(
            primary="affirmation",
            valence=0.5,
            engagement=0.7,
            confidence=0.9,
            source="regex",
            is_affirmation=True,
        )
    if _PERSONAL_RE.search(q):
        return StudentAffect(
            primary="personal",
            valence=0.3,
            engagement=0.75,
            confidence=0.8,
            source="regex",
        )
    if _CURIOUS_RE.search(q) and len(q.split()) >= 3:
        return StudentAffect(
            primary="curious",
            valence=0.4,
            engagement=0.8,
            confidence=0.75,
            source="regex",
            wants_quiz=wants_quiz,
        )
    if q.endswith("?") or _QUESTION_START_RE.match(q):
        return StudentAffect(
            primary="curious",
            valence=0.3,
            engagement=0.7,
            confidence=0.7,
            source="regex",
            wants_quiz=wants_quiz,
        )
    return None


def evaluate_student_affect(
    student_text: str,
    *,
    last_assistant: str = "",
    tutor_state: str = "TEACHING",
) -> StudentAffect:
    """Fast synchronous affect — regex then optional BGE."""
    from app.config import VOICE_AFFECT_ENGINE

    if not VOICE_AFFECT_ENGINE:
        from app.services.voice_tutor import TutorState, evaluate_student_response

        try:
            ts = TutorState(tutor_state)
        except ValueError:
            ts = TutorState.TEACHING
        scores = evaluate_student_response(
            student_text, last_assistant=last_assistant, tutor_state=ts
        )
        return StudentAffect(
            primary="confused" if scores.confusion >= 0.55 else (
                "affirmation" if scores.is_affirmation else "neutral"
            ),
            confusion=scores.confusion,
            engagement=0.5 if scores.confusion < 0.55 else 0.35,
            confidence=0.7,
            source="legacy",
            is_affirmation=scores.is_affirmation,
            wants_expansion=scores.wants_expansion,
            wants_quiz=scores.wants_quiz,
        )

    hit = _regex_affect(student_text)
    if hit and hit.confidence >= 0.75:
        return hit

    label, bge_conf = _classify_affect_by_bge(student_text)
    if label:
        base = hit or StudentAffect()
        base.primary = label
        base.confidence = max(base.confidence, bge_conf)
        base.source = "bge" if not hit else f"{hit.source}+bge"
        if label == "confused":
            base.confusion = max(base.confusion, 0.7)
        if label == "frustrated":
            base.frustration = max(base.frustration, 0.7)
        if label == "excited":
            base.excitement = max(base.excitement, 0.7)
        if label == "affirmation":
            base.is_affirmation = True
        return base

    if hit:
        return hit

    return StudentAffect(confidence=0.4, source="neutral")


def _parse_affect_json(raw: str) -> StudentAffect | None:
    import json

    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    primary = str(data.get("primary") or "neutral").lower()
    return StudentAffect(
        primary=primary,
        valence=max(-1.0, min(1.0, float(data.get("valence", 0)))),
        engagement=_clamp01(data.get("engagement"), 0.5),
        confusion=_clamp01(data.get("confusion")),
        frustration=_clamp01(data.get("frustration")),
        excitement=_clamp01(data.get("excitement")),
        confidence=_clamp01(data.get("confidence"), 0.7),
        source="llm",
        is_affirmation=primary == "affirmation",
    )


async def classify_affect_llm(
    student_text: str,
    *,
    last_assistant: str = "",
) -> StudentAffect | None:
    from app.config import VOICE_AFFECT_LLM_FALLBACK
    from app.services.llm_client import complete

    if not VOICE_AFFECT_LLM_FALLBACK:
        return None
    user = (
        f"Tutor last said:\n{(last_assistant or '(none)')[:400]}\n\n"
        f"Student replied:\n{(student_text or '')[:400]}"
    )
    try:
        import asyncio

        raw = await asyncio.wait_for(
            complete(
                [
                    {"role": "system", "content": _AFFECT_LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_tokens=120,
                feature="voice_affect",
            ),
            timeout=_AFFECT_LLM_TIMEOUT_SEC,
        )
    except Exception as exc:
        logger.debug("Affect LLM classification failed: %s", exc)
        return None
    return _parse_affect_json(raw)


async def evaluate_student_affect_async(
    student_text: str,
    *,
    last_assistant: str = "",
    tutor_state: str = "TEACHING",
) -> StudentAffect:
    """Hybrid affect with optional LLM fallback when confidence is low."""
    base = evaluate_student_affect(
        student_text, last_assistant=last_assistant, tutor_state=tutor_state
    )
    if base.confidence >= _AFFECT_LLM_CONFIDENCE_THRESHOLD:
        return base
    llm = await classify_affect_llm(student_text, last_assistant=last_assistant)
    if llm:
        llm.wants_expansion = base.wants_expansion or llm.wants_expansion
        llm.wants_quiz = base.wants_quiz or llm.wants_quiz
        llm.is_affirmation = base.is_affirmation or llm.is_affirmation
        return llm
    return base


def merge_affect_trajectory(
    trajectory: list[str] | None,
    primary: str,
    *,
    limit: int = 5,
) -> list[str]:
    out = list(trajectory or [])
    if primary and (not out or out[-1] != primary):
        out.append(primary)
    return out[-limit:]


def rapport_from_trajectory(trajectory: list[str]) -> str:
    if not trajectory:
        return ""
    if trajectory.count("confused") + trajectory.count("frustrated") >= 2:
        if trajectory[-1] in ("neutral", "affirmation", "excited", "curious"):
            return (
                "RAPPORT CONTEXT (do not mention explicitly): "
                "The student struggled earlier but seems more engaged now — "
                "acknowledge progress briefly if natural."
            )
    if trajectory[-1:] == ["bored"]:
        return (
            "RAPPORT CONTEXT (do not mention explicitly): "
            "The student seemed bored — keep this turn lively and short."
        )
    return ""
