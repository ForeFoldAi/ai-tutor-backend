"""
Conversation context resolution and visual-retrieval gating for the AI Tutor.

Resolves short follow-ups (yes, quiz me, simplify) using hybrid regex + BGE intent
classification while keeping image retrieval scoped to the *current* turn intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.services.conversation_intent_classifier import (
    FollowupType,
    answer_type_for_followup,
    classify_followup_intent,
    is_clarification_followup,
)

__all__ = [
    "FollowupType",
    "VisualIntent",
    "ResponseMode",
    "ConversationContext",
    "ConversationContextResolver",
    "resolve_conversation_context",
    "should_retrieve_images",
    "is_clarification_followup",
    "answer_type_for_followup",
]


class VisualIntent(str, Enum):
    NO_VISUALS = "no_visuals"
    OPTIONAL_VISUALS = "optional_visuals"
    REQUIRED_VISUALS = "required_visuals"


class ResponseMode(str, Enum):
    EXPLANATION = "explanation"
    QUIZ = "quiz"
    MCQ = "mcq"
    GREETING = "greeting"
    FOLLOWUP = "followup"
    SUMMARY = "summary"
    VISUAL_LEARNING = "visual_learning"
    SMALL_TALK = "small_talk"


@dataclass
class ConversationContext:
    resolved_topic: str
    current_intent: str
    inherited_entities: list[str] = field(default_factory=list)
    inherited_chapter: str | None = None
    followup_type: str = FollowupType.NONE.value
    requires_visuals: bool = False
    response_mode: ResponseMode = ResponseMode.EXPLANATION
    visual_intent: VisualIntent = VisualIntent.OPTIONAL_VISUALS
    retrieval_query: str = ""
    intent_method: str = "regex"
    intent_confidence: float = 0.0

    def to_debug(self) -> dict:
        return {
            "resolved_topic": self.resolved_topic[:120],
            "current_intent": self.current_intent,
            "followup_type": self.followup_type,
            "response_mode": self.response_mode.value,
            "visual_intent": self.visual_intent.value,
            "requires_visuals": self.requires_visuals,
            "inherited_entities": self.inherited_entities[:8],
            "intent_method": self.intent_method,
            "intent_confidence": self.intent_confidence,
        }


_CONCEPTUAL_RE = re.compile(
    r"\b(explain|describe|what\s+is|what\s+are|how\s+does|why\s+does|define|"
    r"tell\s+me\s+about|list|types?\s+of|name\s+the|"
    r"process|formation|structure|location|region|climate|desert|river|mountain|map|"
    r"instrument|instruments|weather|temperature|precipitation|humidity|wind|pressure)\b",
    re.I,
)
_VISUAL_RE = re.compile(
    r"\b(show\s+(me\s+)?(the\s+)?(image|picture|diagram|figure|map)|"
    r"with\s+(a\s+)?(diagram|image|picture)|visual|see\s+the\s+figure)\b",
    re.I,
)
_DIAGRAM_RE = re.compile(
    r"\b(diagram|figure|illustration|picture|draw|sketch|flowchart|chart)\b",
    re.I,
)
_MATH_SUBJECT_RE = re.compile(
    r"\b(math|mathematics|algebra|geometry|arithmetic|trigonometry|calculus)\b",
    re.I,
)
_MATH_TEACHING_RE = re.compile(
    r"\b("
    r"solve|find|calculate|factori[sz]e|prove|simplify|evaluate|"
    r"area|volume|circumference|probability|mean|median|mode|"
    r"circle|triangle|equation|graph|polynomial|remainder|tangent|"
    r"parallel|transversal|construction|mensuration|statistics"
    r")\b",
    re.I,
)
_PRONOUN_FOLLOWUP = re.compile(
    r"\b(that|it|this|they|them|those|these|same\s+thing)\b",
    re.I,
)
_WHY_SHORT = re.compile(r"^why\b", re.I)
_SHORT_FILLER_RE = re.compile(
    r"^(yes|no|ok|okay|sure|thanks|thank you|hi|hello|hey)\s*[.!?]*$",
    re.I,
)


def _last_user_message(history: list[dict]) -> str:
    for turn in reversed(history):
        if (turn.get("role") or "").lower() == "user":
            content = (turn.get("content") or "").strip()
            if content and not _SHORT_FILLER_RE.match(content):
                return content
    return ""


def _last_assistant_snippet(history: list[dict], max_len: int = 200) -> str:
    for turn in reversed(history):
        if (turn.get("role") or "").lower() == "assistant":
            content = (turn.get("content") or "").strip()
            if content:
                return content[:max_len]
    return ""


def _teaching_assistant_snippet(history: list[dict], max_len: int = 400) -> str:
    """
    Last assistant turn with substantive teaching (skip short follow-up questions).

    When the student says they are confused, they usually refer to the lesson —
    not the tutor's closing check question from the previous turn.
    """
    candidates: list[str] = []
    for turn in reversed(history):
        if (turn.get("role") or "").lower() != "assistant":
            continue
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        candidates.append(content)
        words = content.split()
        is_short_question = (
            len(words) <= 18
            and "?" in content
            and not re.search(r"\b(step|activity|challenge|experiment|place|observe)\b", content, re.I)
        )
        if not is_short_question:
            return content[:max_len]
    return candidates[0][:max_len] if candidates else ""


def _extract_entities_from_text(text: str) -> list[str]:
    from app.services.image_service.symbolic_image_filters import extract_educational_entities

    if not text.strip():
        return []
    core = re.sub(r"\s+", " ", text.lower()).strip()[:80]
    return extract_educational_entities(text, core)[:12]


def _response_mode_for(followup: FollowupType, query: str) -> ResponseMode:
    if followup == FollowupType.GREETING:
        return ResponseMode.GREETING
    if followup == FollowupType.SMALL_TALK:
        return ResponseMode.SMALL_TALK
    if followup == FollowupType.GENERATE_MCQ:
        return ResponseMode.MCQ
    if followup == FollowupType.GENERATE_QUESTIONS:
        return ResponseMode.QUIZ
    if followup in (FollowupType.ASK_VISUAL, FollowupType.ASK_DIAGRAM):
        return ResponseMode.VISUAL_LEARNING
    if followup == FollowupType.ASK_SUMMARY:
        return ResponseMode.SUMMARY
    if followup in (
        FollowupType.CONTINUE_EXPLANATION,
        FollowupType.CLARIFICATION,
        FollowupType.SIMPLIFY,
        FollowupType.ASK_EXAMPLE,
        FollowupType.ASK_COMPARISON,
    ):
        return ResponseMode.FOLLOWUP
    if _CONCEPTUAL_RE.search(query):
        return ResponseMode.EXPLANATION
    return ResponseMode.EXPLANATION


def _visual_intent_for(mode: ResponseMode, followup: FollowupType, query: str) -> VisualIntent:
    if mode in (ResponseMode.GREETING, ResponseMode.SMALL_TALK, ResponseMode.QUIZ, ResponseMode.MCQ):
        return VisualIntent.NO_VISUALS
    if mode == ResponseMode.VISUAL_LEARNING:
        return VisualIntent.REQUIRED_VISUALS
    if followup in (FollowupType.GENERATE_QUESTIONS, FollowupType.GENERATE_MCQ):
        return VisualIntent.NO_VISUALS
    if _VISUAL_RE.search(query) or _DIAGRAM_RE.search(query):
        return VisualIntent.REQUIRED_VISUALS
    if mode == ResponseMode.SUMMARY:
        return VisualIntent.OPTIONAL_VISUALS
    if mode == ResponseMode.FOLLOWUP and followup in (
        FollowupType.CONTINUE_EXPLANATION,
        FollowupType.CLARIFICATION,
        FollowupType.SIMPLIFY,
    ):
        return VisualIntent.NO_VISUALS
    if _CONCEPTUAL_RE.search(query):
        return VisualIntent.OPTIONAL_VISUALS
    if _MATH_TEACHING_RE.search(query):
        return VisualIntent.OPTIONAL_VISUALS
    return VisualIntent.NO_VISUALS


class ConversationContextResolver:
    """Resolve conversational follow-ups into retrieval-ready context."""

    def resolve(
        self,
        query: str,
        *,
        conversation_history: list[dict] | None = None,
        chapter: str | None = None,
    ) -> ConversationContext:
        history = conversation_history or []
        q = (query or "").strip()

        classification = classify_followup_intent(q, history)
        followup = classification.followup_type
        mode = _response_mode_for(followup, q)
        visual = _visual_intent_for(mode, followup, q)

        prior_user = _last_user_message(history)
        inherited_entities: list[str] = []
        resolved_topic = q

        if (
            followup == FollowupType.NEW_TOPIC
            and prior_user
            and len(q.split()) <= 8
            and (_WHY_SHORT.match(q) or _PRONOUN_FOLLOWUP.search(q))
        ):
            followup = FollowupType.CONTINUE_EXPLANATION
            mode = _response_mode_for(followup, q)

        if followup in (
            FollowupType.CONTINUE_EXPLANATION,
            FollowupType.CLARIFICATION,
            FollowupType.SIMPLIFY,
            FollowupType.ASK_EXAMPLE,
            FollowupType.ASK_SUMMARY,
            FollowupType.ASK_COMPARISON,
            FollowupType.GENERATE_QUESTIONS,
            FollowupType.GENERATE_MCQ,
        ) and prior_user:
            last_asst = _teaching_assistant_snippet(history, max_len=400)
            inherited_entities = _extract_entities_from_text(prior_user)
            if followup == FollowupType.CLARIFICATION and last_asst:
                resolved_topic = last_asst[:200]
                inherited_entities = _extract_entities_from_text(
                    f"{prior_user} {last_asst[:200]}"
                )
            elif followup == FollowupType.SIMPLIFY:
                resolved_topic = f"{prior_user} (explain in simpler words)"
            elif followup in (FollowupType.ASK_EXAMPLE, FollowupType.ASK_COMPARISON):
                resolved_topic = f"{prior_user} {q}"
            else:
                resolved_topic = prior_user
        elif followup == FollowupType.NEW_TOPIC or _CONCEPTUAL_RE.search(q):
            resolved_topic = q
            inherited_entities = _extract_entities_from_text(q)
        elif prior_user and len(q.split()) <= 4:
            resolved_topic = prior_user
            inherited_entities = _extract_entities_from_text(prior_user)

        retrieval_query = resolved_topic if resolved_topic else q
        if followup == FollowupType.CLARIFICATION and history:
            last_asst = _teaching_assistant_snippet(history, max_len=400)
            if last_asst:
                parts = [p for p in (prior_user, last_asst[:280]) if p]
                retrieval_query = " ".join(parts)
        elif followup == FollowupType.ASK_EXAMPLE and prior_user:
            retrieval_query = f"{prior_user} real life example"
        elif followup == FollowupType.ASK_COMPARISON and prior_user:
            retrieval_query = f"{prior_user} {q}"

        requires_visuals = visual in (VisualIntent.REQUIRED_VISUALS, VisualIntent.OPTIONAL_VISUALS)

        return ConversationContext(
            resolved_topic=resolved_topic or q,
            current_intent=mode.value,
            inherited_entities=inherited_entities,
            inherited_chapter=chapter,
            followup_type=followup.value,
            requires_visuals=requires_visuals and visual != VisualIntent.NO_VISUALS,
            response_mode=mode,
            visual_intent=visual,
            retrieval_query=retrieval_query or q,
            intent_method=classification.method,
            intent_confidence=classification.confidence,
        )


_default_resolver = ConversationContextResolver()


def resolve_conversation_context(
    query: str,
    *,
    conversation_history: list[dict] | None = None,
    chapter: str | None = None,
) -> ConversationContext:
    return _default_resolver.resolve(
        query,
        conversation_history=conversation_history,
        chapter=chapter,
    )


def should_retrieve_images(
    ctx: ConversationContext,
    *,
    chapter_ids: list[str] | None = None,
    heading_scope_kind: str | None = None,
    subject_name: str | None = None,
    voice_mode: bool = False,
) -> bool:
    """
    Master gate: images only when chapter scope exists and pedagogy allows visuals.
    Voice mode relaxes gating so chapter diagrams appear during spoken lessons.
    """
    from app.config import ENABLE_VISUAL_INTENT_DETECTION

    if not chapter_ids:
        return False
    if heading_scope_kind == "main_section":
        return True
    if voice_mode:
        if ctx.response_mode in (
            ResponseMode.QUIZ,
            ResponseMode.MCQ,
            ResponseMode.GREETING,
            ResponseMode.SMALL_TALK,
        ):
            return False
        if ctx.followup_type in (
            FollowupType.GENERATE_QUESTIONS.value,
            FollowupType.GENERATE_MCQ.value,
            FollowupType.GREETING.value,
            FollowupType.SMALL_TALK.value,
        ):
            return False
        return True
    if subject_name and _MATH_SUBJECT_RE.search(subject_name):
        if ctx.response_mode not in (
            ResponseMode.QUIZ,
            ResponseMode.MCQ,
            ResponseMode.GREETING,
            ResponseMode.SMALL_TALK,
        ):
            if ctx.followup_type not in (
                FollowupType.GENERATE_QUESTIONS.value,
                FollowupType.GENERATE_MCQ.value,
                FollowupType.GREETING.value,
                FollowupType.SMALL_TALK.value,
            ):
                return True
    if not ENABLE_VISUAL_INTENT_DETECTION:
        return True
    if ctx.visual_intent == VisualIntent.NO_VISUALS:
        return False
    if ctx.response_mode in (ResponseMode.QUIZ, ResponseMode.MCQ, ResponseMode.GREETING, ResponseMode.SMALL_TALK):
        return False
    if ctx.followup_type in (
        FollowupType.GENERATE_QUESTIONS.value,
        FollowupType.GENERATE_MCQ.value,
        FollowupType.GREETING.value,
        FollowupType.SMALL_TALK.value,
    ):
        return False
    return ctx.visual_intent in (VisualIntent.OPTIONAL_VISUALS, VisualIntent.REQUIRED_VISUALS)
