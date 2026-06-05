"""
Conversation context resolution and visual-retrieval gating for the AI Tutor.

Resolves short follow-ups (yes, quiz me, simplify) using session history while
keeping image retrieval scoped to the *current* turn intent (no image inheritance).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


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


class FollowupType(str, Enum):
    NONE = "none"
    CONTINUE_EXPLANATION = "continue_explanation"
    SIMPLIFY = "simplify"
    GENERATE_QUESTIONS = "generate_questions"
    GENERATE_MCQ = "generate_mcq"
    GREETING = "greeting"
    SMALL_TALK = "small_talk"
    ASK_EXAMPLE = "ask_example"
    ASK_DIAGRAM = "ask_diagram"
    ASK_VISUAL = "ask_visual"
    ASK_SUMMARY = "ask_summary"
    ASK_COMPARISON = "ask_comparison"
    NEW_TOPIC = "new_topic"


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

    def to_debug(self) -> dict:
        return {
            "resolved_topic": self.resolved_topic[:120],
            "current_intent": self.current_intent,
            "followup_type": self.followup_type,
            "response_mode": self.response_mode.value,
            "visual_intent": self.visual_intent.value,
            "requires_visuals": self.requires_visuals,
            "inherited_entities": self.inherited_entities[:8],
        }


_GREETING_RE = re.compile(
    r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|namaste|howdy)\b",
    re.I,
)
_SMALL_TALK_RE = re.compile(
    r"^(how are you|what'?s up|thanks|thank you|bye|goodbye|see you)\b",
    re.I,
)
_THANKS_RE = re.compile(r"^(thanks|thank you|thx)\b", re.I)
_QUIZ_RE = re.compile(
    r"\b("
    r"give\s+(me\s+)?\d+\s+questions?|"
    r"\d+\s+questions?|"
    r"quiz\s+me|"
    r"practice\s+questions?|"
    r"homework\s+questions?|"
    r"test\s+me|"
    r"mcq|multiple\s+choice|"
    r"objective\s+questions?"
    r")\b",
    re.I,
)
_MCQ_RE = re.compile(r"\b(mcq|multiple\s+choice|objective\s+type)\b", re.I)
_SUMMARY_RE = re.compile(r"\b(summarize|summary|in\s+short|briefly)\b", re.I)
_SIMPLIFY_RE = re.compile(r"\b(simplify|simpler|easier|in\s+simple\s+words|eli5)\b", re.I)
_CONTINUE_RE = re.compile(
    r"^(yes|yeah|yep|ok|okay|sure|continue|go\s+on|explain\s+more|tell\s+me\s+more|"
    r"more|next|carry\s+on)\s*[.!?]*$",
    re.I,
)
_EXAMPLE_RE = re.compile(r"\b(example|for\s+instance|show\s+me\s+an\s+example)\b", re.I)
_DIAGRAM_RE = re.compile(
    r"\b(diagram|figure|illustration|picture|draw|sketch|flowchart|chart)\b",
    re.I,
)
_VISUAL_RE = re.compile(
    r"\b(show\s+(me\s+)?(the\s+)?(image|picture|diagram|figure|map)|"
    r"with\s+(a\s+)?(diagram|image|picture)|visual|see\s+the\s+figure)\b",
    re.I,
)
_COMPARISON_RE = re.compile(r"\b(compare|difference\s+between|vs\.?|versus)\b", re.I)
_CONCEPTUAL_RE = re.compile(
    r"\b(explain|describe|what\s+is|what\s+are|how\s+does|why\s+does|define|"
    r"tell\s+me\s+about|list|types?\s+of|name\s+the|"
    r"process|formation|structure|location|region|climate|desert|river|mountain|map|"
    r"instrument|instruments|weather|temperature|precipitation|humidity|wind|pressure)\b",
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


def _extract_entities_from_text(text: str) -> list[str]:
    from app.services.image_service.symbolic_image_filters import extract_educational_entities

    if not text.strip():
        return []
    core = re.sub(r"\s+", " ", text.lower()).strip()[:80]
    return extract_educational_entities(text, core)[:12]


def _classify_followup(query: str) -> FollowupType:
    q = query.strip()
    if not q:
        return FollowupType.NONE
    if _GREETING_RE.match(q):
        return FollowupType.GREETING
    if _SMALL_TALK_RE.match(q) or _THANKS_RE.match(q):
        return FollowupType.SMALL_TALK
    if _MCQ_RE.search(q):
        return FollowupType.GENERATE_MCQ
    if _QUIZ_RE.search(q):
        return FollowupType.GENERATE_QUESTIONS
    if _SIMPLIFY_RE.search(q):
        return FollowupType.SIMPLIFY
    if _SUMMARY_RE.search(q):
        return FollowupType.ASK_SUMMARY
    if _VISUAL_RE.search(q) or _DIAGRAM_RE.search(q):
        return FollowupType.ASK_VISUAL if _VISUAL_RE.search(q) else FollowupType.ASK_DIAGRAM
    if _EXAMPLE_RE.search(q):
        return FollowupType.ASK_EXAMPLE
    if _COMPARISON_RE.search(q):
        return FollowupType.ASK_COMPARISON
    if _CONTINUE_RE.match(q):
        return FollowupType.CONTINUE_EXPLANATION
    if len(q.split()) <= 3 and not _CONCEPTUAL_RE.search(q):
        return FollowupType.CONTINUE_EXPLANATION
    return FollowupType.NEW_TOPIC


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
        FollowupType.SIMPLIFY,
    ):
        return VisualIntent.NO_VISUALS
    if _CONCEPTUAL_RE.search(query):
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
        followup = _classify_followup(q)
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
            FollowupType.SIMPLIFY,
            FollowupType.ASK_EXAMPLE,
            FollowupType.ASK_SUMMARY,
            FollowupType.ASK_COMPARISON,
            FollowupType.GENERATE_QUESTIONS,
            FollowupType.GENERATE_MCQ,
        ) and prior_user:
            resolved_topic = prior_user
            inherited_entities = _extract_entities_from_text(prior_user)
            if followup == FollowupType.SIMPLIFY:
                resolved_topic = f"{prior_user} (explain in simpler words)"
            elif followup == FollowupType.GENERATE_QUESTIONS:
                resolved_topic = prior_user
            elif followup == FollowupType.GENERATE_MCQ:
                resolved_topic = prior_user
        elif followup == FollowupType.NEW_TOPIC or _CONCEPTUAL_RE.search(q):
            resolved_topic = q
            inherited_entities = _extract_entities_from_text(q)
        elif prior_user and len(q.split()) <= 4:
            resolved_topic = prior_user
            inherited_entities = _extract_entities_from_text(prior_user)

        retrieval_query = resolved_topic if resolved_topic else q
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
) -> bool:
    """
    Master gate: images only when chapter scope exists and pedagogy allows visuals.
    """
    from app.config import ENABLE_VISUAL_INTENT_DETECTION

    if not chapter_ids:
        return False
    # Main-section chapter questions (e.g. "what are weather instruments") always allow figures.
    if heading_scope_kind == "main_section":
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
