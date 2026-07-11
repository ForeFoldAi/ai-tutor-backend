"""
Hybrid conversation-intent classifier for the AI Tutor.

Production pipeline:
  1. Fast regex rules for high-confidence dialogue acts (latency ~0 ms)
  2. BGE semantic prototype ranking (BAAI/bge-base-en-v1.5) for ambiguous
     short follow-ups — same embedding stack as textbook RAG
  3. Regex fallback when embeddings are unavailable

Used by ``conversation_context.ConversationContextResolver`` before RAG retrieval.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FollowupType(str, Enum):
    NONE = "none"
    CONTINUE_EXPLANATION = "continue_explanation"
    CLARIFICATION = "clarification"
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

# cosine * 100 — aligned with structured_asset_tagger._MIN_BGE_SIMILARITY
_MIN_BGE_SCORE = 38.0
_MIN_BGE_MARGIN = 2.5

# Regex patterns (mirrored / extended from conversation_context for single source)
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
    r"objective\s+questions?"
    r")\b",
    re.I,
)
_MCQ_RE = re.compile(r"\b(mcq|multiple\s+choice|objective\s+type)\b", re.I)
_SUMMARY_RE = re.compile(
    r"\b(summarize|summary|in\s+short|briefly|tl;?dr|key\s+points?\s+only)\b",
    re.I,
)
_SIMPLIFY_RE = re.compile(
    r"\b(simplify|simpler|easier|in\s+simple\s+words|eli5|like\s+i'?m\s+\d+|"
    r"explain\s+like\s+i'?m)\b",
    re.I,
)
_CONTINUE_RE = re.compile(
    r"^(yes|yeah|yep|yup|ok|okay|sure|continue|go\s+on|explain\s+more|tell\s+me\s+more|"
    r"more|next|carry\s+on|go\s+deeper|more\s+detail|elaborate)\s*[.!?]*$",
    re.I,
)
_DEEPER_RE = re.compile(
    r"\b(go\s+deeper|more\s+detail|in\s+more\s+detail|elaborate|expand\s+on\s+that|"
    r"dive\s+deeper|explain\s+further)\b",
    re.I,
)
_EXAMPLE_RE = re.compile(
    r"\b(example|for\s+instance|show\s+me\s+an\s+example|real[\s-]?life\s+example|"
    r"give\s+an\s+example|practical\s+example)\b",
    re.I,
)
_DIAGRAM_RE = re.compile(
    r"\b(diagram|figure|illustration|picture|draw|sketch|flowchart|chart)\b",
    re.I,
)
_VISUAL_RE = re.compile(
    r"\b(show\s+(me\s+)?(the\s+)?(image|picture|diagram|figure|map)|"
    r"with\s+(a\s+)?(diagram|image|picture)|visual|see\s+the\s+figure)\b",
    re.I,
)
_COMPARISON_RE = re.compile(r"\b(compare|difference\s+between|vs\.?|versus|contrast)\b", re.I)
_CONCEPTUAL_RE = re.compile(
    r"\b(explain|describe|what\s+is|what\s+are|how\s+does|why\s+does|define|"
    r"tell\s+me\s+about|list|types?\s+of|name\s+the|"
    r"challenge|activity|experiment|hands[\s-]?on|implement)\b",
    re.I,
)
_CONFUSION_RE = re.compile(
    r"\b("
    r"don'?t\s+understand|do\s+not\s+understand|didn'?t\s+understand|did\s+not\s+understand|"
    r"i\s+don'?t\s+get|don'?t\s+get\s+the|not\s+clear|confused|what\s+do\s+you\s+mean|"
    r"no\s+idea|still\s+confused|too\s+hard|explain\s+again|say\s+that\s+again|"
    r"can\s+you\s+explain\s+(?:that|this|it)\s+again|lost\s+me|didn'?t\s+follow"
    r")\b",
    re.I,
)
_CHALLENGE_RE = re.compile(
    r"\b(challenge|hands[\s-]?on|activity|experiment|project|implementation|"
    r"practical\s+task|try\s+at\s+home|do\s+myself)\b",
    re.I,
)

# BGE prototype utterances per dialogue act (student voice)
_INTENT_PROTOTYPES: dict[FollowupType, list[str]] = {
    FollowupType.CLARIFICATION: [
        "I don't understand what you just said",
        "can you explain that again more simply",
        "I'm confused about this explanation",
        "that was not clear to me",
        "what do you mean by that",
        "I didn't follow your answer",
        "this is too hard please explain again",
        "I did not understand this challenge",
    ],
    FollowupType.CONTINUE_EXPLANATION: [
        "yes continue please",
        "go on tell me more",
        "explain more about this topic",
        "okay next part",
        "carry on with the lesson",
        "go deeper on that topic",
        "elaborate more please",
        "why does that happen",
        "what about that one",
    ],
    FollowupType.SIMPLIFY: [
        "explain in simpler words",
        "make it easier for me",
        "simplify this please",
        "explain like I'm ten years old",
        "use simpler language",
    ],
    FollowupType.GENERATE_QUESTIONS: [
        "give me five practice questions",
        "quiz me on this topic",
        "test me with questions",
        "I want homework questions",
    ],
    FollowupType.GENERATE_MCQ: [
        "give me multiple choice questions",
        "mcq practice questions",
        "objective type questions please",
    ],
    FollowupType.ASK_SUMMARY: [
        "summarize this in short",
        "give me a brief summary",
        "key points only please",
    ],
    FollowupType.ASK_EXAMPLE: [
        "show me a real life example",
        "give an example from daily life",
        "can you give a practical example",
    ],
    FollowupType.ASK_VISUAL: [
        "show me the picture from the textbook",
        "I want to see the figure",
        "display the diagram please",
    ],
    FollowupType.ASK_DIAGRAM: [
        "draw a diagram of this",
        "show the illustration",
        "I need a flowchart",
    ],
    FollowupType.ASK_COMPARISON: [
        "compare weather and climate",
        "what is the difference between these two",
        "contrast solids and liquids",
    ],
    FollowupType.GREETING: [
        "hello good morning",
        "hi there",
        "hey tutor",
    ],
    FollowupType.SMALL_TALK: [
        "thank you very much",
        "how are you doing",
        "goodbye see you later",
    ],
    FollowupType.NEW_TOPIC: [
        "what is photosynthesis",
        "explain the water cycle in detail",
        "how does the heart pump blood",
        "give a challenge I can implement from this chapter",
        "what are the types of rocks",
        "describe the structure of an atom",
    ],
}

# Intents where regex match is authoritative — skip BGE override
_HIGH_CONFIDENCE_INTENTS = frozenset({
    FollowupType.CLARIFICATION,
    FollowupType.GREETING,
    FollowupType.SMALL_TALK,
    FollowupType.GENERATE_MCQ,
    FollowupType.GENERATE_QUESTIONS,
    FollowupType.SIMPLIFY,
    FollowupType.ASK_SUMMARY,
    FollowupType.ASK_VISUAL,
    FollowupType.ASK_DIAGRAM,
    FollowupType.ASK_EXAMPLE,
    FollowupType.ASK_COMPARISON,
    FollowupType.CONTINUE_EXPLANATION,
})

_prototype_cache: dict[str, Any] | None = None


@dataclass(frozen=True)
class IntentClassification:
    followup_type: FollowupType
    confidence: float
    method: str  # regex | bge | hybrid


def classify_followup_regex(query: str) -> FollowupType:
    """Deterministic regex classifier (fast path)."""
    q = (query or "").strip()
    if not q:
        return FollowupType.NONE
    if _CONFUSION_RE.search(q):
        return FollowupType.CLARIFICATION
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
    if _VISUAL_RE.search(q):
        return FollowupType.ASK_VISUAL
    if _DIAGRAM_RE.search(q):
        return FollowupType.ASK_DIAGRAM
    if _EXAMPLE_RE.search(q):
        return FollowupType.ASK_EXAMPLE
    if _COMPARISON_RE.search(q):
        return FollowupType.ASK_COMPARISON
    if _CONTINUE_RE.match(q) or _DEEPER_RE.search(q):
        return FollowupType.CONTINUE_EXPLANATION
    if len(q.split()) <= 3 and not _CONCEPTUAL_RE.search(q) and not _CHALLENGE_RE.search(q):
        return FollowupType.CONTINUE_EXPLANATION
    if _CONCEPTUAL_RE.search(q) or _CHALLENGE_RE.search(q):
        return FollowupType.NEW_TOPIC
    return FollowupType.NEW_TOPIC


def _contextual_query(query: str, conversation_history: list[dict] | None) -> str:
    """Augment short follow-ups with prior user turn for embedding."""
    q = (query or "").strip()
    if not conversation_history:
        return q
    prior = ""
    for turn in reversed(conversation_history):
        if (turn.get("role") or "").lower() == "user":
            prior = (turn.get("content") or "").strip()
            if prior:
                break
    if not prior:
        return q
    if len(q.split()) <= 12:
        return f"After discussing: {prior[:180]}. Student now says: {q}"
    return q


def _load_intent_prototype_embeddings() -> dict[FollowupType, list[Any]] | None:
    global _prototype_cache
    if _prototype_cache is not None:
        return _prototype_cache

    from app.services.image_service.figure_context_bge import embed_texts

    cache: dict[FollowupType, list[Any]] = {}
    for intent, phrases in _INTENT_PROTOTYPES.items():
        vectors = embed_texts(phrases)
        valid = [v for v in vectors if v is not None]
        if valid:
            cache[intent] = valid
    _prototype_cache = cache if cache else {}
    return _prototype_cache or None


def _classify_by_bge(
    query: str,
    conversation_history: list[dict] | None,
) -> IntentClassification | None:
    from app.services.image_service.figure_context_bge import cosine_100, embed_query
    from app.services.vector_service import is_embedding_model_loaded

    if not is_embedding_model_loaded():
        return None

    prototypes = _load_intent_prototype_embeddings()
    if not prototypes:
        return None

    text = _contextual_query(query, conversation_history)
    qvec = embed_query(text[:2000])
    if qvec is None:
        return None

    best_intent = FollowupType.NEW_TOPIC
    best_score = 0.0
    second_score = 0.0

    for intent, vectors in prototypes.items():
        intent_best = max(cosine_100(qvec, pvec) for pvec in vectors)
        if intent_best > best_score:
            second_score = best_score
            best_score = intent_best
            best_intent = intent
        elif intent_best > second_score:
            second_score = intent_best

    margin = best_score - second_score
    if best_score < _MIN_BGE_SCORE or margin < _MIN_BGE_MARGIN:
        return None

    return IntentClassification(
        followup_type=best_intent,
        confidence=round(best_score / 100.0, 3),
        method="bge",
    )


def classify_followup_intent(
    query: str,
    conversation_history: list[dict] | None = None,
) -> IntentClassification:
    """
    Hybrid intent resolution: regex first, BGE for ambiguous follow-ups.

    BGE is attempted when regex yields NEW_TOPIC on a short utterance with history,
    or when regex confidence is low (generic short phrase).
    """
    q = (query or "").strip()
    regex_intent = classify_followup_regex(q)
    history = conversation_history or []

    if regex_intent in _HIGH_CONFIDENCE_INTENTS and regex_intent != FollowupType.NEW_TOPIC:
        return IntentClassification(regex_intent, 1.0, "regex")

    # Conceptual openers ("what is weather?") are new topics — never let BGE relabel them
    if regex_intent == FollowupType.NEW_TOPIC and (
        _CONCEPTUAL_RE.search(q) or _CHALLENGE_RE.search(q)
    ):
        return IntentClassification(FollowupType.NEW_TOPIC, 0.95, "regex")

    should_try_bge = (
        regex_intent in (FollowupType.NEW_TOPIC, FollowupType.NONE)
        or (regex_intent == FollowupType.CONTINUE_EXPLANATION and len(q.split()) > 3)
    )
    if should_try_bge and (history or len(q.split()) <= 10):
        bge = _classify_by_bge(q, history)
        if bge is not None:
            if regex_intent == FollowupType.NEW_TOPIC:
                if bge.followup_type == FollowupType.NEW_TOPIC:
                    return bge
                return IntentClassification(FollowupType.NEW_TOPIC, 0.6, "regex")
            if bge.followup_type != FollowupType.NEW_TOPIC:
                return IntentClassification(
                    bge.followup_type,
                    bge.confidence,
                    "hybrid",
                )

    conf = 0.95 if regex_intent != FollowupType.NEW_TOPIC else 0.6
    return IntentClassification(regex_intent, conf, "regex")


def is_clarification_followup(
    query: str, conversation_history: list[dict] | None
) -> bool:
    """True when the student is confused about the tutor's previous reply."""
    q = (query or "").strip()
    if not q or not conversation_history:
        return False
    if classify_followup_regex(q) != FollowupType.CLARIFICATION:
        return False
    for turn in reversed(conversation_history):
        if (turn.get("role") or "").lower() == "assistant":
            return bool((turn.get("content") or "").strip())
    return False


def answer_type_for_followup(followup: FollowupType) -> str | None:
    """Map dialogue act to chat_service answer_type (None → use detect_answer_type)."""
    mapping: dict[FollowupType, str] = {
        FollowupType.CLARIFICATION: "clarification",
        FollowupType.GREETING: "greeting",
        FollowupType.SIMPLIFY: "simplified",
        FollowupType.ASK_SUMMARY: "summary",
        FollowupType.GENERATE_QUESTIONS: "quiz",
        FollowupType.GENERATE_MCQ: "mcq",
        FollowupType.ASK_EXAMPLE: "short-answer",
        FollowupType.ASK_COMPARISON: "paragraph",
        FollowupType.CONTINUE_EXPLANATION: "paragraph",
    }
    return mapping.get(followup)
