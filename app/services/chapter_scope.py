"""
Chapter Awareness Mode — programmatic coverage assessment and student choice handling.

Detects when a question is fully, partially, or not covered by the selected chapter
using retrieval evidence (not prompt-only instructions).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)

_CHAPTER_NUM_RE = re.compile(r"chapter\s*(\d+)", re.I)

_TOPIC_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
    "and", "or", "to", "for", "with", "by", "from", "this", "that", "it",
    "its", "as", "be", "been", "being", "have", "has", "had", "not", "no",
    "explain", "describe", "what", "how", "why", "when", "where", "about",
    "tell", "give", "can", "you", "me", "please", "define", "meaning",
    "summarize", "summarise", "summary", "overview",
    "chapter",  # meta — "tell about this chapter" is about the session, not a topic term
    "lesson", "topic",  # meta — "the lesson" refers to current session, not a search term
})

# Tutoring meta-language: never a curriculum topic on its own ("explain in deep").
# Excluded only from chapter-wall scoring — chapter titles and math concept matching
# still use the full vocabulary ("Simple Equations", "Understanding Quadrilaterals").
_META_TUTORING_WORDS = frozenset({
    "deep", "deeper", "deeply", "understand", "understanding", "understood",
    "telling", "detail", "details", "detailed", "want", "more",
    "easier", "again", "confused", "clear",
    "say", "saying", "told", "asking", "asked",
})

_CHAPTER_META_QUERY_RE = re.compile(
    r"(?:"
    r"(?:tell(?:\s+me)?|explain|describe|summarize|summarise|summary|overview|introduce|introduction)\s+"
    r"(?:(?:me|us)\s+)?(?:about\s+)?(?:(?:this|the|current)\s+)?chapter\b"
    r"|(?:(?:this|the|current)\s+chapter)\s+(?:is\s+)?about\b"
    r"|what(?:'s|\s+(?:is|does|will|are))\s+(?:in|this|the)\s+chapter\b"
    r"|topics?\s+(?:in|of|from|covered\s+in)\s+(?:(?:this|the|current)\s+)?chapter\b"
    r")",
    re.I,
)

# Current-lesson requests — "the lesson" means the selected chapter session, not a topic term.
_CURRENT_LESSON_QUERY_RE = re.compile(
    r"(?:"
    r"(?:tell(?:\s+me)?|explain|describe|summarize|summarise|summary|overview|introduce|introduction|teach(?:\s+me)?)\s+"
    r"(?:(?:me|us)\s+)?(?:about\s+)?(?:(?:this|the|current)\s+)?(?:lesson|topic)\b"
    r"|(?:(?:this|the|current)\s+(?:lesson|topic))\s+(?:is\s+)?about\b"
    r"|what(?:'s|\s+(?:is|does|will|are))\s+(?:in|this|the)\s+(?:lesson|topic)\b"
    r"|(?:what\s+(?:is|are)\s+)?(?:(?:this|the|current)\s+)?(?:lesson|topic)\s+about\b"
    r"|(?:the\s+)?(?:lesson|topic)\s+(?:we(?:'re|\s+are)\s+(?:learning|studying|covering))\b"
    r")",
    re.I,
)

_SKIP_TOPIC_SCOPE_RE = re.compile(
    r"^(hi|hello|hey|hii|thanks|thank you|yes|yeah|yep|no|ok|okay|sure|"
    r"continue|go on|tell me more|explain more|simplify|summarize|summarise|"
    r"i\s+(?:did\s+not|don'?t)\s+understand|i\s+am\s+confused|"
    r"what\s+do\s+you\s+mean|explain\s+again|"
    r"how\s+many\s+(?:maps?|figures?|figs?)\b|"
    r"(?:can\s+you\s+)?explain(?:\s+\w+){0,6}\s+(?:in\s+)?(?:deep(?:ly)?|detail)|"
    r"explain\s+this\s+deeply|explain\s+that\s+deeply)\b",
    re.I,
)

# Deepen / clarify follow-ups — continue current lesson, not a new chapter topic.
_DEEPEN_FOLLOWUP_RE = re.compile(
    r"(?:"
    r"\b(?:explain|tell)(?:\s+\w+){0,8}\s+(?:in\s+)?(?:deep(?:ly)?|more\s+detail|detail)\b"
    r"|\b(?:in\s+)?deep(?:ly)?\b"
    r"|\bmore\s+deep\b"
    r"|\bdeep\s+telling\b"
    r"|\b(?:want|need)\s+to\s+understand\b"
    r"|\bexplain\s+(?:this|that|it|what\s+you)\b"
    r"|\bhow\s+did\s+you\s+get\b"
    r"|\bmake\s+it\s+(?:simple|easier)\b"
    r"|\bsay\s+it\s+again\b"
    r")",
    re.I,
)

# Session follow-ups — summary, clarification, pronouns; not new chapter topics.
_SESSION_CONTINUATION_RE = re.compile(
    r"\b(?:"
    r"summarize|summarise|summary|key points?|main points?|recap|overview|"
    r"explain again|in simple words?|simpler|another example|one more example|"
    r"why|how come|what happened next|tell me more|go on|continue|"
    r"deeply|in deep|more deep|more detail|in detail|elaborate|"
    r"understand|understood|confused|"
    r"that|this|it|those|these|he|she|they|him|her|them|"
    r"previous|earlier|second point|first point|again"
    r")\b",
    re.I,
)

# Explicit chapter switch — only these should trigger chapter-change flows.
_EXPLICIT_CHAPTER_SWITCH_RE = re.compile(
    r"\b(?:"
    r"(?:teach|learn|study|open|switch\s+to|move\s+to|go\s+to)\s+(?:me\s+)?chapter\s*\d+"
    r"|chapter\s*\d+\s+(?:please|now|next)"
    r"|(?:next|another)\s+chapter"
    r")\b",
    re.I,
)

# Teaching moves — not out-of-chapter topics (e.g. "conduct quiz", "give me a problem")
_PEDAGOGICAL_INTENT_RE = re.compile(
    r"(?:"
    r"\b(?:conduct|start|give|do|run|make)\s+(?:a\s+|me\s+a\s+|an?\s+)?(?:quiz|test|exam)\b"
    r"|\b(?:quiz|test)\s+me\b"
    r"|\bask\s+me\s+(?:a\s+)?(?:question|questions|quiz)\b"
    r"|\bgive\s+me\s+(?:a\s+|an?\s+)?(?:practice\s+)?(?:problem|question|exercise|quiz)\b"
    r"|\bpractice\s+(?:problem|problems|questions?)\b"
    r"|\b(?:real[- ]?life\s+example|quick\s+quiz|measure\s+weather)\b"
    r")",
    re.I,
)

# Student wants a textbook figure — not asking about the word "image" as a topic.
_IMAGE_REQUEST_RE = re.compile(
    r"\b("
    r"with\s+(?:an?\s+)?(?:images?|diagrams?|pictures?|figures?|maps?|illustrations?)|"
    r"show\s+(?:me\s+)?(?:an?\s+|the\s+)?(?:images?|diagrams?|pictures?|figures?|maps?|illustrations?)|"
    r"include\s+(?:an?\s+)?images?|"
    r"(?:see|want|need)\s+(?:an?\s+)?(?:images?|diagrams?|pictures?|figures?)|"
    r"textbook\s+(?:diagram|figure|image)s?"
    r")\b",
    re.I,
)

# Numeric / solve intents — student may bring a problem that uses chapter concepts
# but is not copied from the textbook.
_MATH_PROBLEM_RE = re.compile(
    r"(?:"
    r"\b(?:solve|calculate|compute|evaluate|simplify|prove|derive|work\s+out)\b"
    r"|\bfind\s+(?:the\s+)?(?:value|area|volume|perimeter|surface|side|length|"
    r"breadth|height|cube|square|root|product|sum|difference|quotient)\b"
    r"|\bwhat\s+is\s+(?:the\s+)?(?:area|volume|perimeter|surface\s+area|cube|square)\b"
    r"|\bhow\s+(?:many|much|long|far)\b"
    r"|\d"
    r")",
    re.I,
)

_MATH_SUBJECT_RE = re.compile(
    r"math|algebra|geometry|calculus|arithmetic|trigonometry|mensuration",
    re.I,
)

_AWARENESS_MARKER = "How would you like to continue?"

_STAY_CHOICE_RE = re.compile(
    r"^(?:a\b|option\s*a\b|stay(?:\s+within)?(?:\s+the)?\s+current\s+chapter|"
    r"current\s+chapter\s+only|within\s+this\s+chapter|chapter\s+only)\b",
    re.I,
)
_SWITCH_CHOICE_RE = re.compile(
    r"^(?:b\b|option\s*b\b|switch(?:\s+chapter)?|other\s+chapter|"
    r"relevant\s+chapter|change\s+chapter)\b",
    re.I,
)
_GENERAL_CHOICE_RE = re.compile(
    r"^(?:c\b|option\s*c\b|general(?:\s+explanation)?|full\s+explanation|"
    r"explain\s+anyway|beyond\s+(?:this\s+)?chapter)\b",
    re.I,
)


class ChapterCoverageLevel(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class ChapterScopeChoice(str, Enum):
    STAY = "stay"
    SWITCH = "switch"
    GENERAL = "general"


@dataclass
class ChapterCoverageAssessment:
    level: ChapterCoverageLevel
    topic_label: str
    current_chapter_label: str
    other_chapter_label: str | None = None
    other_chapter_id: str | None = None
    selected_hits: int = 0
    required_hits: int = 1
    selected_coverage: int = 0


def chapter_number_from_name(name: str) -> int | None:
    """Extract chapter number from labels like 'Chapter 2 - Understanding the Weather'."""
    m = _CHAPTER_NUM_RE.search(name or "")
    return int(m.group(1)) if m else None


def extract_mentioned_chapter_numbers(query: str) -> set[int]:
    return {int(m.group(1)) for m in _CHAPTER_NUM_RE.finditer(query or "")}


def selected_chapter_numbers(chapter_names: list[str]) -> set[int]:
    nums: set[int] = set()
    for name in chapter_names:
        n = chapter_number_from_name(name)
        if n is not None:
            nums.add(n)
    return nums


def substantive_query_terms(query: str) -> set[str]:
    terms = set(re.findall(r"\w+", (query or "").lower()))
    return {t for t in terms if len(t) >= 3 and t not in _TOPIC_STOPWORDS}


def content_substantive_terms(query: str) -> set[str]:
    """Topic terms for chapter-wall scoring — drops tutoring meta words."""
    return {t for t in substantive_query_terms(query) if t not in _META_TUTORING_WORDS}


def is_meta_only_follow_up(query: str) -> bool:
    """True when the utterance has no curriculum content terms (deepen/clarify only)."""
    q = (query or "").strip()
    if not q:
        return True
    if _DEEPEN_FOLLOWUP_RE.search(q):
        return not content_substantive_terms(q)
    if _SKIP_TOPIC_SCOPE_RE.match(q):
        return True
    terms = content_substantive_terms(q)
    return not terms and bool(_SESSION_CONTINUATION_RE.search(q) or len(q.split()) <= 6)


def is_current_lesson_query(query: str) -> bool:
    """True when the student refers to the current lesson/topic session, not a new search term."""
    q = (query or "").strip()
    if not q:
        return False
    if _CHAPTER_META_QUERY_RE.search(q):
        return True
    return bool(_CURRENT_LESSON_QUERY_RE.search(q))


def current_lesson_retrieval_query(chapter_label: str) -> str:
    """RAG query for overview/introduction of the active chapter."""
    label = (chapter_label or "").strip()
    focus = _chapter_focus_hint(label)
    parts = ["main concepts", "key ideas", "introduction", "overview"]
    if label:
        parts.append(label)
    if focus and focus.lower() not in label.lower():
        parts.append(focus)
    return " ".join(parts)


def is_chapter_awareness_prompt(text: str) -> bool:
    return _AWARENESS_MARKER in (text or "")


def _format_chapter_list(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]}, {names[1]}, and {len(names) - 2} more"


def _current_chapter_label(chapter_names: list[str] | None) -> str:
    names = [n.strip() for n in (chapter_names or []) if n and str(n).strip()]
    if not names:
        return "your selected chapter"
    return _format_chapter_list(names)


def _chapter_focus_hint(chapter_label: str) -> str:
    """Short focus line from chapter title after the dash."""
    label = (chapter_label or "").strip()
    if " - " in label:
        return label.split(" - ", 1)[1].strip()
    return label


def build_chapter_awareness_message(assessment: ChapterCoverageAssessment) -> str:
    """Student-facing message for not-covered topics with a/b/c choices."""
    topic = assessment.topic_label or "this topic"
    current = assessment.current_chapter_label

    if assessment.other_chapter_label:
        other = assessment.other_chapter_label
        opener = f"{topic} is in {other}, not in {current}."
        switch_line = f"b) Switch to {other} — open AI Tutor"
    else:
        opener = f"{topic} is not covered in {current}."
        switch_line = "b) Switch to the right chapter — open AI Tutor"

    return (
        f"{opener}\n\n"
        "How would you like to continue?\n"
        f"a) Stay in {current}\n"
        f"{switch_line}\n"
        
        "Reply a or b."
    )


def build_stay_in_chapter_message(current_chapter_label: str) -> str:
    return (
        f"Sure — let's stay with **{current_chapter_label}**.\n\n"
        f"Ask me any question about the topics covered in this chapter, "
        f"and I'll answer using the textbook first."
    )


def build_switch_chapter_message(
    *,
    other_chapter_label: str | None,
    current_chapter_label: str,
) -> str:
    if other_chapter_label:
        return (
            f"To learn about that topic from the textbook, open **Learning Studio**, "
            f"select **{other_chapter_label}**, and start a new session.\n\n"
            f"When you're ready, you can return to **{current_chapter_label}** anytime."
        )
    return (
        "Open **Learning Studio**, select the chapter that covers this topic, "
        "and start a new session there.\n\n"
        f"You can return to **{current_chapter_label}** whenever you like."
    )


def build_partial_coverage_guidance(assessment: ChapterCoverageAssessment) -> str:
    extra = ""
    if assessment.other_chapter_label:
        extra = (
            f" Additional details on **{assessment.topic_label}** appear in "
            f"**{assessment.other_chapter_label}**."
        )
    return (
        "CHAPTER COVERAGE ASSESSMENT: PARTIAL\n"
        f"The question is only partially covered by **{assessment.current_chapter_label}**.\n"
        "- First answer the portion supported by the chapter context below.\n"
        f"- Then briefly cover any missing related piece with "
        f"\"Beyond this chapter...\" (keep it short and age-appropriate).{extra}\n"
        "- Do NOT invent textbook-only facts (page numbers, figure names).\n"
        "- Do NOT force an a/b/c menu. Optionally offer a deeper chapter switch at the end."
    )


def build_current_lesson_guidance(chapter_label: str) -> str:
    focus = _chapter_focus_hint(chapter_label) or chapter_label
    return (
        "CHAPTER COVERAGE: CURRENT LESSON\n"
        f"The student is asking about the current lesson ({chapter_label}).\n"
        f"- Introduce and explain the main ideas of **{focus}** using the chapter text.\n"
        "- Do NOT treat 'lesson' or 'topic' as an uncovered keyword.\n"
        "- Do NOT show an a/b/c choice menu.\n"
        "- Answer warmly as a teacher explaining what this chapter covers."
    )


def build_general_explanation_guidance(topic_label: str) -> str:
    return (
        "CHAPTER COVERAGE: GENERAL EXPLANATION (beyond current chapter)\n"
        f"Give a clear general explanation of **{topic_label}** beyond the current chapter.\n"
        "- Begin with **Beyond this chapter...** or **Additional context...**\n"
        "- You may use accurate general educational knowledge.\n"
        "- Briefly remind them which chapter they are studying, then answer clearly.\n"
        "- Do NOT show an a/b/c choice menu."
    )


def build_visual_follow_up_guidance() -> str:
    return (
        "CHAPTER VISUAL REQUEST\n"
        "The student asked for a textbook diagram or image for the topic you just discussed.\n"
        "- Answer briefly using the chapter; related figures are shown separately in the UI.\n"
        "- Do NOT treat 'image' or 'diagram' as an out-of-chapter topic.\n"
        "- Do NOT show an a/b/c choice menu."
    )


def is_image_follow_up_request(query: str) -> bool:
    return bool(_IMAGE_REQUEST_RE.search(query or ""))


def prior_user_question(conversation_history: list[dict] | None) -> str | None:
    """Most recent real question before an image-only follow-up."""
    if not conversation_history:
        return None
    for turn in reversed(conversation_history):
        if (turn.get("role") or "").lower() != "user":
            continue
        content = (turn.get("content") or "").strip()
        if content and not is_image_follow_up_request(content):
            return content
    return None


def build_related_math_solve_guidance(chapter_label: str) -> str:
    """Student brought a practice problem that uses this chapter's concepts."""
    return (
        "CHAPTER COVERAGE: RELATED MATH PRACTICE (concept match, not a textbook copy)\n"
        f"The student asked to solve a mathematics problem related to **{chapter_label}**.\n"
        "- Answer the student's exact question first (same numbers/conditions — do not change them).\n"
        "- Explain briefly how the answer was obtained; show step-by-step math when appropriate.\n"
        "- SOLVE even if these exact numbers/wording are not in the textbook.\n"
        "- Use the chapter's concepts, definitions, and methods from the context when available.\n"
        "- You may apply standard mathematics for this grade to finish the calculation.\n"
        "- STOP after the answer and explanation only if a follow-up does not fit; otherwise end with ONE short "
        "follow-up question to check understanding. Do NOT lead with a different problem instead of answering.\n"
        "- Optionally note briefly that this is practice using the chapter idea, not a printed exercise.\n"
        "- Do NOT refuse, do NOT say the problem is missing from the book, and do NOT show an a/b/c menu."
    )


def _is_mathematics_subject(subject_name: str) -> bool:
    return bool(_MATH_SUBJECT_RE.search(subject_name or ""))


def chapter_concept_terms(chapter_names: list[str] | None) -> set[str]:
    """Content words from chapter titles (e.g. square, cube from 'A Square and A Cube')."""
    terms: set[str] = set()
    for name in chapter_names or []:
        focus = _chapter_focus_hint(name or "")
        terms |= substantive_query_terms(focus)
        terms |= substantive_query_terms(name or "")
    terms.discard("chapter")
    return terms


def looks_like_math_problem(query: str) -> bool:
    return bool(_MATH_PROBLEM_RE.search(query or ""))


def is_concept_related_math_problem(
    query: str,
    *,
    subject_name: str,
    chapter_names: list[str] | None,
    docs: list | None = None,
) -> bool:
    """True when a math solve/practice prompt uses this chapter's concept words."""
    if not _is_mathematics_subject(subject_name):
        return False
    if not looks_like_math_problem(query):
        return False
    concepts = chapter_concept_terms(chapter_names)
    if not concepts:
        return False
    q_lower = (query or "").lower()
    q_terms = substantive_query_terms(query)
    if q_terms & concepts:
        return True
    if any(c in q_lower for c in concepts):
        return True
    # Soft: retrieved chapter text shares a concept word with the problem
    if docs:
        for doc in docs[:8]:
            text = (getattr(doc, "page_content", "") or "").lower()
            if any(c in text for c in concepts) and (
                looks_like_math_problem(query) and _max_term_hits(q_terms, [doc]) > 0
            ):
                return True
    return False


_CHOICE_LETTER_RE = re.compile(
    r"(?:^|[\s,;])(?:i(?:'ll| will)?\s+)?"
    r"(?:go with|choose|pick|take|select|want|prefer|option|letter)\s*"
    r"(?:option\s*)?['\"]?\s*([abc])\b",
    re.I,
)


def _extract_awareness_choice_letter(query: str) -> str | None:
    """Return a/b/c when the student is picking from a chapter-awareness menu."""
    q = (query or "").strip()
    if not q:
        return None
    if _STAY_CHOICE_RE.match(q):
        return "a"
    if _SWITCH_CHOICE_RE.match(q):
        return "b"
    if _GENERAL_CHOICE_RE.match(q):
        return "c"
    m = _CHOICE_LETTER_RE.search(q)
    if m:
        return m.group(1).lower()
    if re.fullmatch(r"[abc]", q, re.I):
        return q.lower()
    return None


def detect_chapter_scope_choice(
    query: str,
    conversation_history: list[dict] | None,
) -> ChapterScopeChoice | None:
    """Parse a/b/c (or phrases) after a chapter-awareness prompt."""
    q = (query or "").strip()
    if not q or not conversation_history:
        return None
    last_asst = ""
    for turn in reversed(conversation_history):
        if (turn.get("role") or "").lower() == "assistant":
            last_asst = (turn.get("content") or "").strip()
            break
    if not is_chapter_awareness_prompt(last_asst):
        return None
    letter = _extract_awareness_choice_letter(q)
    if letter == "a":
        return ChapterScopeChoice.STAY
    if letter == "b":
        return ChapterScopeChoice.SWITCH
    if letter == "c":
        return ChapterScopeChoice.GENERAL
    return None


def original_question_before_awareness(
    conversation_history: list[dict] | None,
) -> str | None:
    """User question that triggered the chapter-awareness prompt."""
    if not conversation_history:
        return None
    seen_awareness = False
    for turn in reversed(conversation_history):
        role = (turn.get("role") or "").lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant" and is_chapter_awareness_prompt(content):
            seen_awareness = True
            continue
        if seen_awareness and role == "user":
            return content
    return None


def other_chapter_from_awareness_prompt(text: str) -> str | None:
    m = re.search(r"Switch to (.+?) — open Learning Studio", text or "", re.I)
    if m:
        label = m.group(1).strip()
        if label.lower().startswith("the right chapter"):
            return None
        return label
    return None


def chapter_scope_mismatch_message(
    query: str,
    chapter_names: list[str] | None,
) -> str | None:
    """Redirect when the query explicitly names a chapter outside the selection."""
    names = [n.strip() for n in (chapter_names or []) if n and str(n).strip()]
    if not names:
        return None

    mentioned = extract_mentioned_chapter_numbers(query)
    if not mentioned:
        return None

    selected_nums = selected_chapter_numbers(names)
    if not selected_nums:
        return None

    out_of_scope = mentioned - selected_nums
    if not out_of_scope:
        return None

    current_label = _format_chapter_list(names)
    wrong_labels = [f"Chapter {n}" for n in sorted(out_of_scope)]
    wrong_label = wrong_labels[0] if len(wrong_labels) == 1 else ", ".join(wrong_labels)
    topic_nums = " ".join(f"chapter {n}" for n in sorted(out_of_scope))
    assessment = ChapterCoverageAssessment(
        level=ChapterCoverageLevel.NONE,
        topic_label=topic_nums,
        current_chapter_label=current_label,
        other_chapter_label=wrong_label,
    )
    return build_chapter_awareness_message(assessment)


def _should_skip_topic_scope_check(
    query: str,
    conversation_history: list[dict] | None = None,
) -> bool:
    q = (query or "").strip()
    if not q:
        return True
    if _CHAPTER_META_QUERY_RE.search(q):
        return True
    if is_current_lesson_query(q):
        return True
    if _PEDAGOGICAL_INTENT_RE.search(q):
        return True
    if is_image_follow_up_request(q):
        return True
    if _DEEPEN_FOLLOWUP_RE.search(q):
        return True
    if conversation_history and is_session_continuation_follow_up(q, conversation_history):
        return True
    if len(q.split()) <= 2 and _SKIP_TOPIC_SCOPE_RE.match(q):
        return True
    return bool(_SKIP_TOPIC_SCOPE_RE.match(q))


def is_explicit_chapter_switch_request(query: str) -> bool:
    """True only when the student clearly asks to change/open another chapter."""
    return bool(_EXPLICIT_CHAPTER_SWITCH_RE.search((query or "").strip()))


def is_session_continuation_follow_up(
    query: str,
    conversation_history: list[dict] | None,
) -> bool:
    """
    True when the student is continuing the current lesson (not starting a new topic).
    Length does not matter — a long follow-up can still be a continuation.
    """
    q = (query or "").strip()
    if not q or not conversation_history:
        if is_current_lesson_query(q):
            return True
        return False
    if is_explicit_chapter_switch_request(q):
        return False
    if _CHAPTER_META_QUERY_RE.search(q):
        return True
    if is_current_lesson_query(q):
        return True
    if _PEDAGOGICAL_INTENT_RE.search(q):
        return True
    if is_image_follow_up_request(q):
        return True
    if _DEEPEN_FOLLOWUP_RE.search(q):
        return True
    if _SESSION_CONTINUATION_RE.search(q):
        return True
    if is_related_chapter_follow_up(q, conversation_history):
        return True
    # Short utterances with prior teaching ("why", "ok", "yes") — not new searches.
    if len(q.split()) <= 4 and _recent_assistant_terms(conversation_history):
        return True
    return False


def _recent_assistant_terms(conversation_history: list[dict] | None) -> set[str]:
    if not conversation_history:
        return set()
    for turn in reversed(conversation_history):
        if (turn.get("role") or "").lower() != "assistant":
            continue
        content = (turn.get("content") or "").strip()
        if not content or is_chapter_awareness_prompt(content):
            continue
        return substantive_query_terms(content)
    return set()


def is_related_chapter_follow_up(
    query: str,
    conversation_history: list[dict] | None,
    *,
    min_overlap: int = 1,
) -> bool:
    """True when the new question continues a topic the tutor just taught."""
    q = (query or "").strip()
    if not conversation_history:
        return False
    prev = _recent_assistant_terms(conversation_history)
    if not prev:
        return False
    q_terms = substantive_query_terms(q)
    if not q_terms:
        # "why?", "how?", "again" — stopwords leave no terms; still a follow-up.
        return len(q.split()) <= 6
    return len(q_terms & prev) >= min_overlap


def _terms_match_chapter_titles(terms: set[str], chapter_names: list[str] | None) -> bool:
    """True when every query term appears in the selected chapter title(s)."""
    if not terms or not chapter_names:
        return False
    title_parts: list[str] = []
    for name in chapter_names:
        label = (name or "").strip()
        if label:
            title_parts.append(label.lower())
            title_parts.append(_chapter_focus_hint(label).lower())
    title_text = " ".join(title_parts)
    return bool(title_text) and all(term in title_text for term in terms)


def _term_hits_in_text(terms: set[str], text: str) -> int:
    if not terms:
        return 0
    lowered = (text or "").lower()
    return sum(1 for term in terms if term in lowered)


def _docs_with_all_terms(terms: set[str], docs: list) -> int:
    if not terms or not docs:
        return 0
    count = 0
    for doc in docs[:10]:
        if _term_hits_in_text(terms, getattr(doc, "page_content", "") or "") == len(terms):
            count += 1
    return count


def _max_term_hits(terms: set[str], docs: list) -> int:
    if not terms or not docs:
        return 0
    return max(
        _term_hits_in_text(terms, getattr(doc, "page_content", "") or "")
        for doc in docs[:10]
    )


def _upload_id_from_doc(doc) -> str:
    meta = getattr(doc, "metadata", None) or {}
    return str(meta.get("textbook_upload_id") or "")


@lru_cache(maxsize=64)
def _subject_upload_labels(board: str, class_level: str, subject_name: str) -> dict[str, str]:
    """Map textbook upload IDs to human chapter labels for a subject."""
    try:
        from sqlalchemy import select

        from app.core.database import SessionLocal
        from app.modules.catalog.models import BoardEnum, ClassEnum, ProcessingStatusEnum, TextbookUpload

        board_enum = BoardEnum(board)
        class_enum = ClassEnum(class_level)
    except ValueError:
        return {}

    try:
        with SessionLocal() as db:
            rows = list(
                db.scalars(
                    select(TextbookUpload)
                    .where(
                        TextbookUpload.board == board_enum,
                        TextbookUpload.class_level == class_enum,
                        TextbookUpload.subject_name == subject_name,
                        TextbookUpload.embedding_status == ProcessingStatusEnum.EMBEDDED,
                    )
                    .order_by(TextbookUpload.chapter)
                )
            )
    except Exception as exc:
        logger.warning("Failed to load subject uploads for topic scope: %s", exc)
        return {}

    labels: dict[str, str] = {}
    for row in rows:
        label = (row.chapter or "").strip() or row.file_name
        labels[str(row.id)] = label
    return labels


def _best_other_chapter(
    terms: set[str],
    *,
    collection_name: str,
    query: str,
    selected_set: set[str],
) -> tuple[str, int, int]:
    if not terms:
        return "", 0, 0
    from app.services.vector_service import retrieve_from_collection

    cross_docs = retrieve_from_collection(
        query,
        collection_name=collection_name,
        chapter_ids=None,
        k=15,
    )
    per_upload: dict[str, list] = {}
    for doc in cross_docs:
        uid = _upload_id_from_doc(doc)
        if not uid or uid in selected_set:
            continue
        per_upload.setdefault(uid, []).append(doc)

    best_id = ""
    best_hits = 0
    best_coverage = 0
    for uid, upload_docs in per_upload.items():
        hits = _max_term_hits(terms, upload_docs)
        coverage = _docs_with_all_terms(terms, upload_docs)
        if hits > best_hits or (hits == best_hits and coverage > best_coverage):
            best_hits = hits
            best_coverage = coverage
            best_id = uid
    return best_id, best_hits, best_coverage


def assess_chapter_coverage(
    query: str,
    *,
    docs: list,
    collection_name: str,
    chapter_ids: list[str] | None,
    chapter_names: list[str] | None,
    board: str,
    class_level: str,
    subject_name: str,
) -> ChapterCoverageAssessment:
    """
  Assess whether the selected chapter fully, partially, or does not cover the question.
    """
    current_label = _current_chapter_label(chapter_names)
    terms = content_substantive_terms(query)
    # Student-facing label: keep the original phrase order. `terms` (used below
    # for scope matching) is an unordered set — joining it alphabetically turned
    # this into word salad when read back to the student ("is not covered in...").
    topic_label = (query or "").strip()[:80]

    if not chapter_ids:
        return ChapterCoverageAssessment(
            level=ChapterCoverageLevel.FULL,
            topic_label=topic_label,
            current_chapter_label=current_label,
        )
    if _should_skip_topic_scope_check(query):
        return ChapterCoverageAssessment(
            level=ChapterCoverageLevel.FULL,
            topic_label=topic_label,
            current_chapter_label=current_label,
        )
    if not terms:
        return ChapterCoverageAssessment(
            level=ChapterCoverageLevel.FULL,
            topic_label=topic_label or "your question",
            current_chapter_label=current_label,
        )
    if _terms_match_chapter_titles(terms, chapter_names):
        return ChapterCoverageAssessment(
            level=ChapterCoverageLevel.FULL,
            topic_label=topic_label,
            current_chapter_label=current_label,
        )

    selected_hits = _max_term_hits(terms, docs)
    selected_coverage = _docs_with_all_terms(terms, docs)
    required_hits = max(1, len(terms))
    selected_set = {str(cid) for cid in chapter_ids}

    other_id, other_hits, other_coverage = _best_other_chapter(
        terms,
        collection_name=collection_name,
        query=query,
        selected_set=selected_set,
    )
    labels = _subject_upload_labels(board, class_level, subject_name)
    other_label = labels.get(other_id) if other_id else None

    base = ChapterCoverageAssessment(
        level=ChapterCoverageLevel.FULL,
        topic_label=topic_label,
        current_chapter_label=current_label,
        other_chapter_label=other_label,
        other_chapter_id=other_id or None,
        selected_hits=selected_hits,
        required_hits=required_hits,
        selected_coverage=selected_coverage,
    )

    # Strong coverage in selected chapter
    if selected_hits >= required_hits and selected_coverage >= 1:
        if other_hits > selected_hits and other_coverage >= 1:
            base.level = ChapterCoverageLevel.PARTIAL
            return base
        base.level = ChapterCoverageLevel.FULL
        return base

    # Another chapter clearly owns the topic
    if other_id and other_hits >= required_hits and other_hits > selected_hits:
        base.level = ChapterCoverageLevel.NONE
        return base

    # Weak or no match in selected chapter
    if selected_hits == 0:
        base.level = ChapterCoverageLevel.NONE
        return base

    if selected_hits < required_hits:
        base.level = ChapterCoverageLevel.PARTIAL
        return base

    base.level = ChapterCoverageLevel.FULL
    return base


def topic_chapter_mismatch_message(
    query: str,
    *,
    docs: list,
    collection_name: str,
    chapter_ids: list[str] | None,
    chapter_names: list[str] | None,
    board: str,
    class_level: str,
    subject_name: str,
) -> str | None:
    """Legacy entry point — returns awareness message when coverage is NONE."""
    assessment = assess_chapter_coverage(
        query,
        docs=docs,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
    )
    if assessment.level == ChapterCoverageLevel.NONE:
        return build_chapter_awareness_message(assessment)
    return None


def resolve_chapter_scope_message(
    query: str,
    *,
    docs: list | None = None,
    collection_name: str = "",
    chapter_ids: list[str] | None = None,
    chapter_names: list[str] | None = None,
    board: str = "",
    class_level: str = "",
    subject_name: str = "",
) -> str | None:
    """Run explicit and topic-based chapter scope checks."""
    explicit = chapter_scope_mismatch_message(query, chapter_names)
    if explicit:
        return explicit
    if docs is None:
        return None
    return topic_chapter_mismatch_message(
        query,
        docs=docs,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
    )


def resolve_chapter_awareness_turn(
    query: str,
    *,
    docs: list,
    conversation_history: list[dict] | None,
    collection_name: str,
    chapter_ids: list[str] | None,
    chapter_names: list[str] | None,
    board: str,
    class_level: str,
    subject_name: str,
    scope_query: str | None = None,
) -> tuple[str | None, str, ChapterCoverageAssessment | None, str]:
    """
    Handle chapter-awareness for one student turn.

    Returns:
        early_response — if set, return this to the student (no LLM call)
        effective_query — query to answer (may be the original question after option c)
        assessment — coverage assessment when applicable
        coverage_guidance — extra system-prompt block for the LLM
    """
    explicit = chapter_scope_mismatch_message(query, chapter_names)
    if explicit:
        return explicit, query, None, ""

    scope_q = (scope_query or query).strip()
    session_follow_up = is_session_continuation_follow_up(query, conversation_history)

    choice = detect_chapter_scope_choice(query, conversation_history)
    current_label = _current_chapter_label(chapter_names)

    if choice == ChapterScopeChoice.STAY:
        return build_stay_in_chapter_message(current_label), query, None, ""

    if choice == ChapterScopeChoice.SWITCH:
        last_asst = ""
        if conversation_history:
            for turn in reversed(conversation_history):
                if (turn.get("role") or "").lower() == "assistant":
                    last_asst = turn.get("content") or ""
                    break
        other = other_chapter_from_awareness_prompt(last_asst)
        return (
            build_switch_chapter_message(
                other_chapter_label=other,
                current_chapter_label=current_label,
            ),
            query,
            None,
            "",
        )

    effective_query = query
    coverage_guidance = ""

    if choice == ChapterScopeChoice.GENERAL:
        original = original_question_before_awareness(conversation_history)
        if original:
            effective_query = original
            coverage_guidance = build_general_explanation_guidance(original)
            return None, effective_query, None, coverage_guidance

    # "with an image" / "show me a diagram" — reuse prior question, fetch figures
    if is_image_follow_up_request(effective_query):
        prior = prior_user_question(conversation_history)
        if prior:
            effective_query = prior
        return None, effective_query, None, build_visual_follow_up_guidance()

    # Current lesson / chapter meta — teach the selected chapter, never a/b/c wall.
    if is_current_lesson_query(query) or is_current_lesson_query(scope_q):
        lesson_q = current_lesson_retrieval_query(current_label)
        return (
            None,
            lesson_q or effective_query,
            None,
            build_current_lesson_guidance(current_label),
        )

    # Session continuation — answer in current lesson; never show a/b/c wall.
    if session_follow_up:
        prior = prior_user_question(conversation_history) or scope_q or effective_query
        return None, prior or effective_query, None, build_general_explanation_guidance(prior)

    # Related follow-up on a topic just taught → answer beyond chapter, no a/b/c wall
    if is_related_chapter_follow_up(scope_q, conversation_history) or is_related_chapter_follow_up(
        effective_query, conversation_history
    ):
        topic = scope_q or effective_query
        return None, effective_query, None, build_general_explanation_guidance(topic)

    assessment = assess_chapter_coverage(
        scope_q,
        docs=docs,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
    )

    if assessment.level == ChapterCoverageLevel.NONE:
        # Math practice using this chapter's concepts → solve even if not a textbook copy
        if is_concept_related_math_problem(
            scope_q,
            subject_name=subject_name,
            chapter_names=chapter_names,
            docs=docs,
        ):
            return (
                None,
                effective_query,
                assessment,
                build_related_math_solve_guidance(assessment.current_chapter_label),
            )
        # Soft miss with some chapter signal, or adjacent topic → explain beyond chapter
        if assessment.selected_hits > 0:
            return (
                None,
                effective_query,
                assessment,
                build_general_explanation_guidance(assessment.topic_label),
            )
        return build_chapter_awareness_message(assessment), effective_query, assessment, ""

    if assessment.level == ChapterCoverageLevel.PARTIAL:
        # Still solve related math practice; don't under-answer as "only partial theory"
        if is_concept_related_math_problem(
            scope_q,
            subject_name=subject_name,
            chapter_names=chapter_names,
            docs=docs,
        ):
            coverage_guidance = build_related_math_solve_guidance(
                assessment.current_chapter_label
            )
        else:
            coverage_guidance = build_partial_coverage_guidance(assessment)

    return None, effective_query, assessment, coverage_guidance


def resolve_chapter_scope_with_retrieval(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None,
    chapter_names: list[str] | None,
    board: str,
    class_level: str,
    subject_name: str,
    retrieval_query: str | None = None,
    conversation_history: list[dict] | None = None,
) -> str | None:
    """Explicit + topic scope checks using a fresh retrieval pass.

    Returns an early awareness message only for hard out-of-scope topics.
    Pedagogical intents, related follow-ups, and soft misses return None so the
    tutor can answer (with Beyond this chapter… when needed).
    """
    explicit = chapter_scope_mismatch_message(query, chapter_names)
    if explicit:
        return explicit
    if not chapter_ids:
        return None
    scope_q = (retrieval_query or query).strip()
    if _should_skip_topic_scope_check(scope_q, conversation_history):
        return None
    if _should_skip_topic_scope_check(query, conversation_history):
        return None
    if is_session_continuation_follow_up(query, conversation_history):
        return None
    if is_related_chapter_follow_up(scope_q, conversation_history):
        return None
    if is_related_chapter_follow_up(query, conversation_history):
        return None
    if is_image_follow_up_request(query):
        return None
    if is_concept_related_math_problem(
        scope_q, subject_name=subject_name, chapter_names=chapter_names
    ):
        return None

    from app.services.section_retrieval import retrieve_for_tutor_query

    docs, _, _ = retrieve_for_tutor_query(
        scope_q,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
    )
    assessment = assess_chapter_coverage(
        scope_q,
        docs=docs,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
    )
    if is_concept_related_math_problem(
        scope_q,
        subject_name=subject_name,
        chapter_names=chapter_names,
        docs=docs,
    ):
        return None
    # Voice/HTTP early gate: only hard NONE with zero chapter signal
    if assessment.level == ChapterCoverageLevel.NONE and assessment.selected_hits == 0:
        return build_chapter_awareness_message(assessment)
    return None
