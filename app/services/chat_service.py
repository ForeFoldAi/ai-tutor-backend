"""
AI Tutor chat service.

Key improvements over the original:
- Fully async: uses httpx.AsyncClient instead of requests (no event-loop blocking)
- Streaming: stream_chapter_qa() yields tokens for StreamingResponse
- Grade-aware prompt: adapts language complexity to class level (1-12)
- Tiered answer format: direct prose by default; full emoji sections only for exam/long requests
- Answer-type detection: honors brief/one-word and question intent (not every "what is" → essay)
- Question-type detection: factual, conceptual, analytical, opinion, problem-solving pedagogy
- Subject guidelines: Science, Math, History, Geography, Economics, Language/Literature
- Mathematics: mandatory section tutor format + SymPy math engine for verified steps
- No Redis tutor-answer cache (every question hits the LLM; chat history is Postgres)
- Fallback: keyword-chunk answer when Mistral key is missing
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, AsyncIterator

from app.core.student_messages import ANSWER_NOT_IN_CHAPTER

from langchain_core.prompts import PromptTemplate

from app.config import (
    CONTEXT_CHAR_BUDGET,
    IMAGE_RETRIEVAL_TIMEOUT_SEC,
    RETRIEVAL_K,
    TOP_RELATED_IMAGES,
)
from app.services import llm_client
from app.services.vector_service import InMemoryDocVectorStore

logger = logging.getLogger(__name__)


def _log_image_stage(stage: str, images: list[dict]) -> None:
    """Structured per-stage image trace for debugging pipeline leaks."""
    try:
        summary = [
            f"{x.get('file_name') or x.get('image_url', '?')}:"
            f"{(x.get('caption') or '')[:60]}"
            for x in (images or [])
        ]
    except Exception:
        summary = ["<unserializable>"]
    logger.debug("[IMAGE-STAGE] stage=%s count=%d images=%s", stage, len(images or []), summary)


# ── Answer-type detection ────────────────────────────────────────────────────

_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|hii|helo|heya|good (morning|afternoon|evening|night)|"
    r"how are you|how r u|what('s| is) up|sup|yo|namaste|hiya|howdy|"
    r"nice to meet|greet|thanks|thank you|bye|goodbye|see you|ok|okay|"
    r"cool|awesome|great|good|fine|i('m| am) (good|fine|ok|bored|tired|happy|sad)|"
    r"can you help|are you (there|ready|a bot|ai|real)|"
    r"what('s| is) your name|who are you|what are you)\b",
    re.I,
)
_EXPLICIT_ONE_WORD = re.compile(
    r"\b(one word|single word|in one word|just the word|one-word answer)\b",
    re.I,
)
_EXPLICIT_BRIEF = re.compile(
    r"\b(one line|one-liner|very brief|just one sentence|only one sentence)\b",
    re.I,
)
_EXPLICIT_SHORT = re.compile(
    r"\b("
    r"in short|short answer|short version|keep (it )?short|make (it )?short|"
    r"briefly|be brief|tell me briefly|quick answer|just briefly|in brief|"
    r"very short(?:\s+answer)?|answer in short|explain in short|"
    r"(?:give|want|need)\s+(?:me\s+)?(?:a\s+)?short|"
    r"only\s+short|shortly"
    r")\b",
    re.I,
)
_EXAM_PATTERNS = re.compile(
    r"\b(\d\s*marks?|long answer|write (in )?detail|elaborate|"
    r"essay|comprehensive|full answer|describe (in )?detail)\b",
    re.I,
)
_STEPWISE_PATTERNS = re.compile(
    r"\b(steps?|procedure|how (to|does|do)|process of|method|"
    r"working of|mechanism|stages? of)\b",
    re.I,
)
_SIMPLE_PATTERNS = re.compile(
    r"\b(simple|easy|in simple words|for kids|class [1-3]\b|"
    r"explain simply|layman)\b",
    re.I,
)
_CONCEPT_STARTS = re.compile(
    r"^(what is|what are|meaning of|definition of|define)\b",
    re.I,
)
_EXPLAIN_PATTERNS = re.compile(
    r"\b(explain|describe|why|how does|what happens|elaborate on|"
    r"discuss|talk about)\b",
    re.I,
)
_BULLET_PATTERNS = re.compile(
    r"\b(bullet points?|key points?|list (the|all|some)|points? (on|about)|"
    r"give points|write points)\b",
    re.I,
)
_DETAILED_PATTERNS = re.compile(
    r"\b("
    r"in detail|detailed(?:\s+explanation)?|explain in detail|explain fully|"
    r"full explanation|elaborate(?:\s+on)?|comprehensive(?:\s+answer)?|"
    r"with (?:all )?key points|important terms|complete explanation|"
    r"tell me (?:more|everything)|everything about|go deeper|more detail"
    r")\b",
    re.I,
)

_TEACH_LESSON_PATTERNS = re.compile(
    r"\b(?:teach|learn|study)\s+me\b.*\b(lesson|chapter)\b"
    r"|\b(?:teach|learn|study)\s+(?:me\s+)?this\b.*\b(lesson|chapter)\b"
    r"|\bdetailed\s+notes\b",
    re.I,
)
_FACTUAL_LIST_PATTERNS = re.compile(
    r"\b("
    r"what kind of|what types? of|what sort of|"
    r"which materials?|what materials?|"
    r"name the|list the|give the names? of|state the"
    r")\b",
    re.I,
)


def detect_answer_type(query: str) -> str:
    q = query.strip()
    # Greetings and casual chat get a warm-but-brief tutor reply
    if _GREETING_PATTERNS.match(q) and len(q.split()) <= 10:
        return "greeting"
    if _EXAM_PATTERNS.search(q):
        return "exam-format"
    if _BULLET_PATTERNS.search(q):
        return "bullet-points"
    if _STEPWISE_PATTERNS.search(q):
        return "stepwise"
    if _SIMPLE_PATTERNS.search(q):
        return "simplified"
    if _EXPLICIT_ONE_WORD.search(q):
        return "one-word"
    # Honor explicit short/brief requests before "what is …" → direct answer
    if _EXPLICIT_SHORT.search(q) or _EXPLICIT_BRIEF.search(q):
        return "brief"
    # Explicit detail / section tags → full structured teaching format
    if _DETAILED_PATTERNS.search(q):
        return "paragraph"
    if _TEACH_LESSON_PATTERNS.search(q):
        # ponytail: explicit request for lesson/note-level teaching should keep structured format.
        return "paragraph"
    # Simple "what is X?" → direct answer; deep explain / exam → longer formats
    if _CONCEPT_STARTS.match(q):
        if _EXPLAIN_PATTERNS.search(q) or _EXAM_PATTERNS.search(q):
            return "paragraph"
        return "short-answer"
    if _EXPLAIN_PATTERNS.search(q):
        # WHY/HOW/explain requests should be direct and scoped, not a full structured lesson.
        return "short-answer"
    if _FACTUAL_LIST_PATTERNS.search(q):
        return "factual"
    return "short-answer"


# ── Question-type detection (pedagogy layer) ─────────────────────────────────

_PROBLEM_SOLVING_PATTERNS = re.compile(
    r"\b("
    r"solve|calculate|find (?:the )?(?:value|answer|result)|compute|prove|derive|"
    r"simplify|evaluate|work out|show (?:that|your )?work|"
    r"how (?:far|long|many|much)|can (?:you|we|i) reach|"
    r"equation|formula|\d+\s*[\+\-\×\÷/=]|"
    r"\d+\s*(?:km|m|cm|mm|years?|days?|hours?|minutes?)\b|"
    r"square\s+root|cube\s+root|cure\s+root|perfect\s+square|perfect\s+cube|"
    r"squared|cubed"
    r")\b",
    re.I,
)
_OPINION_PATTERNS = re.compile(
    r"\b("
    r"opinion|debate|discuss (?:whether|if)|do you (?:think|agree)|"
    r"should (?:we|one)|evaluate|argue|perspective|viewpoint|"
    r"agree or disagree|in your view|what do you think"
    r")\b",
    re.I,
)
_ANALYTICAL_PATTERNS = re.compile(
    r"\b("
    r"why|compare|contrast|analy[sz]e|cause|effect|consequence|impact|"
    r"relationship|difference between|similarit(?:y|ies)|"
    r"what led to|how did.*(?:lead|result|affect|cause)"
    r")\b",
    re.I,
)
_CONCEPTUAL_PATTERNS = re.compile(
    r"\b("
    r"what is|what are|meaning of|definition|define|concept of|"
    r"explain|describe|how does|how do|significance of|importance of"
    r")\b",
    re.I,
)
_FACTUAL_PATTERNS = re.compile(
    r"^(who|when|where|which|how many|how much|name|list|state|give the)\b",
    re.I,
)


def detect_question_type(query: str) -> str:
    """Classify pedagogical intent: factual, conceptual, analytical, opinion, problem-solving."""
    q = query.strip()
    if _PROBLEM_SOLVING_PATTERNS.search(q):
        return "problem-solving"
    if _OPINION_PATTERNS.search(q):
        return "opinion"
    if _ANALYTICAL_PATTERNS.search(q):
        return "analytical"
    if _CONCEPTUAL_PATTERNS.search(q):
        return "conceptual"
    if _FACTUAL_PATTERNS.search(q):
        return "factual"
    return "conceptual"


_QUESTION_TYPE_INSTRUCTIONS: dict[str, str] = {
    "factual": (
        "FACTUAL: Give a direct answer first. Stay close to the chapter source. "
        "Use simple student-friendly language."
    ),
    "conceptual": (
        "CONCEPTUAL: Explain the meaning and relationships between ideas. "
        "Connect the explanation to the chapter material."
    ),
    "analytical": (
        "ANALYTICAL: Show reasoning step-by-step. Clearly distinguish facts from interpretations."
    ),
    "opinion": (
        "OPINION / DISCUSSION: Acknowledge multiple possible perspectives, evaluate evidence, "
        "then present a balanced conclusion."
    ),
    "problem-solving": (
        "PROBLEM SOLVING: Show the full process step-by-step. Explain why each step is used — "
        "not only the final answer."
    ),
}

_SUBJECT_GUIDELINES: dict[str, str] = {
    "Science": (
        "Emphasize evidence, observations, experiments, and cause-effect relationships."
    ),
    "Mathematics": (
        "You are an expert Mathematics Tutor for school students.\n"
        "Your goal is not just to give answers but to help students understand mathematical "
        "concepts and solve similar problems independently.\n\n"
        "Additional rules:\n"
        "- Use simple, student-friendly language adapted to the student's grade level.\n"
        "- Never provide only the final answer.\n"
        "- Explain the reasoning behind every operation.\n"
        "- For large numbers, explain place value when relevant.\n"
        "- For fractions, decimals, percentages, ratios, algebra, geometry, mensuration, "
        "statistics, and probability, explain the underlying concept before solving.\n"
        "- If multiple methods exist, show the simplest method first.\n"
        "- Encourage understanding rather than memorization.\n"
        "- Use real-life examples whenever helpful.\n"
        "- Use proper mathematical notation (fractions, exponents, symbols). "
        "Use ONLY dollar delimiters for LaTeX: inline $x^2 + 7x + 12$, display $$\\frac{a}{b}$$. "
        "Never use \\(...\\) or \\[...\\] — those show as broken slashes in the app.\n"
        "- Formula layout: ALWAYS show both forms — (1) word/symbol line, then (2) display LaTeX on the next line. "
        "Never write the word 'or' between them.\n"
        "- Solution layout: heading **Solution**, then 'Substituting the values:', then each step as "
        "$$= ...$$ on its own line.\n"
        "- Section titles must be bold: **To Find**, **Given Information**, **The Formula**, etc.\n"
        "- Use Indian number grouping in India-context problems (e.g. 12,00,000).\n"
        "- When MATH ENGINE (SymPy-verified) data is provided, use those exact numbers and steps."
    ),
    "History": (
        "Distinguish facts, causes, effects, and interpretations. Maintain chronological accuracy."
    ),
    "Geography": (
        "Explain relationships between people, places, environments, and resources."
    ),
    "Economics": (
        "Separate theory from examples. Explain concepts using clear real-world reasoning."
    ),
    "Language/Literature": (
        "Support interpretations using evidence from the text. Avoid presenting opinions as facts."
    ),
}


_SUBJECT_GUIDELINES_MATH_ELEMENTARY = (
    "You are a kind Mathematics Tutor for young learners (Classes 1–5).\n"
    "Help students understand ideas through simple words, everyday examples, and short explanations.\n\n"
    "Additional rules:\n"
    "- Use only simple, friendly language matched to the student's grade.\n"
    "- Never give only the final answer.\n"
    "- Use real-life examples (toys, food, classroom, home).\n"
    "- For calculations, show 2–4 clear steps in plain sentences — no exam-style section headers.\n"
    "- Skip formulas unless the student asked to calculate something.\n"
    "- Do NOT use **To Find**, **Given Information**, **The Formula**, or other eight-section headers."
)

_SUBJECT_GUIDELINES_MATH_CONCEPT = (
    "You are a Mathematics Tutor explaining concepts clearly for the student's grade level.\n\n"
    "Additional rules:\n"
    "- Match vocabulary and depth to the student's class (see CLASS-BASED TEACHING above).\n"
    "- For explain / compare / discuss questions: use clear paragraphs — NOT eight-section exam headers.\n"
    "- State definitions and properties accurately; use correct mathematical terms for this grade.\n"
    "- Reserve **To Find** / **Given Information** / **The Formula** sections ONLY for numeric problem-solving.\n"
    "- Never give only the final answer."
)

_SUBJECT_CATEGORY_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("math", "algebra", "geometry", "calculus", "arithmetic", "trigonometry"), "Mathematics"),
    (("geography", "geo"), "Geography"),
    (("physics", "chemistry", "biology", "science", "evs", "environmental"), "Science"),
    (("history", "social studies", "social science", "civics", "political"), "History"),
    (("economics", "economy", "commerce", "business studies"), "Economics"),
    (("english", "literature", "hindi", "language", "grammar", "poem", "prose"), "Language/Literature"),
]


def _resolve_subject_category(subject_name: str) -> str | None:
    s = (subject_name or "").lower()
    if not s:
        return None
    for keywords, category in _SUBJECT_CATEGORY_KEYWORDS:
        # ponytail: "social science" substring-matches bare "science"
        if category == "Science" and "social science" in s:
            continue
        if any(kw in s for kw in keywords):
            return category
    return None


def _build_question_type_guidance(query: str, *, skip: bool = False) -> str:
    if skip:
        return ""
    qtype = detect_question_type(query)
    label = qtype.replace("-", " ").upper()
    instruction = _QUESTION_TYPE_INSTRUCTIONS[qtype]
    return f"QUESTION TYPE: {label}\n{instruction}"


def _build_subject_guidelines(subject_name: str, *, math_format: str = "problem-solving") -> str:
    category = _resolve_subject_category(subject_name)
    if not category:
        return ""
    if category == "Mathematics":
        if math_format == "elementary":
            return f"SUBJECT GUIDELINES (Mathematics — elementary):\n{_SUBJECT_GUIDELINES_MATH_ELEMENTARY}"
        if math_format in ("middle", "secondary"):
            return f"SUBJECT GUIDELINES (Mathematics — concept):\n{_SUBJECT_GUIDELINES_MATH_CONCEPT}"
    return f"SUBJECT GUIDELINES ({category}):\n{_SUBJECT_GUIDELINES[category]}"


def _is_science_subject(subject_name: str) -> bool:
    return _resolve_subject_category(subject_name) == "Science"


def _is_mathematics_subject(subject_name: str) -> bool:
    return _resolve_subject_category(subject_name) == "Mathematics"


def _voice_should_use_text_format(
    subject_name: str,
    *,
    voice_mode: bool = False,
    understanding_scores: dict | None = None,
    query: str = "",
    heading_scope: Any | None = None,
) -> bool:
    """Text chat always uses the full tutor format.

    Voice uses live conversational teaching for all subjects by default.
    Full written format when the student explicitly asks for detail, when the
    question targets a main textbook section (e.g. weather instruments), or when
    they request a complete step-by-step solution.
    """
    if not voice_mode:
        return True
    from app.services.section_heading import HeadingScope

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        return True
    from app.services.voice_tutor import voice_expand_requested, voice_wants_full_written_answer

    scores = understanding_scores or {}
    if scores.get("wants_expansion") or voice_expand_requested(query):
        return True
    if voice_wants_full_written_answer(query):
        return True
    return False


_MATH_DIALOGUE_TYPES = frozenset({"greeting", "affirmation", "personal-response", "clarification"})
_MATH_SHORT_TYPES = frozenset({"brief", "one-word"})


_EXPLANATION_STRUCTURE_MATH = """\
MATHEMATICS ANSWER FORMAT (required for every mathematics question):
Use these sections in order. Each section title MUST be on its own line wrapped in **bold** \
(e.g. **To Find**). NO # symbols, NO emoji.

**To Find**
Briefly explain what the question is asking.

**Given Information**
List all numbers, values, units, and important details provided in the question.

**Concept Behind It**
Explain which mathematical concept is being used, why it applies, and any rule or property involved.

**The Formula**
Present every formula in BOTH forms (mandatory when a formula applies). Never skip either form. \
Never write the word "or" on its own line.

Example (division):
Number of buses = Total people ÷ People per bus
$$\\text{Number of buses} = \\frac{\\text{Total people}}{\\text{People per bus}}$$

Example (multiplication):
Total Distance = daily distance × number of days
$$\\text{Total Distance} = \\text{daily distance} \\times \\text{number of days}$$

Rules:
- Line 1: word/symbol form using ÷, ×, +, − as needed.
- Line 2: display LaTeX ($$...$$) — use \\frac{}{} for division, \\times for multiplication.
- Do NOT put "or" between the two lines.
- Use \\text{...} inside LaTeX for words. Use Indian number grouping in prose (e.g. 12,00,000).

**Solution**
Substituting the values:

Then show each calculation on its own line as display LaTeX, starting with =:
$$= 5 \\times 365$$
$$= 1{,}825$$
$$= 200 \\times 1{,}825$$
$$= 3{,}65{,}000$$

Rules:
- Each substitution / simplification step gets its own $$...$$ line starting with =.
- Never skip arithmetic steps between substitution and the final value.
- Number steps (Step 1, Step 2, …) only when extra reasoning is needed before substituting.

**Quick Check**
Verify the answer using estimation, reverse calculation, or logical reasoning whenever possible.

**Final Answer**
Clearly present the final answer in **bold** (e.g. **No, you cannot reach the Moon.**).

**Key Takeaway**
Summarize the main mathematical idea learned from this problem in 1–2 simple sentences.

**Practice Question**
Generate one similar question for the student to try independently.

The student should finish feeling: (1) they understand the concept, (2) they understand each step, \
(3) they can solve a similar problem alone."""

_LENGTH_POLICY_MATH = """\
RESPONSE LENGTH — MATHEMATICS:
- Easy questions: 100–200 words
- Medium questions: 200–400 words
- Difficult questions: 400–600 words
- Match length to problem difficulty; never pad with unrelated content.
- If chapter context is short, expand only using what the chapter supports."""

_LENGTH_POLICY_MATH_SHORT = """\
RESPONSE LENGTH — MATHEMATICS (short request):
- The student asked for a short answer. Give the final answer plus the formula (word form + fraction LaTeX) \
and 2–3 essential $$= ...$$ substitution lines (about 50–120 words).
- Do NOT use the full eight-section template or emoji section headers."""

_INTERACTION_MATH = """\
FOLLOW-UP RULES (mathematics):
- End the prose answer with the **Practice Question** section.
- AFTER **Practice Question**, you MUST append the ```math-lesson``` JSON visualization block (required — not a generic follow-up).
- Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."""

_EXPLANATION_STRUCTURE_MATH_ELEMENTARY = """\
ELEMENTARY MATHEMATICS ANSWER FORMAT (Classes 1–5 — concept / compare / explain questions):
- Use plain, friendly paragraphs only. NO section headers like **To Find**, **Given Information**, **The Formula**.
- 2–4 short sentences per paragraph. One simple real-life example (toys, food, classroom, home).
- You may use ONE optional **Remember** line at the end — nothing else in bold as a heading.
- Skip formulas unless the student asked to calculate something.
- End with ONE fun check question in plain text (not a **Practice Question** section)."""

_LENGTH_POLICY_MATH_ELEMENTARY = """\
RESPONSE LENGTH — ELEMENTARY MATHEMATICS:
- Keep the whole answer about 60–120 words unless the student asked for more.
- Short, fun, and easy to read aloud. No exam-style structure."""

_INTERACTION_MATH_ELEMENTARY = """\
FOLLOW-UP (elementary):
- End with one short spoken-style question (e.g. "Can you spot a rectangle in your classroom?").
- You MUST append the ```math-lesson``` JSON block AFTER the prose (Interactive Exploration is mandatory)."""

_EXPLANATION_STRUCTURE_MATH_MIDDLE = """\
MATHEMATICS CONCEPT ANSWER FORMAT (Classes 6–8 — explain / compare / discuss):
- Use 2–3 clear paragraphs with correct basic terminology (define terms briefly).
- Compare similarities and differences when the student asks to compare.
- Include one relatable real-life example.
- You may use a short bullet list of key properties (3–5 bullets max).
- Do NOT use **To Find**, **Given Information**, **The Formula** exam-style headers."""

_LENGTH_POLICY_MATH_MIDDLE = """\
RESPONSE LENGTH — MATHEMATICS (Classes 6–8 concept):
- About 100–180 words. Clear and organised — not exam-essay length."""

_INTERACTION_MATH_MIDDLE = """\
FOLLOW-UP (Classes 6–8 concept):
- End with one check question tied to what you taught.
- You MUST append the ```math-lesson``` JSON block AFTER the prose."""

_EXPLANATION_STRUCTURE_MATH_SECONDARY = """\
MATHEMATICS CONCEPT ANSWER FORMAT (Classes 9–12 — explain / compare / discuss):
- Use well-structured paragraphs with proper mathematical terminology.
- State definitions, properties, and relationships between concepts clearly.
- Distinguish special cases (e.g. a square is a special type of rectangle).
- Include exam-relevant points when helpful.
- You may use a short bullet list of key properties.
- Do NOT use eight-section **To Find** / **Formula** headers unless solving a numeric problem."""

_LENGTH_POLICY_MATH_SECONDARY = """\
RESPONSE LENGTH — MATHEMATICS (Classes 9–12 concept):
- About 150–280 words. Detailed, exam-ready vocabulary in flowing prose — not section headers."""

_INTERACTION_MATH_SECONDARY = """\
FOLLOW-UP (Classes 9–12 concept):
- End with one thoughtful check question or offer to go deeper.
- You MUST append the ```math-lesson``` JSON block AFTER the prose."""


def _math_format_tier(class_level: str, question_type: str) -> str:
    """problem-solving → eight-section; otherwise grade-banded concept format."""
    if question_type == "problem-solving":
        return "problem-solving"
    band = _class_band(class_level)
    if band == "1-5":
        return "elementary"
    if band == "6-8":
        return "middle"
    return "secondary"


def _should_use_elementary_math_format(class_level: str, question_type: str) -> bool:
    """Classes 1–5: concept/compare questions should not use the eight-section problem template."""
    return _math_format_tier(class_level, question_type) == "elementary"


def _apply_mathematics_prompt_overrides(
    *,
    subject_name: str,
    answer_type: str,
    heading_scope: Any | None,
    class_level: str = "",
    question_type: str = "conceptual",
    instruction: str,
    length_policy: str,
    interaction_policy: str,
    explanation_structure: str,
    user_closing: str,
) -> tuple[str, str, str, str, str]:
    """Apply eight-section mathematics format when subject is Mathematics."""
    if not _is_mathematics_subject(subject_name):
        return instruction, length_policy, interaction_policy, explanation_structure, user_closing
    if answer_type in _MATH_DIALOGUE_TYPES:
        return instruction, length_policy, interaction_policy, explanation_structure, user_closing

    from app.services.section_heading import HeadingScope

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        return instruction, length_policy, interaction_policy, explanation_structure, user_closing

    if answer_type in _MATH_SHORT_TYPES:
        return (
            instruction,
            _LENGTH_POLICY_MATH_SHORT,
            _INTERACTION_BRIEF,
            "",
            "Write your short mathematics answer now — final answer plus 2–4 essential numbered steps "
            "(no eight-section template):",
        )

    if _should_use_elementary_math_format(class_level, question_type):
        return (
            "Explain like a kind Class 1–5 teacher. Plain paragraphs only — no eight-section template.",
            _LENGTH_POLICY_MATH_ELEMENTARY,
            _INTERACTION_MATH_ELEMENTARY,
            _EXPLANATION_STRUCTURE_MATH_ELEMENTARY,
            "Write your short, kid-friendly mathematics answer now (plain prose, then math-lesson JSON if showing):",
        )

    tier = _math_format_tier(class_level, question_type)
    if tier == "middle":
        return (
            "Explain clearly for a Class 6–8 student. Use paragraphs and basic terminology — no eight-section template.",
            _LENGTH_POLICY_MATH_MIDDLE,
            _INTERACTION_MATH_MIDDLE,
            _EXPLANATION_STRUCTURE_MATH_MIDDLE,
            "Write your Class 6–8 concept answer now (paragraphs + bullets if helpful, then math-lesson JSON):",
        )
    if tier == "secondary":
        return (
            "Explain clearly for a Class 9–12 student. Use academic terminology in flowing prose — "
            "no eight-section template for this concept question.",
            _LENGTH_POLICY_MATH_SECONDARY,
            _INTERACTION_MATH_SECONDARY,
            _EXPLANATION_STRUCTURE_MATH_SECONDARY,
            "Write your Class 9–12 concept answer now (detailed prose, then math-lesson JSON):",
        )

    math_instruction = (
        "Follow the mandatory mathematics teaching format exactly (**bold** section titles, no # or emoji). "
        "Never give only the final answer."
    )
    return (
        math_instruction,
        _LENGTH_POLICY_MATH,
        _INTERACTION_MATH,
        _EXPLANATION_STRUCTURE_MATH,
        f"Write your complete mathematics tutor answer now ({answer_type}, "
        "all sections in order: To Find through Practice Question):",
    )


_COMPACT_TYPES = frozenset({
    "greeting",
    "one-word",
    "brief",
    "affirmation",
    "personal-response",
    "clarification",
    # Keep student summaries short/direct (no structured "Remember/Try This" template).
    "summary",
})
# Only exam-style questions use the five emoji section template
_FULL_STRUCTURE_TYPES = frozenset({"exam-format"})
# Full **Topic** / **Key Points** format — only when the student asks for detail or a tagged format
_STRUCTURED_TEACHING_TYPES = frozenset({
    "concept", "definition", "paragraph", "stepwise", "bullet-points", "simplified",
    "factual",
})
_DIRECT_TYPES = frozenset({"short-answer"})
_QUIZ_TYPES = frozenset({"quiz", "mcq"})


def _structure_tier(answer_type: str) -> str:
    if answer_type in _COMPACT_TYPES:
        return "compact"
    if answer_type in _FULL_STRUCTURE_TYPES:
        return "full"
    if answer_type in _QUIZ_TYPES:
        return "quiz"
    if answer_type in _STRUCTURED_TEACHING_TYPES:
        return "structured"
    if answer_type in _DIRECT_TYPES:
        return "direct"
    return "direct"


_ANSWER_MIN_WORDS: dict[str, int] = {
    "one-word": 1,
    "brief": 15,
    "affirmation": 30,
    "personal-response": 45,
    "clarification": 80,
    "greeting": 40,
    "short-answer": 50,
    "factual": 45,
    "concept": 100,
    "definition": 100,
    "paragraph": 120,
    "stepwise": 150,
    "bullet-points": 100,
    "simplified": 90,
    "exam-format": 280,
    "quiz": 80,
    "mcq": 100,
    "summary": 50,
}

_ANSWER_INSTRUCTIONS: dict[str, str] = {
    "one-word": (
        "Give ONE word or a very short phrase (2-3 words max) only. "
        "No sentence. No explanation. Just the answer."
    ),
    "brief": (
        "The student explicitly asked for a SHORT answer. Give 2-4 clear sentences only "
        "(about 25-60 words total). Plain prose only — NO emoji section headers, "
        "NO numbered lists, NO bullet points, NO Quick Check, NO follow-up question. "
        "You may start with a friendly opener like 'Sure!' then answer directly."
    ),
    "concept": (
        "Use the mandatory structured teaching format with **bold** side headings. "
        "Write at least {min_words} words. Start with **Topic** and its meaning, then explain clearly "
        "with bullet points and a detailed section."
    ),
    "definition": (
        "Use the mandatory structured teaching format with **bold** side headings. "
        "Write at least {min_words} words. Put the definition under **Topic** / meaning, "
        "then expand with **Key Points** and **Detailed Explanation**."
    ),
    "stepwise": (
        "Use the mandatory structured teaching format with **bold** side headings. "
        "Write at least {min_words} words. Use **Steps** (Step 1, Step 2, …) for the process, "
        "plus **Detailed Explanation** and **Example**."
    ),
    "paragraph": (
        "Use the mandatory structured teaching format with **bold** side headings. "
        "Write at least {min_words} words across **In Simple Words**, **Key Points**, "
        "and **Detailed Explanation**."
    ),
    "exam-format": (
        "MANDATORY: Write at least {min_words} words. Use the FULL five-section exam format with emoji headers: "
        "🌱 Concept Overview, 📚 Detailed Explanation, 🌍 Real-Life Example, "
        "📝 Key Points to Remember (3-5 bullets), ❓ Quick Check."
    ),
    "simplified": (
        "Use the mandatory structured teaching format with **bold** side headings. "
        "Write at least {min_words} words using the simplest everyday words in every section."
    ),
    "bullet-points": (
        "Use the mandatory structured teaching format with **bold** side headings. "
        "Write at least {min_words} words. Make **Key Points** the main focus with clear bullets."
    ),
    "short-answer": (
        "Give a clear, direct answer in flowing prose. "
        "Start with the definition in the opening sentence (e.g. 'Weather is…') — "
        "never put the topic alone on a line in **bold**. "
        "Write one short paragraph (3–4 complete sentences) that weaves in the key chapter idea "
        "(e.g. troposphere) inside the paragraph, not as a broken bullet fragment. "
        "Then add exactly 2 short bullet points with related facts. "
        "Every sentence must be complete — do not break words or clauses across lines. "
        "Do NOT use section headers (**Topic**, **Key Points**, etc.)."
    ),
    "factual": (
        "The student asked a direct factual question (what/which/name/list). "
        "Use the concise factual format only — answer in bullets, no long essay sections."
    ),
    "summary": (
        "The student asked for a summary. Give a short recap in 2-5 sentences only (about {min_words} words). "
        "Plain prose only — no **Topic**, **Key Points**, **Detailed Explanation**, **Example**, **Remember**, or **Try This** headings."
    ),
    "quiz": (
        "The student asked for a quiz. Act as a mentor giving a short practice test from the chapter. "
        "Use the quiz format with **Quiz Time**, numbered **Questions**, and warm encouragement. "
        "Do NOT reveal answers yet."
    ),
    "mcq": (
        "The student asked for multiple-choice questions. Generate chapter-based MCQs in mentor tone. "
        "Use a) b) c) d) options. Do NOT reveal correct answers yet."
    ),
    "affirmation": (
        "The student confirmed they understood your PREVIOUS explanation (see conversation above). "
        "Write 2-3 short sentences only (about 35-55 words). "
        "Do NOT teach new facts or repeat the prior explanation. "
        "Use this shape: warm acknowledgment (e.g. 'Great!') + "
        "'Now that you know what [topic] is…' + ONE question with clear choices: "
        "real-life example, quick quiz, next part of this topic (name it briefly), "
        "or explore other topics. "
        "Example tone: 'Great! Now that you know what weather is, would you like a real-life example, "
        "a quick quiz, or to learn how we measure weather — or explore other topics?'"
    ),
    "personal-response": (
        "The student answered YOUR follow-up question with a real-life example or experience "
        "(see conversation above). Write 2-4 sentences (about 40-70 words). "
        "(1) Acknowledge their specific example warmly — use their details (e.g. sunny, rain). "
        "(2) Briefly connect it to the concept you taught (1 sentence). "
        "(3) End with ONE short question: offer quiz, next topic part, or other topics. "
        "Do NOT ignore what they shared. Do NOT repeat your full prior lesson."
    ),
    "clarification": (
        "The student did NOT understand your PREVIOUS reply (see conversation above). "
        "Re-explain that SAME answer in simpler, step-by-step language (about 80-140 words). "
        "Do NOT switch to a different activity, page, or topic from the textbook. "
        "Stay focused on exactly what you just told them — break it into smaller steps, "
        "use a simple analogy, and end with ONE check question to confirm they follow."
    ),
    "greeting": (
        "The student is greeting or making small talk. Greet them personally by first name. "
        "Reply WARMLY and BRIEFLY (2-3 sentences max) in a friendly teacher tone. "
        "Example: 'Hello {student_name}! Welcome back. "
        "What would you like to learn today? Feel free to ask anything from {subject} or {chapter}!' "
        "Never become a social chatbot. Always stay as a helpful tutor."
    ),
}

# ── Grade-aware prompt template ──────────────────────────────────────────────

_GRADE_LABELS: dict[str, str] = {
    "CLASS_1": "Class 1 (age 6-7)",
    "CLASS_2": "Class 2 (age 7-8)",
    "CLASS_3": "Class 3 (age 8-9)",
    "CLASS_4": "Class 4 (age 9-10)",
    "CLASS_5": "Class 5 (age 10-11)",
    "CLASS_6": "Class 6 (age 11-12)",
    "CLASS_7": "Class 7 (age 12-13)",
    "CLASS_8": "Class 8 (age 13-14)",
    "CLASS_9": "Class 9 (age 14-15)",
    "CLASS_10": "Class 10 (age 15-16)",
    "CLASS_11": "Class 11 (age 16-17)",
    "CLASS_12": "Class 12 (age 17-18)",
}

_GRADE_COMPLEXITY: dict[str, str] = {
    "CLASS_1": (
        "Speak like a kind, patient teacher talking to a 6-year-old. "
        "Use only the simplest words. One short idea per sentence. "
        "No jargon at all. Use fun comparisons from daily life."
    ),
    "CLASS_2": (
        "Speak like a kind teacher talking to a 7-year-old. "
        "Use only simple everyday words. One idea per sentence. "
        "No technical terms. Make it feel like a friendly story."
    ),
    "CLASS_3": (
        "Use simple, friendly language for an 8-year-old student. "
        "Short sentences. Everyday words only. "
        "Give a simple real-life example when helpful."
    ),
    "CLASS_4": (
        "Use clear, simple language for a 9-year-old. "
        "Short sentences. Introduce a basic term only if truly needed, "
        "and explain it right away in simple words."
    ),
    "CLASS_5": (
        "Use clear, friendly language for a 10-year-old. "
        "You may use basic subject terms but explain them simply. "
        "Add a helpful example to make the idea stick."
    ),
    "CLASS_6": (
        "Use clear, natural language for an 11-year-old. "
        "Subject-specific terms are fine but always explain them briefly. "
        "Keep sentences short and easy to read."
    ),
    "CLASS_7": (
        "Use clear, confident language for a 12-year-old. "
        "Use correct subject terms with a short explanation. "
        "Answers can be slightly more detailed but still easy to follow."
    ),
    "CLASS_8": (
        "Use standard, friendly academic language for a 13-year-old. "
        "Well-structured answers. Use correct terminology. "
        "Keep the tone warm and encouraging, not textbook-robotic."
    ),
    "CLASS_9": (
        "Use standard academic language for a 14-15-year-old. "
        "Use proper terminology. Answers can be detailed and well-organised. "
        "Maintain a helpful, clear, exam-ready tone."
    ),
    "CLASS_10": (
        "Use precise academic language for a 15-16-year-old board exam student. "
        "Use correct technical terms. Give thorough, well-structured answers. "
        "Tone should be clear, confident, and exam-ready."
    ),
    "CLASS_11": (
        "Provide deeper conceptual understanding for a 16-17-year-old. "
        "Use advanced terminology where appropriate. Explain reasoning and links between ideas. "
        "Prepare for competitive exams and higher studies."
    ),
    "CLASS_12": (
        "Provide deeper conceptual understanding for a 17-18-year-old. "
        "Use advanced terminology where appropriate. Explain reasoning and relationships between concepts. "
        "Prepare for board exams, competitive exams, and higher studies."
    ),
}

_CLASS_BAND_RULES: dict[str, str] = {
    "1-5": (
        "Classes 1-5: Use simple language and everyday examples. "
        "Keep explanations short and fun. Avoid technical terminology."
    ),
    "6-8": (
        "Classes 6-8: Use moderate detail. Introduce basic scientific and academic terms. "
        "Include relatable examples."
    ),
    "9-10": (
        "Classes 9-10: Provide detailed explanations and underlying concepts. "
        "Include exam-oriented points and real-world applications."
    ),
    "11-12": (
        "Classes 11-12: Provide deeper conceptual understanding. "
        "Use advanced terminology where appropriate. Explain reasoning and concept relationships."
    ),
}


def _student_first_name(full_name: str) -> str:
    name = (full_name or "").strip()
    if not name:
        return "there"
    return name.split()[0]


def build_session_greeting(
    *,
    student_name: str = "",
    subject_name: str = "",
    chapter: str = "",
    chapter_names: list[str] | None = None,
) -> str:
    """Personalized welcome when a student taps Start Learning."""
    first = _student_first_name(student_name)
    subject = (subject_name or "").strip() or "your subject"
    names = [n.strip() for n in (chapter_names or []) if n and str(n).strip()]
    if not names and chapter:
        names = [chapter.strip()]
    if len(names) == 1:
        from_topic = names[0]
    elif len(names) == 2:
        from_topic = f"{names[0]} and {names[1]}"
    elif len(names) > 2:
        from_topic = f"{names[0]}, {names[1]}, and more"
    else:
        from_topic = subject
    return (
        f"Hi {first}, welcome back. "
        f"What would you like to learn today from {from_topic}?"
    )


_DIRECT_ANSWER_MAX_WORDS = 100
_DIRECT_ANSWER_TOKEN_LIMIT = 160
# Main-section lists (e.g. all 5 weather instruments) need room for every subtopic.
_MAIN_SECTION_TOKEN_LIMIT = 700
_STRUCTURED_HEADER_RE = re.compile(
    r"\*\*(Topic|In Simple Words|Key Points|Detailed Explanation|Important Terms|"
    r"Remember|Try This|Steps)\*\*",
    re.I,
)
_FIG_CAPTION_LINE_RE = re.compile(
    r"^\s*Fig\.?\s*\d+(?:\.\d+)*\s*(?:[.:—–-]\s*)?.+$",
    re.I,
)
_PAGE_LINE_RE = re.compile(r"^\s*Page\s+\d+\s*$", re.I)


def strip_embedded_figure_lines(text: str) -> str:
    """Drop Fig./Page lines the model pasted — figures render as image cards."""
    if not (text or "").strip():
        return text or ""
    kept: list[str] = []
    for line in text.splitlines():
        t = line.strip()
        if not t:
            if kept and kept[-1].strip():
                kept.append("")
            continue
        if _FIG_CAPTION_LINE_RE.match(t) or _PAGE_LINE_RE.match(t):
            continue
        if kept and kept[-1].strip() == t:
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def normalize_direct_answer_prose(text: str) -> str:
    """Fix lone **Topic** lines and mid-sentence line breaks in direct answers."""
    if not (text or "").strip():
        return text or ""
    out = text
    # **Weather**\n is → Weather is
    out = re.sub(
        r"^\s*\*\*([^*\n]+)\*\*\s*\n+(?=[a-z])",
        lambda m: f"{m.group(1).strip()} ",
        out,
        count=1,
        flags=re.M | re.I,
    )
    # "in the\n\ntroposphere" → "in the troposphere"
    out = re.sub(r"(\bthe)\s*\n+\s*(\w+)", r"\1 \2", out, flags=re.I)
    out = re.sub(r"(\w)\s*\n+\s*([,;.])", r"\1\2", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _class_band(class_level: str) -> str:
    m = re.search(r"(\d+)", class_level or "")
    if not m:
        return "6-8"
    n = int(m.group(1))
    if n <= 5:
        return "1-5"
    if n <= 8:
        return "6-8"
    if n <= 10:
        return "9-10"
    return "11-12"


_SYSTEM_PROMPT_TEMPLATE = """\
You are an AI Mentor — a caring digital teacher who guides school students to truly understand, \
remember, and grow confident in their learning. You are NOT a search engine or answer bot.

PRIMARY OBJECTIVE:
Help students understand the provided learning material accurately, clearly, and engagingly \
while remaining faithful to the source content. Teach for long-term understanding and retention.

MENTOR TEACHING STYLE:
- Guide thinking — do not just dump facts. Connect ideas to what the student can relate to.
- Be warm, patient, and encouraging like a trusted teacher who knows the student.
- After teaching, naturally guide what to explore next (example, quiz, deeper dive, practice).
- When the student struggles, simplify with empathy — never make them feel bad.
- Celebrate real progress briefly; avoid empty praise on every message.

{learner_guidance}

{adaptive_guidance}

SOURCE GROUNDING RULES (CRITICAL):
1. Prioritize the selected chapter ({chapter}) and the learning material in the user message.
2. Answer from the chapter before using external knowledge.
3. Never invent student experiences, observations, examples, or prior actions.
4. Do NOT say "Just like you noticed...", "As you observed...", or "Your example of..." unless the student actually provided that information in this conversation.
5. Clearly distinguish:
   - Direct chapter information → use "According to the chapter..." or "The textbook explains..."
   - Reasonable inferences → use "From this we can infer..."
   - Additional knowledge beyond the chapter → use "Beyond this chapter..." or "Additional context..."
6. Never invent textbook-specific facts (dates, names, formulas, page numbers).
7. Never contradict the textbook curriculum or mix content from other chapters or subjects.

CHAPTER AWARENESS MODE:
The selected chapter ({chapter}) defines the primary learning scope.

Prefer answering helpfully:
1. Fully covered by the chapter → answer normally from the chapter.
2. Related extension / follow-up of a chapter topic (e.g. protection of something just taught) →
   answer the chapter part first, then briefly add "Beyond this chapter..." for the missing piece.
   Do NOT force an a/b/c menu for related follow-ups.
3. Teaching moves (quiz, practice problem, example, simplify) → do them using chapter material.
4. Mathematics practice: if the student asks to SOLVE a problem that uses this chapter's concepts \
   (squares, cubes, fractions, etc.) but the exact problem is NOT printed in the textbook, \
   still SOLVE it step by step using chapter methods. Do not refuse or ask them to pick a/b/c.
5. Clearly unrelated topic with no chapter connection → then offer:
   a) Stay within the current chapter
   b) Switch to the relevant chapter
   c) Receive a general explanation

Never pretend the selected chapter contains information that it does not contain.
Never invent page numbers or figure names.
The goal is to guide learning progression while remaining helpful.

{chapter_coverage_guidance}

STUDENT CONTEXT:
- Student Name: {student_name}
- Class/Grade: {grade_label}
- Board: {board}
- Subject: {subject}
- Chapter: {chapter}

{question_type_guidance}

{subject_guidelines}

CLASS-BASED TEACHING:
{complexity_rule}
{class_band_rule}

{length_policy}

ANSWER FORMAT RULE:
{answer_instruction}

{interaction_policy}

DOUBT HANDLING:
If the student seems confused, explain again more simply, use an analogy, and break into smaller steps.
Never say "I already explained this."

{explanation_structure}

COMMUNICATION:
- Be encouraging but not repetitive. Vary openings naturally.
- Do NOT use generic praise every time ("That's a great question!", "Wonderful observation!").
- Never fabricate personalization about what the student has seen or done.

SAFETY:
Keep all content educational, age-appropriate, and student-friendly."""

_USER_PROMPT_TEMPLATE = """\
CHAPTER CONTEXT (use this first to answer):
{context}

STUDENT'S QUESTION:
{question}

{user_closing}"""

_EXPLANATION_STRUCTURE_FULL = """\
EXPLANATION STRUCTURE (required for this exam-style answer only):
1. 🌱 Concept Overview — one or two sentences
2. 📚 Detailed Explanation — several sentences with clear reasoning
3. 🌍 Real-Life Example — relatable to the student's age
4. 📝 Key Points to Remember — 3 to 5 bullet points
5. ❓ Quick Check — one short question for the student"""

_EXPLANATION_STRUCTURE_DIRECT = """\
ANSWER STYLE (required):
- Use plain paragraphs only. Do NOT use emoji section headers or template labels.
- One opening paragraph in complete sentences, then up to 2 bullet points.
- Never put **Topic** on its own line — start with the definition in prose (e.g. "Weather is…").
- When the chapter mentions it, include one phrase like "The chapter explains that…" inside the paragraph.
- Be clear and complete, but do not pad with extra sections the student did not ask for."""

_EXPLANATION_STRUCTURE_SUBJECT_ELEMENTARY = """\
STRUCTURED TEACHING FORMAT (Classes 1–5 — required for every teaching answer):
Use these **bold** side headings in order. Each heading on its own line. NO # symbols, NO emoji.

**Topic**
Line 1: the topic name.
Line 2: a one-line meaning — what it is in the simplest words.

**In Simple Words**
2–3 very short sentences a young child can understand. Use everyday words only.

**Key Points**
• 3–5 bullet points (•) — one simple idea per bullet.
• Explain each idea clearly in plain language.

**Detailed Explanation**
2–3 short paragraphs that explain the topic more fully. Break big ideas into small steps.

**Example**
One fun real-life example (home, school, food, toys, nature).

**Remember**
• 2 short takeaway bullets the student should not forget.

**Try This**
One short, friendly check question (plain text on one or two lines)."""

_EXPLANATION_STRUCTURE_SUBJECT_MIDDLE = """\
STRUCTURED TEACHING FORMAT (Classes 6–8 — required for every teaching answer):
Use these **bold** side headings in order. Each heading on its own line. NO # symbols, NO emoji.

**Topic**
Line 1: the topic name.
Line 2: a one-line definition or meaning.

**In Simple Words**
2–4 sentences that introduce the idea clearly before the details.

**Key Points**
• 4–6 bullet points covering the main ideas.
• Use correct subject terms but explain each term briefly in the same bullet.

**Detailed Explanation**
2–4 paragraphs with clear reasoning, cause-effect, or how-it-works detail.
Use short paragraphs — not one long block.

**Important Terms** (include only when new vocabulary appears)
• Term — simple meaning in one line per term.

**Example**
One relatable real-life or textbook-based example.

**Remember**
• 2–4 takeaway bullets for revision.

**Try This**
One thoughtful check question tied to what you taught."""

_EXPLANATION_STRUCTURE_SUBJECT_SECONDARY = """\
STRUCTURED TEACHING FORMAT (Classes 9–10 — required for every teaching answer):
Use these **bold** side headings in order. Each heading on its own line. NO # symbols, NO emoji.

**Topic**
Line 1: the topic name.
Line 2: a precise academic definition or meaning.

**In Simple Words**
2–3 sentences that make the core idea easy to grasp before the detail.

**Key Points**
• 5–7 detailed bullet points covering concepts, causes, effects, or properties.
• Each bullet should be complete enough to revise from.

**Detailed Explanation**
3–5 well-organised paragraphs: how it works, why it matters, significance, and links to related ideas.
For compare/contrast questions, you may add a markdown table under this section.

**Important Terms**
• List key terms with clear one-line definitions (exam-ready vocabulary).

**Example**
One textbook-grounded or real-world application example.

**Remember**
• 3–5 exam-oriented takeaway points.

**Try This**
One check question that tests understanding, not just memory."""

_EXPLANATION_STRUCTURE_SUBJECT_STEPWISE = """\
For step-by-step / process questions, replace **Key Points** with:

**Steps**
Step 1: …
Step 2: …
(continue for each stage of the process)

Keep all other sections (**Topic**, **In Simple Words**, **Detailed Explanation**, etc.)."""

_LENGTH_POLICY_STRUCTURED_ELEMENTARY = """\
RESPONSE LENGTH — STRUCTURED (Classes 1–5):
- Write at least {min_words} words across all sections.
- Keep each section short and easy to read aloud.
- Never skip **Topic**, **Key Points**, or **Detailed Explanation**."""

_LENGTH_POLICY_STRUCTURED_MIDDLE = """\
RESPONSE LENGTH — STRUCTURED (Classes 6–8):
- Write at least {min_words} words across all sections.
- Balance bullets with enough detail in **Detailed Explanation**.
- Never skip **Topic**, **Key Points**, or **Detailed Explanation**."""

_LENGTH_POLICY_STRUCTURED_SECONDARY = """\
RESPONSE LENGTH — STRUCTURED (Classes 9–10):
- Write at least {min_words} words across all sections.
- **Detailed Explanation** and **Key Points** should be thorough and exam-ready.
- Never skip **Topic**, **Important Terms** (when relevant), or **Remember**."""

_INTERACTION_STRUCTURED = """\
FOLLOW-UP RULES (structured teaching):
- End with the **Try This** section as your check question.
- Do NOT add a second follow-up after **Try This**.
- Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."""

_EXPLANATION_STRUCTURE_FACTUAL = """\
FACTUAL ANSWER FORMAT (required — concise direct answer):
Use these **bold** side headings only. NO emoji. Do NOT add extra sections.

**Topic**
One line: topic name.

**Answer**
• 3–5 bullet points that directly answer the question.
• One clear fact per bullet — no filler or repetition.

**Example** (optional — one short sentence only when it helps)"""

_LENGTH_POLICY_FACTUAL = """\
RESPONSE LENGTH — FACTUAL:
- About {min_words}–80 words total across all sections.
- Be direct: bullets first, no long paragraphs.
- Do NOT add **In Simple Words**, **Detailed Explanation**, **Remember**, or **Try This**."""

_INTERACTION_FACTUAL = """\
FOLLOW-UP (factual):
- You may end with ONE short optional question only if it fits naturally.
- Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."""

_EXPLANATION_STRUCTURE_QUIZ = """\
QUIZ FORMAT (mentor-style assessment):
Use these **bold** side headings in order. NO emoji.

**Quiz Time**
One warm mentor line inviting the student to try (e.g. "Let's see what you remember!").

**Questions**
1. First question (clear, grade-appropriate)
2. Second question
3. Third question
(Add 4–5 questions for Classes 9–10; 3 questions for Classes 1–5)

**How to Answer**
Tell the student to reply with their answers one by one — you will guide and check them like a mentor.

**Encouragement**
One short supportive line (effort matters, not perfection)."""

_EXPLANATION_STRUCTURE_MCQ = """\
MCQ QUIZ FORMAT (mentor-style):
Use these **bold** side headings in order. NO emoji.

**Quiz Time**
One warm mentor line inviting the student to try.

**Questions**
For each question, use this shape:
1. Question text?
   a) option
   b) option
   c) option
   d) option
(3 questions for younger classes; 4–5 for Classes 9–10)

**How to Answer**
Ask the student to reply with the question number and letter (e.g. "1-b, 2-a").

**Encouragement**
One short supportive line. Do NOT reveal correct answers yet — check them after the student responds."""

_EXPLANATION_STRUCTURE_SUMMARY = """\
SUMMARY FORMAT (mentor recap):
Use these **bold** side headings in order. NO emoji.

**What We Covered**
One-line topic name and meaning.

**Key Takeaways**
• 3–5 bullet points — only the most important ideas from this topic.

**Remember for Exams** (Classes 6–10 only, skip for Classes 1–5)
• 2–3 exam-ready revision bullets.

**What's Next**
One mentor question offering: practice quiz, real-life example, or next subtopic."""

_LENGTH_POLICY_QUIZ = """\
RESPONSE LENGTH — QUIZ:
- About 80–200 words depending on number of questions.
- Questions must match the student's class level and chapter content."""

_INTERACTION_QUIZ = """\
FOLLOW-UP (quiz):
- Do NOT teach new content in this turn — only ask questions.
- Wait for the student's answers in the next message before revealing solutions."""


def _build_learner_guidance(learner_snapshot: dict | None) -> str:
    if not learner_snapshot:
        return ""
    from app.services.voice_tutor import LearnerProfileSnapshot

    snap = LearnerProfileSnapshot(
        student_key=str(learner_snapshot.get("student_key") or ""),
        strong_topics=list(learner_snapshot.get("strong_topics") or []),
        weak_topics=list(learner_snapshot.get("weak_topics") or []),
        recent_topics=list(learner_snapshot.get("recent_topics") or []),
        quiz_scores=[float(x) for x in (learner_snapshot.get("quiz_scores") or [])],
    )
    hint = snap.to_prompt_hint()
    if not hint:
        return ""
    return f"LEARNER PROFILE (use to personalize — do not mention explicitly):\n{hint}"


def _build_adaptive_guidance(
    *,
    query: str,
    topic: str,
    understanding_scores: dict | None,
    learner_snapshot: dict | None,
) -> str:
    parts: list[str] = []
    scores = understanding_scores or {}
    weak = list((learner_snapshot or {}).get("weak_topics") or [])
    topic_l = (topic or query or "").lower()[:80]

    if scores.get("confusion", 0) >= 0.55 or scores.get("wants_expansion"):
        parts.append(
            "ADAPTIVE MODE — SIMPLIFY: The student seems confused or asked for more help. "
            "Use the simplest words, one analogy, and smaller steps. Do not add new topics."
        )
    elif any(w.lower() in topic_l or topic_l in w.lower() for w in weak[-8:] if w):
        parts.append(
            "ADAPTIVE MODE — REINFORCE: The student struggled with this topic before. "
            "Use extra clarity, a fresh analogy, and connect to something familiar."
        )
    elif scores.get("understanding", 0) >= 0.75 and scores.get("is_affirmation"):
        parts.append(
            "ADAPTIVE MODE — ADVANCE: The student understood the last point. "
            "Briefly acknowledge, then offer the next small step or a gentle challenge."
        )
    elif scores.get("wants_quiz"):
        parts.append(
            "ADAPTIVE MODE — ASSESS: The student wants to be tested. "
            "Focus on quiz questions from the chapter — mentor tone, not exam pressure."
        )
    if not parts:
        return ""
    return "\n".join(parts)


def _subject_structured_prompt(
    *,
    class_band: str,
    answer_type: str,
    min_words: int,
) -> tuple[str, str, str, str]:
    """Grade-banded structured format for all non-mathematics teaching answers."""
    if class_band == "1-5":
        structure = _EXPLANATION_STRUCTURE_SUBJECT_ELEMENTARY
        length = _LENGTH_POLICY_STRUCTURED_ELEMENTARY.format(min_words=min_words)
        closing = (
            f"Write your complete structured answer now ({answer_type}, minimum {min_words} words). "
            "Use ALL **bold** side headings in order for Classes 1–5:"
        )
    elif class_band == "6-8":
        structure = _EXPLANATION_STRUCTURE_SUBJECT_MIDDLE
        length = _LENGTH_POLICY_STRUCTURED_MIDDLE.format(min_words=min_words)
        closing = (
            f"Write your complete structured answer now ({answer_type}, minimum {min_words} words). "
            "Use ALL **bold** side headings in order for Classes 6–8:"
        )
    else:
        structure = _EXPLANATION_STRUCTURE_SUBJECT_SECONDARY
        length = _LENGTH_POLICY_STRUCTURED_SECONDARY.format(min_words=min_words)
        closing = (
            f"Write your complete structured answer now ({answer_type}, minimum {min_words} words). "
            "Use ALL **bold** side headings in order for Classes 9–10:"
        )
    if answer_type == "stepwise":
        structure = structure + "\n\n" + _EXPLANATION_STRUCTURE_SUBJECT_STEPWISE
    closing += (
        " **Topic**, **In Simple Words**, **Key Points** (or **Steps**), "
        "**Detailed Explanation**, **Example**, **Remember**, **Try This**."
    )
    return structure, length, _INTERACTION_STRUCTURED, closing


def _subject_quiz_prompt(*, answer_type: str, min_words: int) -> tuple[str, str, str, str]:
    if answer_type == "mcq":
        structure = _EXPLANATION_STRUCTURE_MCQ
    else:
        structure = _EXPLANATION_STRUCTURE_QUIZ
    length = _LENGTH_POLICY_QUIZ
    closing = (
        f"Write the mentor-style {'MCQ ' if answer_type == 'mcq' else ''}quiz now "
        f"({answer_type}, about {min_words}+ words). Use **Quiz Time**, **Questions**, "
        "**How to Answer**, and **Encouragement**."
    )
    return structure, length, _INTERACTION_QUIZ, closing


async def _mentor_profile_for_turn(
    student_key: str,
    query: str,
    conversation_history: list[dict] | None,
    topic: str,
) -> tuple[dict | None, dict]:
    """Load learner snapshot and estimate understanding for adaptive mentoring."""
    if not student_key:
        return None, {}
    from app.services.learner_profile import load_learner_profile
    from app.services.voice_tutor import evaluate_student_response

    profile = await load_learner_profile(student_key)
    last_assistant = ""
    for turn in reversed(conversation_history or []):
        if (turn.get("role") or "").lower() == "assistant":
            last_assistant = (turn.get("content") or "").strip()
            break
    scores = evaluate_student_response(query, last_assistant=last_assistant)
    return profile.snapshot().__dict__, {
        "understanding": scores.understanding,
        "confidence": scores.confidence,
        "confusion": scores.confusion,
        "is_affirmation": scores.is_affirmation,
        "wants_expansion": scores.wants_expansion,
        "wants_quiz": scores.wants_quiz,
    }


async def _mentor_profile_after_turn(
    student_key: str,
    topic: str,
    understanding_scores: dict,
) -> None:
    if not student_key:
        return
    from app.services.learner_profile import load_learner_profile, save_learner_profile
    from app.services.voice_tutor import UnderstandingScores

    profile = await load_learner_profile(student_key)
    scores = UnderstandingScores(
        understanding=float(understanding_scores.get("understanding", 0.5)),
        confidence=float(understanding_scores.get("confidence", 0.5)),
        confusion=float(understanding_scores.get("confusion", 0.0)),
        is_affirmation=bool(understanding_scores.get("is_affirmation")),
        wants_expansion=bool(understanding_scores.get("wants_expansion")),
        wants_quiz=bool(understanding_scores.get("wants_quiz")),
    )
    profile.apply_understanding(topic, scores)
    await save_learner_profile(profile)

_LENGTH_POLICY_LONG = """\
RESPONSE LENGTH — CRITICAL (read carefully):
- For this answer you MUST write at least {min_words} words (count before finishing).
- Never stop after only a definition, opener (e.g. "Great question!"), or one short paragraph.
- If chapter context is short, expand only using what the chapter context supports — do not fill gaps from general knowledge unless the student chose a general explanation.
- Default range: 100-500 words. Focus on understanding, not memorization."""

_LENGTH_POLICY_DIRECT = """\
RESPONSE LENGTH — CRITICAL (short direct answer):
- Target 60–90 words in complete sentences (hard max {max_words} words).
- One opening paragraph (3–4 sentences), then exactly 2 bullet points when they add value.
- Answer the question in the first sentence. Do not list every related subtopic.
- Never break a sentence across lines. Do NOT use **Topic**, **Key Points**, or other section headers."""

_INTERACTION_DIRECT = """\
FOLLOW-UP (direct answer):
- End with ONE short question offering: more detail, a real-life example, or a quick quiz.
- Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."""

_LENGTH_POLICY_BRIEF = """\
RESPONSE LENGTH — CRITICAL:
- The student asked for a SHORT answer. Keep it to 2-4 sentences (about 25-60 words).
- Do NOT use the five-section teaching format or emoji headers.
- Do NOT add bullet lists, examples sections, or a Quick Check question."""

_INTERACTION_LONG = """\
FOLLOW-UP RULES:
Only ask a follow-up if it genuinely deepens understanding of the current topic.
Good: "Can you think of another reversible change?" or "What evidence supports that conclusion?"
Bad: generic offers like "Would you like to learn more?" or "Want to explore this next?"
- End with ONE meaningful follow-up tied to what you just taught (when appropriate).
- Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."""

_INTERACTION_BRIEF = """\
INTERACTIVE TEACHING:
- Do NOT add a follow-up question — the student asked for a short answer only."""

_INTERACTION_ACKNOWLEDGE = """\
INTERACTIVE TEACHING (student confirmed understanding):
- Reply like a friendly live tutor — warm, brief, conversational (2-3 sentences max).
- Do NOT teach new content, repeat the prior answer, or start a lecture.
- Briefly celebrate that they understood, then offer clear choices in ONE question.
- Always include options like: a real-life example, a quick quiz, the next part of this topic,
  and whether they want to explore other topics — adapted to what was just explained."""

_INTERACTION_PERSONAL = """\
INTERACTIVE TEACHING (student shared a personal example):
- Listen first — reference their specific story (weather, place, what changed).
- Connect it briefly to the lesson, then guide with ONE choice question.
- Stay conversational; do not lecture or list instruments unless they choose that next."""

_INTERACTION_CLARIFICATION = """\
INTERACTIVE TEACHING (student is confused):
- Re-explain your PREVIOUS answer only — do not introduce a new textbook activity or page.
- Use simpler words, smaller steps, and one everyday analogy.
- End with ONE short check question to see if they follow now."""

_AFFIRMATION_RE = re.compile(
    r"^(?:yes|yeah|yep|yup|ok|okay|sure|right|correct|exactly|got\s+it|"
    r"(?:i\s+)?(?:understand|understood)|makes\s+sense|that\s+helps|"
    r"clear\s+now|sounds\s+good)(?:\s+.*)?[.!?]*$",
    re.I,
)

# Student wants the tutor to keep teaching (not a bare "got it")
_EXPAND_REQUEST_RE = re.compile(
    r"\b(?:"
    r"explain(?:\s+more|\s+please)?|please\s+explain|explore|elaborate|"
    r"tell\s+me(?:\s+more)?|go\s+(?:on|ahead)|continue|carry\s+on|"
    r"next(?:\s+part)?|deeper|more\s+(?:about|detail|on)|"
    r"how\s+(?:it|they|one)\s+affects?"
    r")\b",
    re.I,
)

# Tutor offered a single next teaching step (not a multi-choice menu)
_TUTOR_CONTINUE_OFFER_RE = re.compile(
    r"would you like to\s+(?:explore|learn|look|see|try|continue|explain|"
    r"dive|understand|know|hear|discuss|go)|"
    r"want (?:me )?to\s+(?:explain|explore|show|teach|continue)|"
    r"shall (?:we|i)\s+(?:explore|look|continue|explain)",
    re.I,
)

_MULTI_CHOICE_MENU_RE = re.compile(
    r"(?:real[- ]?life\s+example|quick\s+quiz|explore\s+other\s+topics|"
    r"example,\s*(?:a\s+)?(?:quick\s+)?quiz|,\s*or\s+(?:to\s+learn|explore))",
    re.I,
)

_LEGACY_PROMPT_TEMPLATE = """\
You are a friendly AI Tutor helping school students learn clearly and confidently.

Use the provided document context to answer the student's question.
Each context block shows the page, source file, and section when available — use these to stay accurate.

If the answer is not in the document, say:
"This doesn't seem to be covered in the uploaded document. Here is a general explanation:"
Then give a brief, accurate, student-friendly answer.

If the question refers to a specific chapter (for example "Chapter 1"), only use context clearly from that chapter. Do not mix in other chapters.

Rules:
- Use simple, clear, friendly language.
- Match the answer length to the question (one word if asked, full paragraph if needed).
- Never invent facts not present in the context.
- Keep the tone warm and encouraging.

Context:
{context}

Student's Question:
{question}

Your Answer:""".strip()

PROMPT = PromptTemplate(
    template=_LEGACY_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


_PERSONAL_EXAMPLE_RE = re.compile(
    r"\b(i was|it was|when i|where i|one day|one time|suddenly|yesterday|last\s+(?:week|month|year)|"
    r"started\s+(?:rain|snow)|raining|sunny|cloudy|snowing|storm)\b",
    re.I,
)

_SESSION_GREETING_RE = re.compile(
    r"(?:"
    r"\b(?:welcome back|i(?:'m| am) your ai tutor)\b.*\bwhat would you like to learn\b"
    r"|"
    r"\bhere are a few things you can ask me\b"
    r"|"
    r"\bpick one, or type your own question\b"
    r")",
    re.I | re.S,
)

_ACADEMIC_QUESTION_RE = re.compile(
    r"\b("
    r"how (?:far|long|many|much)|can (?:you|we|i) reach|what is|what are|explain|define|"
    r"solve|calculate|find|prove|distance|travel|moon|speed|rate|per day|per hour|"
    r"equation|formula|\d+\s*(?:km|m|cm|mm|years?|days?|hours?|minutes?|%)"
    r")\b",
    re.I,
)


def _is_session_greeting_message(text: str) -> bool:
    return bool(_SESSION_GREETING_RE.search((text or "").strip()))


def _last_assistant_text(conversation_history: list[dict] | None) -> str:
    if not conversation_history:
        return ""
    for turn in reversed(conversation_history):
        if (turn.get("role") or "").lower() == "assistant":
            return (turn.get("content") or "").strip()
    return ""


def _is_personal_dialogue_response(
    query: str, conversation_history: list[dict] | None
) -> bool:
    """Student answered the tutor's question with an example or personal experience."""
    q = (query or "").strip()
    if not q or not conversation_history:
        return False
    if _ACADEMIC_QUESTION_RE.search(q):
        return False
    last_asst = _last_assistant_text(conversation_history)
    if not last_asst or "?" not in last_asst:
        return False
    if _is_session_greeting_message(last_asst):
        return False
    if _PERSONAL_EXAMPLE_RE.search(q):
        return True
    words = q.split()
    if "," in q and len(words) >= 5 and _PERSONAL_EXAMPLE_RE.search(q):
        return True
    return False


def _last_assistant_offered_to_continue(
    conversation_history: list[dict] | None,
) -> bool:
    """True when the tutor asked a single yes/no-style 'shall we explore…?' offer."""
    last = _last_assistant_text(conversation_history)
    if not last or "?" not in last:
        return False
    if _MULTI_CHOICE_MENU_RE.search(last):
        return False
    return bool(_TUTOR_CONTINUE_OFFER_RE.search(last))


def _is_accepting_tutor_continue_offer(
    query: str, conversation_history: list[dict] | None
) -> bool:
    """Student accepted the tutor's offer to keep explaining (e.g. 'Yes please explain')."""
    q = (query or "").strip()
    if not q or not conversation_history:
        return False
    if _EXPAND_REQUEST_RE.search(q):
        return bool(_last_assistant_text(conversation_history))
    if not _last_assistant_offered_to_continue(conversation_history):
        return False
    if not _AFFIRMATION_RE.match(q):
        return False
    return len(q.split()) <= 7


def _is_affirmation_followup(query: str, conversation_history: list[dict] | None) -> bool:
    """True when the student gives a short bare acknowledgment (not a story or answer)."""
    if not conversation_history:
        return False
    q = (query or "").strip()
    if not q or not _AFFIRMATION_RE.match(q):
        return False
    if not _last_assistant_text(conversation_history):
        return False
    # "Yes please explain" / accepting an explore offer → keep teaching
    if _EXPAND_REQUEST_RE.search(q):
        return False
    if _is_accepting_tutor_continue_offer(q, conversation_history):
        return False
    words = q.split()
    if len(words) > 7:
        return False
    if "," in q and len(words) > 4:
        return False
    if _PERSONAL_EXAMPLE_RE.search(q):
        return False
    return True


def _skip_answer_expansion(
    query: str, conversation_history: list[dict] | None
) -> bool:
    from app.services.conversation_context import is_clarification_followup

    return (
        _is_affirmation_followup(query, conversation_history)
        or _is_personal_dialogue_response(query, conversation_history)
        or is_clarification_followup(query, conversation_history)
    )


def _prepare_math_engine_block(
    query: str,
    *,
    subject_name: str,
    class_level: str,
    context: str,
) -> str:
    """SymPy-verified steps for mathematics — injected into the user prompt."""
    if not _is_mathematics_subject(subject_name):
        return ""
    from app.services.math_engine import try_solve

    result = try_solve(query, class_level=class_level, chapter_context=context)
    if not result:
        return ""
    return result.to_prompt_block()


_AGENT_MODE_PROMPTS = {
    "ask": (
        "AGENT MODE — ASK ANYTHING:\n"
        "Answer the student's question using ONLY the retrieved textbook chapter material. "
        "Be clear and direct (short Q&A). Do not invent facts outside the chapter. "
        "Do not turn this into a quiz or a long lecture unless the student asks. "
        "If the question is outside the textbook, say so and tell them to use Ask AI Tutor."
    ),
    "practice": (
        "AGENT MODE — PRACTICE PROBLEMS (STRICT):\n"
        "You are a practice coach. Stay in practice mode for the whole conversation.\n"
        "- If the student asks a concrete math question / pastes a problem: answer that exact question "
        "(same numbers/conditions), explain briefly with step-by-step reasoning when appropriate, "
        "then ask ONE short follow-up question. Do NOT lead with a different problem instead of answering.\n"
        "- If they only ask to practice a topic (no concrete problem): generate ONE problem from the "
        "retrieved textbook chapter, wait for their attempt, then brief feedback.\n"
        "- Do NOT deliver a full lesson dump or open-ended lecture.\n"
        "- Prefer short problems; one at a time unless they ask for a set.\n"
        "- If they ask something unrelated to practicing this chapter, or outside the textbook, "
        "redirect them to Ask AI Tutor for general help, or give a textbook practice problem."
    ),
    "explain": (
        "AGENT MODE — EXPLAIN TOPIC:\n"
        "Teach a clear, structured explanation of the topic from the retrieved textbook chapter only.\n"
        "- Use short sections / steps when helpful.\n"
        "- Do NOT start a quiz unless the student explicitly asks.\n"
        "- Offer to simplify, give an example, or go to the next section at the end.\n"
        "- If the topic is outside the textbook, say so and tell them to use Ask AI Tutor."
    ),
}


def _normalize_agent_mode(agent_mode: str | None) -> str | None:
    if not agent_mode:
        return None
    mode = str(agent_mode).strip().lower()
    return mode if mode in _AGENT_MODE_PROMPTS else None


def _apply_agent_mode(conv: Any, agent_mode: str | None) -> str:
    """Force response_mode for Quick Start agents; return system addendum (may be empty)."""
    from app.services.conversation_context import ResponseMode

    mode = _normalize_agent_mode(agent_mode)
    if not mode:
        return ""
    if mode == "practice":
        conv.response_mode = ResponseMode.QUIZ
    else:
        conv.response_mode = ResponseMode.EXPLANATION
    return _AGENT_MODE_PROMPTS[mode]


def _resolve_answer_type(
    query: str,
    *,
    subject_name: str = "",
    conversation_history: list[dict] | None = None,
    chapter: str = "",
    conv: Any | None = None,
) -> str:
    """Shared answer-type routing for prompts, token limits, and post-processing."""
    from app.services.conversation_context import (
        ResponseMode,
        answer_type_for_followup,
        is_clarification_followup,
        resolve_conversation_context,
    )
    from app.services.conversation_intent_classifier import FollowupType

    q = (query or "").strip()
    if conv is None:
        conv = resolve_conversation_context(
            q, conversation_history=conversation_history, chapter=chapter
        )
    if conv.response_mode == ResponseMode.QUIZ:
        return "quiz"
    if conv.response_mode == ResponseMode.MCQ:
        return "mcq"
    if conv.response_mode == ResponseMode.SUMMARY:
        return "summary"
    if _is_mathematics_subject(subject_name) and q:
        return detect_answer_type(q)
    if _is_personal_dialogue_response(q, conversation_history):
        return "personal-response"
    if _is_accepting_tutor_continue_offer(q, conversation_history):
        return "paragraph"
    if _is_affirmation_followup(q, conversation_history):
        return "affirmation"
    if mapped := answer_type_for_followup(FollowupType(conv.followup_type)):
        # ponytail: BGE can mislabel "what is X?" as ask_comparison after a greeting
        if _CONCEPT_STARTS.match(q) and conv.followup_type in (
            FollowupType.ASK_COMPARISON.value,
            FollowupType.CONTINUE_EXPLANATION.value,
            FollowupType.ASK_EXAMPLE.value,
        ):
            return detect_answer_type(q)
        return mapped
    if is_clarification_followup(q, conversation_history):
        return "clarification"
    return detect_answer_type(q)


def _build_chat_messages(
    query: str,
    context: str,
    *,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    student_name: str = "",
    section_instruction: str = "",
    heading_scope: Any | None = None,
    conversation_history: list[dict] | None = None,
    conversation_memory: Any | None = None,
    chapter_coverage_guidance: str = "",
    learner_snapshot: dict | None = None,
    understanding_scores: dict | None = None,
    resolved_topic: str = "",
    agent_mode: str | None = None,
    student_key: str = "",
    chapter_ids: list[str] | None = None,
) -> list[dict[str, str]]:
    from app.services.section_heading import HeadingScope

    from app.services.conversation_context import resolve_conversation_context

    q = (query or "").strip()
    conv = resolve_conversation_context(
        q,
        conversation_history=conversation_history,
        chapter=chapter,
        memory=conversation_memory,
    )
    agent_addendum = _apply_agent_mode(conv, agent_mode)
    topic = (resolved_topic or conv.resolved_topic or q).strip()
    answer_type = _resolve_answer_type(
        q,
        subject_name=subject_name,
        conversation_history=conversation_history,
        chapter=chapter,
        conv=conv,
    )
    # Forced practice agent always uses quiz-style answers
    if _normalize_agent_mode(agent_mode) == "practice":
        answer_type = "quiz"
    elif _normalize_agent_mode(agent_mode) in ("ask", "explain"):
        if answer_type in ("quiz", "mcq"):
            answer_type = "explanation"
    min_words = _ANSWER_MIN_WORDS.get(answer_type, 120)
    grade_label = _GRADE_LABELS.get(class_level, class_level or "School student")
    complexity = _GRADE_COMPLEXITY.get(class_level, "Use clear, age-appropriate language.")
    class_band = _CLASS_BAND_RULES.get(_class_band(class_level), _CLASS_BAND_RULES["6-8"])
    display_name = _student_first_name(student_name)
    raw_instruction = _ANSWER_INSTRUCTIONS.get(answer_type, _ANSWER_INSTRUCTIONS["short-answer"])
    if answer_type == "greeting":
        instruction = raw_instruction.format(
            student_name=display_name,
            subject=subject_name or "your subject",
            chapter=chapter or "the current chapter",
            min_words=min_words,
        )
    elif "{min_words}" in raw_instruction:
        instruction = raw_instruction.format(min_words=min_words)
    else:
        instruction = raw_instruction

    tier = _structure_tier(answer_type)

    if tier == "compact":
        if answer_type == "affirmation":
            length_policy = "RESPONSE LENGTH: 2-3 sentences only (about 35-55 words). Stay conversational."
            interaction_policy = _INTERACTION_ACKNOWLEDGE
            explanation_structure = ""
            user_closing = (
                "The student understood your last explanation. Reply now with a warm acknowledgment "
                "and ONE question offering: real-life example, quick quiz, next part of this topic, "
                "or other topics — do not teach new content yet:"
            )
        elif answer_type == "personal-response":
            length_policy = "RESPONSE LENGTH: 2-4 sentences (about 40-70 words). Stay conversational."
            interaction_policy = _INTERACTION_PERSONAL
            explanation_structure = ""
            user_closing = (
                "The student shared a real-life example in answer to your question. "
                "Acknowledge their story, connect it briefly to the lesson, then offer next steps:"
            )
        elif answer_type == "clarification":
            length_policy = "RESPONSE LENGTH: About 80-140 words. Plain prose, step by step."
            interaction_policy = _INTERACTION_CLARIFICATION
            explanation_structure = ""
            user_closing = (
                "The student did not understand your last reply. Re-explain that SAME answer "
                "more simply — do not switch to a different activity or page:"
            )
        elif answer_type in ("brief", "one-word"):
            length_policy = _LENGTH_POLICY_BRIEF
            interaction_policy = _INTERACTION_BRIEF
            explanation_structure = ""
            user_closing = f"Write your SHORT answer now ({answer_type} — 2-4 sentences max, no sections):"
        elif answer_type == "summary":
            length_policy = "RESPONSE LENGTH: 2-5 sentences only (about 50-110 words). Plain prose. No bold headings."
            interaction_policy = ""
            explanation_structure = ""
            user_closing = "Write your short summary now:"
        else:
            length_policy = "RESPONSE LENGTH: 2-3 sentences only."
            interaction_policy = ""
            explanation_structure = ""
            user_closing = "Write your greeting now:"
    elif tier == "full":
        length_policy = _LENGTH_POLICY_LONG.format(min_words=min_words)
        interaction_policy = _INTERACTION_LONG
        explanation_structure = _EXPLANATION_STRUCTURE_FULL
        user_closing = (
            f"Write your complete exam-style answer now ({answer_type}, minimum {min_words} words, "
            "all five emoji sections):"
        )
    elif tier == "structured" and not _is_mathematics_subject(subject_name):
        if answer_type == "summary":
            explanation_structure = _EXPLANATION_STRUCTURE_SUMMARY
            length_policy = _LENGTH_POLICY_STRUCTURED_MIDDLE.format(min_words=min_words)
            interaction_policy = _INTERACTION_STRUCTURED
            user_closing = (
                f"Write the mentor recap now ({answer_type}, about {min_words} words). "
                "Use **What We Covered**, **Key Takeaways**, and **What's Next**."
            )
        elif answer_type == "factual":
            explanation_structure = _EXPLANATION_STRUCTURE_FACTUAL
            length_policy = _LENGTH_POLICY_FACTUAL.format(min_words=min_words)
            interaction_policy = _INTERACTION_FACTUAL
            user_closing = (
                f"Write your concise factual answer now ({answer_type}, about {min_words}–80 words). "
                "Use **Topic**, **Answer**, and optional **Example** only:"
            )
        else:
            explanation_structure, length_policy, interaction_policy, user_closing = (
                _subject_structured_prompt(
                    class_band=_class_band(class_level),
                    answer_type=answer_type,
                    min_words=min_words,
                )
            )
    elif tier == "quiz":
        explanation_structure, length_policy, interaction_policy, user_closing = (
            _subject_quiz_prompt(answer_type=answer_type, min_words=min_words)
        )
    else:
        length_policy = _LENGTH_POLICY_DIRECT.format(max_words=_DIRECT_ANSWER_MAX_WORDS)
        interaction_policy = _INTERACTION_DIRECT
        explanation_structure = _EXPLANATION_STRUCTURE_DIRECT
        user_closing = (
            f"Write your direct answer now ({answer_type}, MAX {_DIRECT_ANSWER_MAX_WORDS} words, "
            "plain prose only — no section headers):"
        )
        if not _is_mathematics_subject(subject_name):
            complexity = (
                "Write flowing prose the student can read aloud — complete sentences, "
                "not choppy fragments."
            )
            class_band = (
                "Direct definition mode: one paragraph plus 2 bullets, then a follow-up question. "
                "Mention the chapter naturally when the material supports it."
            )

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        length_policy = (
            "RESPONSE LENGTH: Cover every required subtopic; each subtopic gets 2–3 bullet points. "
            "Do not merge into one paragraph."
        )
        interaction_policy = (
            "After all subtopic blocks, you may add ONE short closing question. "
            "Do not add a question before finishing every subtopic. "
            "Never write figure captions, 'Fig. 2.x' lines, or 'Page N' lines — figures are shown separately."
        )
        explanation_structure = _EXPLANATION_STRUCTURE_DIRECT
        instruction = (
            "Use the mandatory main-section format in TEXTBOOK SCOPE: "
            "one **bold subtopic** heading per instrument/topic, then 1–2 short sentences "
            "(or • bullets) under each heading."
        )
        user_closing = (
            "Write the answer now using **bold subtopic headings** for EACH subtopic "
            "listed in TEXTBOOK SCOPE (in order). Under each heading write 1–2 sentences "
            "about that instrument only. Do not use a single paragraph."
        )

    qtype = detect_question_type(q)
    math_format = _math_format_tier(class_level, qtype)
    elementary_math = math_format == "elementary"
    concept_math = math_format in ("elementary", "middle", "secondary")

    (
        instruction,
        length_policy,
        interaction_policy,
        explanation_structure,
        user_closing,
    ) = _apply_mathematics_prompt_overrides(
        subject_name=subject_name,
        answer_type=answer_type,
        heading_scope=heading_scope,
        class_level=class_level,
        question_type=qtype,
        instruction=instruction,
        length_policy=length_policy,
        interaction_policy=interaction_policy,
        explanation_structure=explanation_structure,
        user_closing=user_closing,
    )

    from app.services.math_lesson.service import (
        get_visualization_appendix_prompt,
        should_use_interactive_math_lesson,
    )

    if should_use_interactive_math_lesson(
        subject_name=subject_name,
        answer_type=answer_type,
        query=q,
        heading_scope=heading_scope,
        question_type=qtype,
    ):
        from app.services.math_lesson.fallbacks import (
            get_animation_prompt_note,
            get_visualization_catalog_hint,
            query_requests_assessment,
            query_requests_hints,
        )

        catalog_hint = get_visualization_catalog_hint(
            q, class_level, conversation_history=conversation_history
        )
        viz_appendix = get_visualization_appendix_prompt(q, elementary=elementary_math)
        explanation_structure = (
            (explanation_structure + "\n\n" + catalog_hint + "\n\n" + viz_appendix)
            if explanation_structure
            else catalog_hint + "\n\n" + viz_appendix
        )
        animation_note = get_animation_prompt_note(q, class_level)
        optional_note = ""
        if query_requests_hints(q):
            optional_note += " Include progressive aiHints in the math-lesson JSON."
        if query_requests_assessment(q):
            optional_note += " Include 3–5 conceptual assessment questions in the math-lesson JSON."
        if not optional_note:
            optional_note = (
                " Do NOT include aiHints, assessment, practiceMode, or commonMistakes "
                "in the math-lesson JSON."
            )
        if concept_math:
            closing_by_tier = {
                "elementary": (
                    f"Write your short, kid-friendly answer now ({answer_type}): "
                    "plain prose only (no **To Find** / **Formula** sections). "
                ),
                "middle": (
                    f"Write your Class 6–8 concept answer now ({answer_type}): "
                    "2–3 paragraphs, no **To Find** / **Formula** sections. "
                ),
                "secondary": (
                    f"Write your Class 9–12 concept answer now ({answer_type}): "
                    "detailed academic prose, no **To Find** / **Formula** sections. "
                ),
            }
            user_closing = (
                closing_by_tier[math_format]
                + "You MUST append a complete ```math-lesson``` JSON visualization block at the very end "
                "(Interactive Exploration is mandatory). Do NOT use ```markdown fences."
                f"{animation_note}{optional_note}"
            )
        else:
            user_closing = (
                f"Write your complete mathematics tutor answer now ({answer_type}): "
                "use ALL standard sections (**To Find** through **Practice Question**) first. "
                "You MUST append a complete ```math-lesson``` JSON visualization block at the very end "
                "(Interactive Exploration is mandatory for every mathematics question). "
                "Do NOT use ```markdown fences."
                f"{animation_note}{optional_note}"
            )

    from app.services.science_experiment.service import (
        get_experiment_appendix_prompt,
        should_use_interactive_science_experiment,
    )

    if should_use_interactive_science_experiment(
        subject_name=subject_name,
        answer_type=answer_type,
        query=q,
        heading_scope=heading_scope,
        class_level=class_level,
    ):
        from app.services.science_experiment.fallbacks import get_experiment_catalog_hint

        catalog_hint = get_experiment_catalog_hint(q, class_level)
        exp_appendix = get_experiment_appendix_prompt()
        explanation_structure = (
            (explanation_structure + "\n\n" + catalog_hint + "\n\n" + exp_appendix)
            if explanation_structure
            else catalog_hint + "\n\n" + exp_appendix
        )
        user_closing = (
            f"Write your complete science tutor answer now ({answer_type}) using structured headings. "
            "You MUST append a complete ```science-experiment``` JSON block at the very end "
            "(Interactive Experiment with three synchronized views is mandatory). "
            "Do NOT use ```markdown fences."
        )

    skip_question_type = answer_type in (
        "greeting", "affirmation", "personal-response", "clarification",
        "quiz", "mcq", "summary", "short-answer",
    )
    question_type_guidance = _build_question_type_guidance(
        query, skip=skip_question_type
    )
    subject_guidelines = (
        ""
        if tier == "direct" and not _is_mathematics_subject(subject_name)
        else _build_subject_guidelines(subject_name, math_format=math_format)
    )
    learner_guidance = _build_learner_guidance(learner_snapshot)
    adaptive_guidance = _build_adaptive_guidance(
        query=q,
        topic=topic,
        understanding_scores=understanding_scores,
        learner_snapshot=learner_snapshot,
    )
    lia_addendum = ""
    if student_key and student_key.isdigit():
        from app.services.learning_intelligence.clients.lia_client import get_guidance_for_turn_sync

        lia = get_guidance_for_turn_sync(
            student_user_id=int(student_key),
            query=q,
            topic=topic,
            subject_name=subject_name,
            chapter=chapter,
            chapter_ids=chapter_ids,
            class_level=class_level,
            board=board,
            agent_mode=agent_mode,
            understanding_scores=understanding_scores,
        )
        if lia:
            if lia.get("learner_guidance"):
                learner_guidance = lia["learner_guidance"]
            if lia.get("adaptive_guidance"):
                adaptive_guidance = (
                    f"{adaptive_guidance}\n{lia['adaptive_guidance']}".strip()
                    if adaptive_guidance
                    else lia["adaptive_guidance"]
                )
            lia_addendum = lia.get("prompt_instructions") or ""

    system = _SYSTEM_PROMPT_TEMPLATE.format(
        student_name=display_name,
        grade_label=grade_label,
        board=board or "General",
        subject=subject_name or "General",
        chapter=chapter or "Current chapter",
        question_type_guidance=question_type_guidance,
        subject_guidelines=subject_guidelines,
        chapter_coverage_guidance=chapter_coverage_guidance,
        complexity_rule=complexity,
        class_band_rule=class_band,
        length_policy=length_policy,
        answer_instruction=instruction,
        interaction_policy=interaction_policy,
        explanation_structure=explanation_structure,
        learner_guidance=learner_guidance,
        adaptive_guidance=adaptive_guidance,
    )
    if agent_addendum:
        system = system + "\n\n" + agent_addendum
    if lia_addendum:
        system = system + "\n\n" + lia_addendum
    from app.services.conversation_memory import format_memory_for_prompt

    memory_block = format_memory_for_prompt(conversation_memory)
    if memory_block:
        system = system + "\n\n" + memory_block
    section_block = ""
    if section_instruction.strip():
        section_block = f"\n\nTEXTBOOK SCOPE:\n{section_instruction.strip()}\n"

    math_block = _prepare_math_engine_block(
        q,
        subject_name=subject_name,
        class_level=class_level,
        context=context,
    )
    math_section = ""
    if math_block:
        math_section = f"\n\n{math_block}\n"

    user = _USER_PROMPT_TEMPLATE.format(
        context=context or "(No chapter text retrieved — use accurate general knowledge.)",
        question=query,
        user_closing=user_closing,
    )
    if math_section:
        user = math_section + user
    if section_block:
        user = section_block + user

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if conversation_history:
        for turn in conversation_history[-10:]:
            role = (turn.get("role") or "").lower()
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    return messages


def _build_prompt(
    query: str,
    context: str,
    *,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    student_name: str = "",
) -> str:
    """Legacy single-string prompt (used by old /chat path)."""
    msgs = _build_chat_messages(
        query,
        context,
        class_level=class_level,
        board=board,
        subject_name=subject_name,
        chapter=chapter,
        student_name=student_name,
    )
    return f"{msgs[0]['content']}\n\n{msgs[1]['content']}"


# ── Context assembly ─────────────────────────────────────────────────────────

def _format_chunk_for_context(doc) -> str:
    meta = getattr(doc, "metadata", None) or {}
    page_raw = meta.get("page")
    human_page = None
    if page_raw is not None:
        try:
            human_page = int(page_raw) + 1
        except (TypeError, ValueError):
            pass
    src = meta.get("source", "")
    short_src = os.path.basename(str(src)) if src else ""
    sec = (meta.get("section_hint") or "").strip()
    tag = f"[page {human_page}]" if human_page is not None else "[page ?]"
    if short_src:
        tag += f" [source: {short_src}]"
    if sec:
        tag += f" [section: {sec}]"
    body = (getattr(doc, "page_content", "") or "").strip()
    return f"{tag}\n{body}"


def _join_context_within_budget(docs: list, char_budget: int | None = None) -> str:
    budget = char_budget if char_budget is not None else CONTEXT_CHAR_BUDGET
    parts: list[str] = []
    used = 0
    for d in docs:
        block = _format_chunk_for_context(d)
        add = len(block) + (2 if parts else 0)
        if parts and used + add > budget:
            break
        if not parts and len(block) > budget:
            parts.append(block[:budget])
            break
        parts.append(block)
        used += add
    return "\n\n".join(parts)


# ── Mistral API calls (async) ────────────────────────────────────────────────

def _answer_word_count(text: str) -> int:
    return len((text or "").split())


def _direct_answer_needs_shrink(answer: str) -> bool:
    wc = _answer_word_count(answer)
    if wc > _DIRECT_ANSWER_MAX_WORDS:
        return True
    return bool(_STRUCTURED_HEADER_RE.search(answer or ""))


def _mistral_token_limit_for_answer_type(
    answer_type: str,
    *,
    heading_scope: Any | None = None,
    subject_name: str = "",
) -> int | None:
    from app.services.section_heading import HeadingScope

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        return _MAIN_SECTION_TOKEN_LIMIT
    # Math answers must include ```math-lesson``` JSON — 160 tokens cuts it off
    # and the incomplete fence wipes the reply ("Sorry — something went wrong").
    if _is_mathematics_subject(subject_name) and answer_type not in (
        _MATH_DIALOGUE_TYPES | _MATH_SHORT_TYPES
    ):
        return None
    if _structure_tier(answer_type) == "direct":
        return _DIRECT_ANSWER_TOKEN_LIMIT
    return None


async def _finalize_direct_answer(
    messages: list[dict[str, str]],
    answer: str,
    query: str,
    *,
    answer_type: str,
    heading_scope: Any | None = None,
    conversation_history: list[dict] | None = None,
    subject_name: str = "",
) -> str:
    """Shrink direct answers that leaked into long/structured form. Never expand them."""
    from app.services.section_heading import HeadingScope

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        return strip_embedded_figure_lines(answer.strip())
    if _is_mathematics_subject(subject_name) and answer_type not in (
        _MATH_DIALOGUE_TYPES | _MATH_SHORT_TYPES
    ):
        # Don't shrink math — required sections + math-lesson JSON look "too long".
        return strip_embedded_figure_lines(answer.strip())
    if _structure_tier(answer_type) != "direct":
        if not _skip_answer_expansion(query, conversation_history):
            answer = await _expand_short_answer(
                messages, answer, query, heading_scope=heading_scope
            )
        return strip_embedded_figure_lines(answer.strip())
    return strip_embedded_figure_lines(
        normalize_direct_answer_prose(
            (await _shrink_overlong_direct_answer(messages, answer, query)).strip()
        )
    )


async def _shrink_overlong_direct_answer(
    messages: list[dict[str, str]],
    answer: str,
    query: str,
) -> str:
    """One retry when the model ignores direct-answer length / format rules."""
    if not _direct_answer_needs_shrink(answer):
        return answer
    logger.info(
        "Direct answer too long or structured (%d words); shrinking for query=%r",
        _answer_word_count(answer),
        query[:60],
    )
    shrink_msgs = [
        *messages,
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": (
                f"Rewrite that answer in MAX {_DIRECT_ANSWER_MAX_WORDS} words. "
                "Plain prose only — NO **Topic**, **Key Points**, **Detailed Explanation**, "
                "**Remember**, or **Try This** headers. "
                "Keep the definition, at most 3 bullets, and one short follow-up question."
            ),
        },
    ]
    try:
        shorter = await _call_mistral_async(
            shrink_msgs, max_tokens=_DIRECT_ANSWER_TOKEN_LIMIT
        )
        if shorter.strip() and _answer_word_count(shorter) <= _DIRECT_ANSWER_MAX_WORDS + 25:
            return shorter.strip()
    except Exception as exc:
        logger.warning("Direct-answer shrink failed: %s", exc)
    return answer


async def _expand_short_answer(
    messages: list[dict[str, str]],
    answer: str,
    query: str,
    *,
    heading_scope: Any | None = None,
) -> str:
    """One retry when the model stops too early despite length instructions."""
    from app.services.section_heading import HeadingScope

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        return answer
    atype = detect_answer_type(query)
    if _structure_tier(atype) == "direct":
        return answer
    if atype in ("greeting", "one-word", "brief"):
        return answer
    min_w = _ANSWER_MIN_WORDS.get(atype, 120)
    if _answer_word_count(answer) >= int(min_w * 0.65):
        return answer
    logger.info(
        "Answer too short (%d words, need ~%d); requesting expansion for query=%r",
        _answer_word_count(answer),
        min_w,
        query[:60],
    )
    if _structure_tier(atype) == "full":
        expand_hint = (
            "Include ALL sections with emoji headers: "
            "🌱 Concept Overview, 📚 Detailed Explanation, 🌍 Real-Life Example, "
            "📝 Key Points to Remember (3-5 bullets), ❓ Quick Check."
        )
    elif _structure_tier(atype) == "structured":
        expand_hint = (
            "Include ALL **bold** side headings: **Topic**, **In Simple Words**, **Key Points**, "
            "**Detailed Explanation**, **Example**, **Remember**, and **Try This**."
        )
    else:
        expand_hint = (
            "Continue in plain paragraphs only — NO emoji section headers, "
            "no 'Concept Overview' / 'Quick Check' labels."
        )
    extend_msgs = [
        *messages,
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": (
                f"Your answer was too short ({_answer_word_count(answer)} words). "
                f"Continue and expand until the full answer is at least {min_w} words. "
                f"{expand_hint}"
            ),
        },
    ]
    try:
        extra = await _call_mistral_async(extend_msgs)
        if extra.strip():
            return f"{answer.strip()}\n\n{extra.strip()}"
    except Exception as exc:
        logger.warning("Short-answer expansion failed: %s", exc)
    return answer


def _ensure_mistral_config() -> None:
    llm_client.ensure_llm_config()


async def _call_mistral_async(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    feature: str = "chat",
) -> str:
    """Non-blocking OpenAI-compatible chat completion (legacy name kept)."""
    return await llm_client.complete(
        messages,
        feature=feature,
        max_tokens=max_tokens,
        empty_fallback=ANSWER_NOT_IN_CHAPTER,
    )


async def _stream_mistral_async(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    feature: str = "chat",
) -> AsyncIterator[str]:
    """Yield tokens from OpenAI-compatible SSE stream (legacy name kept)."""
    async for token in llm_client.stream(messages, feature=feature, max_tokens=max_tokens):
        yield token


# ── Fallback (no API key) ────────────────────────────────────────────────────

def _best_chunk_fallback(query: str, docs: list) -> str:
    from app.services.query_match import document_page, keyword_match_score

    best_text, best_score, best_page = "", -1, 10**9
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        score = keyword_match_score(query, text)
        pg = document_page(d)
        if score > best_score or (score == best_score and pg < best_page):
            best_score, best_page, best_text = score, pg, text

    if not best_text:
        return ANSWER_NOT_IN_CHAPTER

    parts = re.split(r"(?<=[.?!])\s+", best_text)
    snippet = " ".join(parts[:3]).strip()
    return snippet or best_text[:600].strip()


# ── Public API ───────────────────────────────────────────────────────────────

async def chapter_aware_qa(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None = None,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    chapter_names: list[str] | None = None,
    conversation_history: list[dict] | None = None,
    conversation_memory: Any | None = None,
    student_name: str = "",
    student_key: str = "",
    images_only: bool = False,
    agent_mode: str | None = None,
) -> tuple[str, list[dict], dict | None, dict | None]:
    """
    Retrieve relevant chunks from ChromaDB and answer via Mistral (async).

    Returns ``(answer_text, related_images, math_lesson, science_experiment)``.
    """
    from app.services.math_lesson.service import finalize_math_answer
    from app.services.science_experiment.service import finalize_science_answer
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.conversation_memory import prepare_conversation_inputs
    from app.services.section_retrieval import retrieve_for_tutor_query
    from app.services.chapter_scope import resolve_chapter_awareness_turn

    recent_hist, session_mem = prepare_conversation_inputs(
        conversation_history,
        memory=conversation_memory,
        chapter=chapter,
    )

    conv = resolve_conversation_context(
        query,
        conversation_history=recent_hist,
        chapter=chapter,
        memory=session_mem,
    )
    _apply_agent_mode(conv, agent_mode)
    retrieval_query = conv.retrieval_query or query

    docs, scope, section_instruction = retrieve_for_tutor_query(
        retrieval_query,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
    )

    # Topics-left questions: answer from coverage store (no LLM needed).
    if student_key and student_key.isdigit() and chapter_ids:
        try:
            from app.core.database import SessionLocal
            from app.modules.student_learning.topic_progress import try_topics_left_reply

            db = SessionLocal()
            try:
                left_reply = try_topics_left_reply(
                    db,
                    user_id=int(student_key),
                    query=query,
                    chapter_ids=chapter_ids,
                    subject_name=subject_name,
                    chapter=chapter,
                    board=board,
                    class_level=class_level,
                )
            finally:
                db.close()
            if left_reply:
                return left_reply, [], None, None
        except Exception:
            logger.exception("topics-left reply failed")

    early, effective_query, _assessment, coverage_guidance = await resolve_chapter_awareness_turn(
        query,
        docs=docs,
        conversation_history=recent_hist,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
        scope_query=retrieval_query,
    )
    if early:
        return early, [], None, None

    img_allowed = should_retrieve_images(
        conv,
        chapter_ids=chapter_ids,
        heading_scope_kind=scope.kind,
        subject_name=subject_name,
    )
    if images_only:
        related: list[dict] = []
        if chapter_ids and img_allowed and docs:
            img_top_n = _image_top_n_for_scope(scope, docs=docs)
            try:
                related = await asyncio.wait_for(
                    asyncio.to_thread(
                        _fetch_related_images,
                        scope,
                        collection_name=collection_name,
                        chapter_ids=chapter_ids,
                        chapter_names=chapter_names or [],
                        chapter=chapter,
                        query=effective_query,
                        docs=docs,
                        conversation_history=recent_hist,
                        top_n=img_top_n,
                    ),
                    timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
                )
            except Exception as exc:
                logger.warning("Image-only retrieval failed: %s", exc)
        return "", related, None, None
    if not docs and not coverage_guidance:
        return ANSWER_NOT_IN_CHAPTER, [], None, None

    learner_snapshot, understanding_scores = await _mentor_profile_for_turn(
        student_key,
        effective_query,
        recent_hist,
        conv.resolved_topic or effective_query,
    )

    if student_key and student_key.isdigit():
        from app.services.learning_intelligence.clients.lia_client import emit_chat_user_question

        last_assistant = ""
        for turn in reversed(recent_hist or []):
            if (turn.get("role") or "").lower() == "assistant":
                last_assistant = (turn.get("content") or "").strip()
                break
        emit_chat_user_question(
            student_user_id=int(student_key),
            school_id=None,
            query=effective_query,
            subject_name=subject_name,
            chapter=chapter,
            chapter_ids=chapter_ids,
            class_level=class_level,
            board=board,
            understanding_scores=understanding_scores,
            agent_mode=agent_mode,
            last_assistant=last_assistant,
        )
        try:
            from app.modules.student_learning.topic_progress import record_turn_topic_progress

            scope_title = None
            if getattr(scope, "matched", None) is not None:
                scope_title = getattr(scope.matched, "title", None)
            record_turn_topic_progress(
                student_user_id=int(student_key),
                query=effective_query,
                chapter_ids=chapter_ids,
                subject_name=subject_name,
                chapter=chapter,
                board=board,
                class_level=class_level,
                scope_title=scope_title,
            )
        except Exception:
            logger.exception("topic progress record failed")

    context = _join_context_within_budget(docs)
    messages = _build_chat_messages(
        effective_query,
        context,
        class_level=class_level,
        board=board,
        subject_name=subject_name,
        chapter=chapter,
        student_name=student_name,
        section_instruction=section_instruction,
        heading_scope=scope,
        conversation_history=recent_hist,
        conversation_memory=session_mem,
        chapter_coverage_guidance=coverage_guidance,
        learner_snapshot=learner_snapshot,
        understanding_scores=understanding_scores,
        resolved_topic=conv.resolved_topic or effective_query,
        agent_mode=agent_mode,
        student_key=student_key,
        chapter_ids=chapter_ids,
    )
    img_top_n = _image_top_n_for_scope(scope, docs=docs)
    answer_type = _resolve_answer_type(
        effective_query,
        subject_name=subject_name,
        conversation_history=recent_hist,
        chapter=chapter,
        conv=conv,
    )
    if _normalize_agent_mode(agent_mode) == "practice":
        answer_type = "quiz"
    elif _normalize_agent_mode(agent_mode) in ("ask", "explain") and answer_type in ("quiz", "mcq"):
        answer_type = "explanation"
    token_limit = _mistral_token_limit_for_answer_type(
        answer_type, heading_scope=scope, subject_name=subject_name
    )

    try:
        answer = await _call_mistral_async(messages, max_tokens=token_limit)
        answer = await _finalize_direct_answer(
            messages,
            answer,
            effective_query,
            answer_type=answer_type,
            heading_scope=scope,
            conversation_history=recent_hist,
            subject_name=subject_name,
        )
    except FileNotFoundError:
        answer = _best_chunk_fallback(effective_query, docs)
    except Exception as exc:
        logger.error("Mistral call failed: %s: %s", type(exc).__name__, exc)
        answer = _best_chunk_fallback(effective_query, docs)

    related: list[dict] = []
    if chapter_ids and img_allowed:
        try:
            related = await asyncio.wait_for(
                asyncio.to_thread(
                    _fetch_related_images,
                    scope,
                    collection_name=collection_name,
                    chapter_ids=chapter_ids,
                    chapter_names=chapter_names or [],
                    chapter=chapter,
                    query=effective_query,
                    docs=docs,
                    conversation_history=recent_hist,
                    top_n=img_top_n,
                ),
                timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Image retrieval timed out after %.0fs for query=%r",
                IMAGE_RETRIEVAL_TIMEOUT_SEC,
                effective_query[:80],
            )
        except Exception as exc:
            logger.warning("Image retrieval failed: %s", exc)

    answer, math_lesson = finalize_math_answer(
        answer,
        effective_query,
        class_level=class_level,
        subject_name=subject_name,
        conversation_history=recent_hist,
    )
    answer, science_experiment = finalize_science_answer(
        answer,
        effective_query,
        class_level=class_level,
        subject_name=subject_name,
    )

    await _mentor_profile_after_turn(
        student_key,
        conv.resolved_topic or effective_query,
        understanding_scores,
    )
    if student_key and student_key.isdigit():
        from app.services.learning_intelligence.clients.lia_client import emit_chat_assistant_response

        emit_chat_assistant_response(
            student_user_id=int(student_key),
            subject_name=subject_name,
            chapter=chapter,
            topic=conv.resolved_topic or effective_query,
            agent_mode=agent_mode,
        )
    return answer, related, math_lesson, science_experiment


def _image_top_n_for_scope(scope, *, docs: list | None = None) -> int:
    from app.services.section_heading import HeadingScope, subtopics_for_main_section

    if isinstance(scope, HeadingScope) and scope.is_main_section:
        n = len(subtopics_for_main_section(scope, docs or []))
        if n == 0:
            n = len(scope.child_headings or [])
        return min(12, max(TOP_RELATED_IMAGES, n + 1))
    return TOP_RELATED_IMAGES


def _fetch_related_images(
    scope,
    *,
    collection_name: str,
    chapter_ids: list[str],
    chapter_names: list[str],
    chapter: str,
    query: str,
    docs: list,
    conversation_history: list[dict] | None,
    top_n: int,
) -> list[dict]:
    from app.services.section_retrieval import related_images_for_heading_scope

    return related_images_for_heading_scope(
        scope,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        chapter_single=chapter,
        query=query,
        retrieved_docs=docs,
        conversation_history=conversation_history,
        fallback_top_n=top_n,
    )


async def chapter_aware_qa_stream(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None = None,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    chapter_names: list[str] | None = None,
    emit_related_images: Callable[[list[dict]], Awaitable[None]] | None = None,
    emit_clean_answer: Callable[[str], Awaitable[None]] | None = None,
    emit_math_lesson: Callable[[dict | None, str], Awaitable[None]] | None = None,
    emit_science_experiment: Callable[[dict | None, str], Awaitable[None]] | None = None,
    conversation_history: list[dict] | None = None,
    conversation_memory: Any | None = None,
    student_name: str = "",
    student_key: str = "",
    voice_mode: bool = False,
    tutor_state: str = "TEACHING",
    understanding_scores: dict | None = None,
    learner_snapshot: dict | None = None,
    pipeline_timing: Any | None = None,
    agent_mode: str | None = None,
    quiz_pending: bool = False,
    quiz_question: str = "",
    quiz_attempts: int = 0,
    explained_points: list[str] | None = None,
) -> AsyncIterator[str]:
    """
    Streaming version of chapter_aware_qa.

    Optionally invokes *emit_related_images* when ranked images are ready
    (usually during the first tokens, without blocking retrieval).
    """
    from app.config import CONTEXT_CHAR_BUDGET, EARLY_IMAGE_MIN_CHARS, RETRIEVAL_K, VOICE_EARLY_IMAGE_MIN_CHARS, VOICE_MAX_TOKENS, VOICE_CONTEXT_CHAR_BUDGET, VOICE_RETRIEVAL_K
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.conversation_memory import format_memory_for_prompt, prepare_conversation_inputs
    from app.services.image_service.textbook_image_retrieval import early_related_images_for_query
    from app.services.math_lesson.service import finalize_math_answer
    from app.services.science_experiment.service import finalize_science_answer
    from app.services.section_retrieval import retrieve_for_tutor_query

    retrieval_k = VOICE_RETRIEVAL_K if voice_mode else RETRIEVAL_K
    context_budget = VOICE_CONTEXT_CHAR_BUDGET if voice_mode else CONTEXT_CHAR_BUDGET

    if voice_mode:
        from app.services.voice_stt_postprocess import (
            INCOMPLETE_UTTERANCE_REPLY,
            is_incomplete_voice_utterance,
        )

        if is_incomplete_voice_utterance(query):
            if emit_related_images:
                await emit_related_images([])
            yield INCOMPLETE_UTTERANCE_REPLY
            return

    from app.services.chapter_scope import resolve_chapter_awareness_turn

    recent_hist, session_mem = prepare_conversation_inputs(
        conversation_history,
        memory=conversation_memory,
        chapter=chapter,
    )

    conv = resolve_conversation_context(
        query,
        conversation_history=recent_hist,
        chapter=chapter,
        memory=session_mem,
    )
    _apply_agent_mode(conv, agent_mode)
    retrieval_query = conv.retrieval_query or query

    if voice_mode:
        docs, scope, section_instruction = await asyncio.to_thread(
            retrieve_for_tutor_query,
            retrieval_query,
            collection_name=collection_name,
            chapter_ids=chapter_ids,
            chapter_names=chapter_names,
            k=retrieval_k,
        )
        if pipeline_timing is not None:
            pipeline_timing.mark_rag_done()
    else:
        docs, scope, section_instruction = retrieve_for_tutor_query(
            retrieval_query,
            collection_name=collection_name,
            chapter_ids=chapter_ids,
            chapter_names=chapter_names,
            k=retrieval_k,
        )

    if student_key and student_key.isdigit() and chapter_ids:
        try:
            from app.core.database import SessionLocal
            from app.modules.student_learning.topic_progress import try_topics_left_reply

            db = SessionLocal()
            try:
                left_reply = try_topics_left_reply(
                    db,
                    user_id=int(student_key),
                    query=query,
                    chapter_ids=chapter_ids,
                    subject_name=subject_name,
                    chapter=chapter,
                    board=board,
                    class_level=class_level,
                )
            finally:
                db.close()
            if left_reply:
                if emit_related_images:
                    await emit_related_images([])
                yield left_reply
                return
        except Exception:
            logger.exception("topics-left reply failed (stream)")

    early, effective_query, _assessment, coverage_guidance = await resolve_chapter_awareness_turn(
        query,
        docs=docs,
        conversation_history=recent_hist,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        board=board,
        class_level=class_level,
        subject_name=subject_name,
        scope_query=retrieval_query,
    )
    if early:
        if emit_related_images:
            await emit_related_images([])
        yield early
        return

    use_text_format = _voice_should_use_text_format(
        subject_name,
        voice_mode=voice_mode,
        understanding_scores=understanding_scores,
        query=effective_query,
        heading_scope=scope,
    )
    voice_live_teaching = voice_mode and not use_text_format

    img_allowed = should_retrieve_images(
        conv,
        chapter_ids=chapter_ids,
        heading_scope_kind=scope.kind,
        subject_name=subject_name,
        voice_mode=voice_mode,
    )
    img_top_n = _image_top_n_for_scope(scope, docs=docs)
    if not docs and not coverage_guidance:
        if emit_related_images:
            await emit_related_images([])
        yield ANSWER_NOT_IN_CHAPTER
        return

    if emit_related_images and not img_allowed:
        await emit_related_images([])

    if student_key and student_key.isdigit():
        from app.services.learning_intelligence.clients.lia_client import emit_chat_user_question

        last_assistant = ""
        for turn in reversed(recent_hist or []):
            if (turn.get("role") or "").lower() == "assistant":
                last_assistant = (turn.get("content") or "").strip()
                break
        emit_chat_user_question(
            student_user_id=int(student_key),
            school_id=None,
            query=effective_query,
            subject_name=subject_name,
            chapter=chapter,
            chapter_ids=chapter_ids,
            class_level=class_level,
            board=board,
            understanding_scores=understanding_scores,
            agent_mode=agent_mode,
            last_assistant=last_assistant,
        )
        try:
            from app.modules.student_learning.topic_progress import record_turn_topic_progress

            scope_title = None
            if getattr(scope, "matched", None) is not None:
                scope_title = getattr(scope.matched, "title", None)
            record_turn_topic_progress(
                student_user_id=int(student_key),
                query=effective_query,
                chapter_ids=chapter_ids,
                subject_name=subject_name,
                chapter=chapter,
                board=board,
                class_level=class_level,
                scope_title=scope_title,
            )
        except Exception:
            logger.exception("topic progress record failed (stream)")

    context = _join_context_within_budget(docs, char_budget=context_budget)

    if not voice_live_teaching and student_key and learner_snapshot is None:
        learner_snapshot, understanding_scores = await _mentor_profile_for_turn(
            student_key,
            effective_query,
            recent_hist,
            conv.resolved_topic or effective_query,
        )

    if voice_live_teaching:
        from app.services.voice_tutor import (
            LearnerProfileSnapshot,
            TutorState,
            UnderstandingScores,
            build_voice_mistral_messages,
            classify_reply_intent,
        )

        try:
            tutor_st = TutorState(tutor_state)
        except ValueError:
            tutor_st = TutorState.TEACHING
        scores = understanding_scores or {}
        understanding = UnderstandingScores(
            understanding=float(scores.get("understanding", 0.5)),
            confidence=float(scores.get("confidence", 0.5)),
            confusion=float(scores.get("confusion", 0.0)),
            is_affirmation=bool(scores.get("is_affirmation")),
            wants_expansion=bool(scores.get("wants_expansion")),
            wants_quiz=bool(scores.get("wants_quiz")),
        )
        learner = (
            LearnerProfileSnapshot(**learner_snapshot)
            if learner_snapshot
            else None
        )
        messages = build_voice_mistral_messages(
            # Always the student's raw utterance — retrieval may use a different
            # string via retrieval_query / scope, but never silently swap this.
            query,
            context,
            class_level=class_level,
            board=board,
            subject_name=subject_name,
            chapter=chapter,
            student_name=student_name,
            conversation_history=recent_hist,
            tutor_state=tutor_st,
            understanding=understanding,
            learner=learner,
            expand_deep=understanding.wants_expansion or understanding.confusion >= 0.55,
            quiz_pending=quiz_pending,
            quiz_question=quiz_question,
            quiz_attempts=quiz_attempts,
            explained_points=explained_points,
            # Classified from the student's actual raw message, not
            # effective_query — chapter-scope resolution above may have
            # rewritten effective_query (e.g. after a misheard-term
            # confirmation) to something the student never literally said.
            reply_intent=classify_reply_intent(query, quiz_pending=quiz_pending),
        )
        if coverage_guidance:
            messages[0]["content"] = messages[0]["content"] + "\n\n" + coverage_guidance
        if session_mem.turn_count or session_mem.conversation_summary:
            mem_block = format_memory_for_prompt(session_mem)
            if mem_block:
                messages[0]["content"] = messages[0]["content"] + "\n\n" + mem_block
        if student_key and student_key.isdigit():
            from app.services.learning_intelligence.clients.lia_client import get_guidance_for_turn_sync

            lia = get_guidance_for_turn_sync(
                student_user_id=int(student_key),
                query=effective_query,
                topic=conv.resolved_topic or effective_query,
                subject_name=subject_name,
                chapter=chapter,
                chapter_ids=chapter_ids,
                class_level=class_level,
                board=board,
                agent_mode=agent_mode,
                understanding_scores=understanding_scores,
            )
            if lia and lia.get("prompt_instructions"):
                messages[0]["content"] = messages[0]["content"] + "\n\n" + lia["prompt_instructions"]
    else:
        messages = _build_chat_messages(
            effective_query,
            context,
            class_level=class_level,
            board=board,
            subject_name=subject_name,
            chapter=chapter,
            student_name=student_name,
            section_instruction=section_instruction,
            heading_scope=scope,
            conversation_history=recent_hist,
            conversation_memory=session_mem,
            chapter_coverage_guidance=coverage_guidance,
            learner_snapshot=learner_snapshot,
            understanding_scores=understanding_scores,
            resolved_topic=conv.resolved_topic or effective_query,
            agent_mode=agent_mode,
            student_key=student_key,
            chapter_ids=chapter_ids,
        )

    last_imgs: list[dict] = []
    images_emitted = False
    early_image_min = VOICE_EARLY_IMAGE_MIN_CHARS if voice_live_teaching else EARLY_IMAGE_MIN_CHARS
    answer_type = _resolve_answer_type(
        effective_query,
        subject_name=subject_name,
        conversation_history=recent_hist,
        chapter=chapter,
        conv=conv,
    )
    if _normalize_agent_mode(agent_mode) == "practice":
        answer_type = "quiz"
    elif _normalize_agent_mode(agent_mode) in ("ask", "explain") and answer_type in ("quiz", "mcq"):
        answer_type = "explanation"
    voice_token_limit = VOICE_MAX_TOKENS if voice_live_teaching else None
    if voice_token_limit is None:
        voice_token_limit = _mistral_token_limit_for_answer_type(
            answer_type, heading_scope=scope, subject_name=subject_name
        )

    async def _retrieve_images_for_answer(answer_text: str) -> list[dict]:
        if not chapter_ids or not img_allowed:
            return []
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    _fetch_related_images,
                    scope,
                    collection_name=collection_name,
                    chapter_ids=chapter_ids,
                    chapter_names=chapter_names or [],
                    chapter=chapter,
                    query=query,
                    docs=docs,
                    conversation_history=recent_hist,
                    top_n=img_top_n,
                ),
                timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Image retrieval timed out after %.0fs (stream end)",
                IMAGE_RETRIEVAL_TIMEOUT_SEC,
            )
            return []
        except Exception as exc:
            logger.warning("Image retrieval failed (stream end): %s", exc)
            return []

    # Bootstrap image search from retrieved textbook context while the LLM streams.
    from app.services.image_service.textbook_image_retrieval import _context_excerpt_from_docs

    bootstrap_ctx = _context_excerpt_from_docs(docs)[:2000]
    img_task: asyncio.Task[list[dict]] | None = None
    early_img_task: asyncio.Task[list[dict]] | None = None
    emit_early_task: asyncio.Task[None] | None = None
    if chapter_ids and img_allowed:
        early_img_task = asyncio.create_task(
            asyncio.to_thread(
                early_related_images_for_query,
                chapter_ids,
                docs,
                query,
                top_n=img_top_n,
                conversation_history=recent_hist,
                chapter_single=chapter,
            )
        )
        img_task = asyncio.create_task(_retrieve_images_for_answer(bootstrap_ctx))

    async def _emit_early_images_when_ready() -> None:
        nonlocal images_emitted, last_imgs
        if early_img_task is None or not emit_related_images or not img_allowed:
            return
        try:
            imgs = await early_img_task
        except Exception:
            imgs = []
        if imgs and not images_emitted:
            last_imgs = imgs
            _log_image_stage("early_emit", imgs)
            await emit_related_images(imgs)
            images_emitted = True

    if chapter_ids and img_allowed and early_img_task is not None:
        emit_early_task = asyncio.create_task(_emit_early_images_when_ready())

    async def _cancel_bg_tasks() -> None:
        for task in (emit_early_task, img_task, early_img_task):
            if task is not None and not task.done():
                task.cancel()
        for task in (emit_early_task, img_task, early_img_task):
            if task is None:
                continue
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def _maybe_emit_bootstrap_images(answer_so_far: str = "") -> None:
        nonlocal images_emitted, last_imgs
        if images_emitted or not emit_related_images or not img_allowed:
            return
        if early_img_task is not None and early_img_task.done():
            try:
                last_imgs = early_img_task.result()
            except Exception:
                last_imgs = []
            if last_imgs:
                _log_image_stage("early_emit", last_imgs)
                await emit_related_images(last_imgs)
                images_emitted = True
                return
        min_chars = early_image_min
        if len(answer_so_far) < min_chars:
            return
        if img_task is None or not img_task.done():
            return
        try:
            last_imgs = img_task.result()
        except Exception:
            last_imgs = []
        if last_imgs:
            _log_image_stage("bootstrap_emit", last_imgs)
            await emit_related_images(last_imgs)
            images_emitted = True

    try:
        full_answer: list[str] = []
        async for token in _stream_mistral_async(
            messages,
            max_tokens=voice_token_limit,
            feature="voice" if voice_mode else "chat",
        ):
            full_answer.append(token)
            await _maybe_emit_bootstrap_images("".join(full_answer))
            yield token
        answer_text = "".join(full_answer)
        # Empty LLM stream → textbook snippet so the UI never shows a blank bubble.
        if not (answer_text or "").strip():
            fb = _best_chunk_fallback(effective_query, docs)
            if fb.strip():
                answer_text = fb
                yield fb
            else:
                answer_text = ANSWER_NOT_IN_CHAPTER
                yield ANSWER_NOT_IN_CHAPTER
        if not voice_live_teaching:
            original_len = len(answer_text)
            try:
                answer_text = await _finalize_direct_answer(
                    messages,
                    answer_text,
                    effective_query,
                    answer_type=answer_type,
                    heading_scope=scope,
                    conversation_history=recent_hist,
                    subject_name=subject_name,
                )
                if emit_clean_answer and answer_text and len(answer_text) != original_len:
                    await emit_clean_answer(answer_text)
                elif len(answer_text) > original_len:
                    supplement = answer_text[original_len:]
                    if supplement:
                        yield supplement
                elif len(answer_text) < original_len:
                    # ponytail: shrink replaced streamed text — client keeps full stream today;
                    # cache stores the shorter final answer.
                    pass
            except Exception as exc:
                logger.warning("Stream answer finalize failed: %s", exc)
        math_lesson = None
        science_experiment = None
        should_finalize_math = use_text_format or (
            voice_mode and _is_mathematics_subject(subject_name)
        )
        should_finalize_science = use_text_format or (
            voice_mode and _is_science_subject(subject_name)
        )
        if should_finalize_math:
            from app.services.chapter_scope import detect_chapter_scope_choice

            is_scope_choice = (
                detect_chapter_scope_choice(effective_query, conversation_history) is not None
            )
            allow_fallback = not is_scope_choice and (use_text_format or voice_live_teaching)
            streamed_prose = answer_text
            clean_answer, math_lesson = finalize_math_answer(
                answer_text,
                effective_query,
                class_level=class_level,
                subject_name=subject_name,
                allow_fallback=allow_fallback,
                conversation_history=recent_hist,
            )
            # Never ship an empty clean_answer — incomplete ```math-lesson fences can wipe prose.
            display_answer = (clean_answer or "").strip() or streamed_prose
            if math_lesson and emit_math_lesson:
                await emit_math_lesson(math_lesson, display_answer)
            if use_text_format or voice_live_teaching:
                answer_text = display_answer
        if should_finalize_science:
            streamed_prose = answer_text
            clean_answer, science_experiment = finalize_science_answer(
                answer_text,
                effective_query,
                class_level=class_level,
                subject_name=subject_name,
            )
            display_answer = (clean_answer or "").strip() or streamed_prose
            if science_experiment and emit_science_experiment:
                await emit_science_experiment(science_experiment, display_answer)
            if use_text_format or (voice_mode and _is_science_subject(subject_name)):
                answer_text = display_answer
        if img_task is not None and not img_task.done():
            img_task.cancel()
            try:
                await img_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        final_imgs = await _retrieve_images_for_answer(answer_text)
        last_imgs = final_imgs
        _log_image_stage("final_emit", final_imgs)
        if emit_related_images:
            await emit_related_images(final_imgs)
        if answer_text and not voice_live_teaching:
            if student_key and understanding_scores:
                await _mentor_profile_after_turn(
                    student_key,
                    conv.resolved_topic or effective_query,
                    understanding_scores,
                )
                if student_key.isdigit():
                    from app.services.learning_intelligence.clients.lia_client import (
                        emit_chat_assistant_response,
                    )

                    emit_chat_assistant_response(
                        student_user_id=int(student_key),
                        subject_name=subject_name,
                        chapter=chapter,
                        topic=conv.resolved_topic or effective_query,
                        agent_mode=agent_mode,
                    )
    except FileNotFoundError:
        fb = _best_chunk_fallback(query, docs)
        last_imgs = await _retrieve_images_for_answer(fb)
        if emit_related_images:
            await emit_related_images(last_imgs)
        yield fb
    except Exception as exc:
        logger.error("Mistral stream failed: %s: %s", type(exc).__name__, exc)
        fb = _best_chunk_fallback(query, docs)
        last_imgs = await _retrieve_images_for_answer(fb)
        if emit_related_images:
            await emit_related_images(last_imgs)
        yield fb
    finally:
        await _cancel_bg_tasks()


# ── Legacy compatibility (used by /upload + /chat in main.py) ────────────────

def get_qa_chain(vectorstore):
    """
    Backward-compatible wrapper for the old main.py ``/chat`` endpoint.
    Returns a sync-compatible object whose ``.run()`` blocks the caller
    (acceptable only for the legacy /chat route).
    """

    class _QARunnable:
        def run(self, query: str) -> str:
            import asyncio

            retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
            docs = retriever.invoke(query)

            if isinstance(vectorstore, InMemoryDocVectorStore):
                def _page(d):
                    m = getattr(d, "metadata", None) or {}
                    try:
                        return int(m.get("page", 0) or 0)
                    except Exception:
                        return 0
                docs = sorted(docs, key=_page)

            context = _join_context_within_budget(docs)
            prompt = PROMPT.format(context=context, question=query)

            # Run the async Mistral call in a new event loop (legacy sync path)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        legacy_msgs = [{"role": "user", "content": prompt}]
                        future = pool.submit(asyncio.run, _call_mistral_async(legacy_msgs))
                        return future.result(timeout=90)
                return loop.run_until_complete(
                    _call_mistral_async([{"role": "user", "content": prompt}])
                )
            except Exception:
                return _best_chunk_fallback(query, docs)

    return _QARunnable()
