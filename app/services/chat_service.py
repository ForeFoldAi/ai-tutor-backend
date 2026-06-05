"""
AI Tutor chat service.

Key improvements over the original:
- Fully async: uses httpx.AsyncClient instead of requests (no event-loop blocking)
- Streaming: stream_chapter_qa() yields tokens for StreamingResponse
- Grade-aware prompt: adapts language complexity to class level (1-12)
- Tiered answer format: direct prose by default; full emoji sections only for exam/long requests
- Answer-type detection: honors brief/one-word and question intent (not every "what is" → essay)
- Redis cache: repeated questions answered instantly at zero LLM cost
- Fallback: keyword-chunk answer when Mistral key is missing
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, AsyncIterator

import httpx
from langchain_core.prompts import PromptTemplate

from app.config import (
    CONTEXT_CHAR_BUDGET,
    IMAGE_RETRIEVAL_TIMEOUT_SEC,
    MISTRAL_API_KEY,
    MISTRAL_MAX_TOKENS,
    MISTRAL_MODEL,
    MISTRAL_TEMPERATURE,
    RETRIEVAL_K,
    TOP_RELATED_IMAGES,
)
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
    r"can you help|are you (there|ready|a bot|ai|real))\b",
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
    r"\b(bullet points?|list (the|all|some)|points? (on|about)|"
    r"give points|write points)\b",
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
    # Honor explicit short/brief requests before "what is …" → full teaching mode
    if _EXPLICIT_SHORT.search(q) or _EXPLICIT_BRIEF.search(q):
        return "brief"
    # Simple "what is X?" → direct answer; deep explain / exam → longer formats
    if _CONCEPT_STARTS.match(q):
        if _EXPLAIN_PATTERNS.search(q) or _EXAM_PATTERNS.search(q):
            return "paragraph"
        return "short-answer"
    if _EXPLAIN_PATTERNS.search(q):
        return "paragraph"
    return "short-answer"


_COMPACT_TYPES = frozenset({"greeting", "one-word", "brief"})
# Only exam-style questions use the five emoji section template
_FULL_STRUCTURE_TYPES = frozenset({"exam-format"})


def _structure_tier(answer_type: str) -> str:
    if answer_type in _COMPACT_TYPES:
        return "compact"
    if answer_type in _FULL_STRUCTURE_TYPES:
        return "full"
    return "direct"


_ANSWER_MIN_WORDS: dict[str, int] = {
    "one-word": 1,
    "brief": 15,
    "greeting": 40,
    "short-answer": 70,
    "concept": 100,
    "definition": 100,
    "paragraph": 120,
    "stepwise": 150,
    "bullet-points": 100,
    "simplified": 90,
    "exam-format": 280,
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
        "Write at least {min_words} words in clear flowing paragraphs. "
        "Define the idea, explain why it matters, give one real-life example. "
        "NO emoji section headers (no 🌱 Concept Overview, etc.). "
        "You may use a short bullet list only if it genuinely helps."
    ),
    "definition": (
        "Write at least {min_words} words: definition first, then a short explanation and one example. "
        "Plain prose only — NO emoji section headers."
    ),
    "stepwise": (
        "Write at least {min_words} words. Explain step by step (Step 1, Step 2, ...). "
        "Plain headings only — NO emoji section headers. End with one short check question."
    ),
    "paragraph": (
        "Write at least {min_words} words in 2-4 clear paragraphs. "
        "Cover the main idea, how it works, and one example. "
        "NO emoji section headers. No padded sections."
    ),
    "exam-format": (
        "MANDATORY: Write at least {min_words} words. Use the FULL five-section exam format with emoji headers: "
        "🌱 Concept Overview, 📚 Detailed Explanation, 🌍 Real-Life Example, "
        "📝 Key Points to Remember (3-5 bullets), ❓ Quick Check."
    ),
    "simplified": (
        "Write at least {min_words} words using the simplest everyday words. "
        "2-3 short paragraphs, one example — NO emoji section headers."
    ),
    "bullet-points": (
        "Write at least {min_words} words. Lead with a one-sentence intro, then use a clear bullet list "
        "for the main points and one short closing sentence. NO emoji section headers."
    ),
    "short-answer": (
        "Write a clear, direct answer of about {min_words} words (roughly 2-4 short paragraphs). "
        "Answer the question first, then add one helpful example if useful. "
        "NO emoji section headers, NO Quick Check section, NO five-part template. "
        "Do NOT use labels like 'Concept Overview' or 'Key Points to Remember'."
    ),
    "greeting": (
        "The student is greeting or making small talk. Greet them personally by first name. "
        "Reply WARMLY and BRIEFLY (2-3 sentences max) in a friendly teacher tone. "
        "Example: 'Hello {student_name}! Welcome back. I'm your AI Tutor. "
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
    if names:
        if len(names) == 1:
            scope = f"about {names[0]} in {subject}"
        elif len(names) == 2:
            scope = f"about {names[0]} and {names[1]} in {subject}"
        else:
            scope = f"about {names[0]}, {names[1]}, and more in {subject}"
    else:
        scope = f"from your {subject} textbook"
    return (
        f"Hello {first}! Welcome back. I'm your AI Tutor. "
        f"What would you like to learn today? "
        f"Feel free to ask any question {scope}, from your homework, or about topics you're curious about."
    )


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
You are an AI Tutor designed specifically for school students.
Your purpose is not only to answer questions but to help students understand concepts deeply —
like a caring teacher explaining step by step.

STUDENT CONTEXT:
- Student Name: {student_name}
- Class/Grade: {grade_label}
- Board: {board}
- Subject: {subject}
- Chapter: {chapter}

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

TEXTBOOK PRIORITY:
- Answer using the chapter context provided in the user message FIRST.
- If insufficient, use accurate general educational knowledge and say:
  "This is a general explanation as it is not covered in detail in this chapter."
- Never contradict the textbook curriculum.
- Never invent textbook-specific facts (dates, names, formulas, page numbers).
- Never mix content from other chapters or subjects.

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
- Be clear and complete, but do not pad with extra sections the student did not ask for."""

_LENGTH_POLICY_LONG = """\
RESPONSE LENGTH — CRITICAL (read carefully):
- For this answer you MUST write at least {min_words} words (count before finishing).
- Never stop after only a definition, opener (e.g. "Great question!"), or one short paragraph.
- If chapter context is short, expand with accurate general teaching — do not stay brief.
- Default range: 100-500 words. Focus on understanding, not memorization."""

_LENGTH_POLICY_DIRECT = """\
RESPONSE LENGTH:
- Write about {min_words} words (roughly 70-150 words unless the question needs more detail).
- Be direct: answer the question in the first paragraph.
- Do NOT force a five-section essay format."""

_LENGTH_POLICY_BRIEF = """\
RESPONSE LENGTH — CRITICAL:
- The student asked for a SHORT answer. Keep it to 2-4 sentences (about 25-60 words).
- Do NOT use the five-section teaching format or emoji headers.
- Do NOT add bullet lists, examples sections, or a Quick Check question."""

_INTERACTION_LONG = """\
INTERACTIVE TEACHING (after every non-greeting answer):
1. End with ONE follow-up question to check understanding (e.g. "Does this make sense so far?").
2. Encourage curiosity and invite doubts briefly."""

_INTERACTION_BRIEF = """\
INTERACTIVE TEACHING:
- Do NOT add a follow-up question — the student asked for a short answer only."""

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
) -> list[dict[str, str]]:
    from app.services.section_heading import HeadingScope

    answer_type = detect_answer_type(query)
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
        length_policy = _LENGTH_POLICY_BRIEF if answer_type in ("brief", "one-word") else (
            "RESPONSE LENGTH: 2-3 sentences only."
        )
        interaction_policy = _INTERACTION_BRIEF if answer_type in ("brief", "one-word") else ""
        explanation_structure = ""
        user_closing = (
            f"Write your SHORT answer now ({answer_type} — 2-4 sentences max, no sections):"
            if answer_type in ("brief", "one-word")
            else "Write your greeting now:"
        )
    elif tier == "full":
        length_policy = _LENGTH_POLICY_LONG.format(min_words=min_words)
        interaction_policy = _INTERACTION_LONG
        explanation_structure = _EXPLANATION_STRUCTURE_FULL
        user_closing = (
            f"Write your complete exam-style answer now ({answer_type}, minimum {min_words} words, "
            "all five emoji sections):"
        )
    else:
        length_policy = _LENGTH_POLICY_DIRECT.format(min_words=min_words)
        interaction_policy = _INTERACTION_LONG
        explanation_structure = _EXPLANATION_STRUCTURE_DIRECT
        user_closing = (
            f"Write your direct answer now ({answer_type}, about {min_words} words, "
            "plain prose only — no emoji section headers):"
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
        answer_instruction = (
            "Use the mandatory main-section format in TEXTBOOK SCOPE: "
            "one **bold subtopic** heading per instrument/topic, then • bullet points."
        )
        user_closing = (
            "Write the answer now using **bold subtopic headings** and • bullets for EACH subtopic "
            "listed in TEXTBOOK SCOPE (in order). Do not use a single paragraph."
        )

    system = _SYSTEM_PROMPT_TEMPLATE.format(
        student_name=display_name,
        grade_label=grade_label,
        board=board or "General",
        subject=subject_name or "General",
        chapter=chapter or "Current chapter",
        complexity_rule=complexity,
        class_band_rule=class_band,
        length_policy=length_policy,
        answer_instruction=instruction,
        interaction_policy=interaction_policy,
        explanation_structure=explanation_structure,
    )
    section_block = ""
    if section_instruction.strip():
        section_block = f"\n\nTEXTBOOK SCOPE:\n{section_instruction.strip()}\n"

    user = _USER_PROMPT_TEMPLATE.format(
        context=context or "(No chapter text retrieved — use accurate general knowledge.)",
        question=query,
        user_closing=user_closing,
    )
    if section_block:
        user = section_block + user
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


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
    if not MISTRAL_API_KEY:
        raise FileNotFoundError(
            "Missing MISTRAL_API_KEY env var. Set it to enable AI answers."
        )


async def _call_mistral_async(messages: list[dict[str, str]]) -> str:
    """Non-blocking Mistral chat completion via httpx.AsyncClient."""
    _ensure_mistral_config()
    logger.debug("Mistral request model=%s max_tokens=%d", MISTRAL_MODEL, MISTRAL_MAX_TOKENS)

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": messages,
                "max_tokens": MISTRAL_MAX_TOKENS,
                "temperature": MISTRAL_TEMPERATURE,
            },
        )
        resp.raise_for_status()

    payload: dict[str, Any] = resp.json()
    text = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    logger.debug("Mistral response length=%d chars", len(text))
    return text or "The answer is not found in the document."


async def _stream_mistral_async(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    """Yield tokens from Mistral SSE stream for use with StreamingResponse."""
    _ensure_mistral_config()
    logger.debug("Mistral streaming request model=%s", MISTRAL_MODEL)

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": messages,
                "max_tokens": MISTRAL_MAX_TOKENS,
                "temperature": MISTRAL_TEMPERATURE,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue


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
        return "The answer is not found in the document."

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
    student_name: str = "",
) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks from ChromaDB and answer via Mistral (async).

    Returns ``(answer_text, related_images)``. Checks Redis cache first.
    """
    from app.core.cache import (
        deserialize_tutor_cache,
        get_cached_answer,
        set_cached_answer,
    )
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.section_retrieval import retrieve_for_tutor_query

    conv = resolve_conversation_context(
        query, conversation_history=conversation_history, chapter=chapter
    )
    retrieval_query = conv.retrieval_query or query

    cached = await get_cached_answer(collection_name, chapter_ids, query)
    if cached:
        answer, _ = deserialize_tutor_cache(cached)
        # Images are never stored in cache — re-rank fresh per query.
        imgs: list[dict] = []
        docs_for_imgs, scope, _ = retrieve_for_tutor_query(
            retrieval_query,
            collection_name=collection_name,
            chapter_ids=chapter_ids,
        )
        img_allowed = should_retrieve_images(
            conv, chapter_ids=chapter_ids, heading_scope_kind=scope.kind
        )
        if chapter_ids and img_allowed:
            img_top_n = _image_top_n_for_scope(scope, docs=docs_for_imgs)
            if docs_for_imgs:
                try:
                    imgs = await asyncio.wait_for(
                        asyncio.to_thread(
                            _fetch_related_images,
                            scope,
                            collection_name=collection_name,
                            chapter_ids=chapter_ids,
                            chapter_names=chapter_names or [],
                            chapter=chapter,
                            query=query,
                            docs=docs_for_imgs,
                            conversation_history=conversation_history,
                            top_n=img_top_n,
                        ),
                        timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
                    )
                except Exception as exc:
                    logger.warning("Image retrieval on cache hit failed: %s", exc)
        return answer, imgs

    docs, scope, section_instruction = retrieve_for_tutor_query(
        retrieval_query,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
    )
    img_allowed = should_retrieve_images(
        conv, chapter_ids=chapter_ids, heading_scope_kind=scope.kind
    )
    if not docs:
        return "The answer is not found in the document.", []

    context = _join_context_within_budget(docs)
    messages = _build_chat_messages(
        query,
        context,
        class_level=class_level,
        board=board,
        subject_name=subject_name,
        chapter=chapter,
        student_name=student_name,
        section_instruction=section_instruction,
        heading_scope=scope,
    )
    img_top_n = _image_top_n_for_scope(scope, docs=docs)

    try:
        answer = await _call_mistral_async(messages)
        answer = await _expand_short_answer(messages, answer, query, heading_scope=scope)
    except FileNotFoundError:
        answer = _best_chunk_fallback(query, docs)
    except Exception as exc:
        logger.error("Mistral call failed: %s: %s", type(exc).__name__, exc)
        answer = _best_chunk_fallback(query, docs)

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
                    query=query,
                    docs=docs,
                    conversation_history=conversation_history,
                    top_n=img_top_n,
                ),
                timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Image retrieval timed out after %.0fs for query=%r",
                IMAGE_RETRIEVAL_TIMEOUT_SEC,
                query[:80],
            )
        except Exception as exc:
            logger.warning("Image retrieval failed: %s", exc)

    await set_cached_answer(
        collection_name,
        chapter_ids,
        query,
        answer,
        related_images=related,
    )
    return answer, related


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
    conversation_history: list[dict] | None = None,
    student_name: str = "",
    voice_mode: bool = False,
    tutor_state: str = "TEACHING",
    understanding_scores: dict | None = None,
    learner_snapshot: dict | None = None,
) -> AsyncIterator[str]:
    """
    Streaming version of chapter_aware_qa.

    Optionally invokes *emit_related_images* when ranked images are ready
    (usually during the first tokens, without blocking retrieval).
    """
    from app.config import VOICE_CONTEXT_CHAR_BUDGET, VOICE_RETRIEVAL_K
    from app.core.cache import (
        deserialize_tutor_cache,
        get_cached_answer,
        set_cached_answer,
    )
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.image_service.textbook_image_retrieval import early_related_images_for_query, related_images_for_query
    from app.services.section_retrieval import retrieve_for_tutor_query

    retrieval_k = VOICE_RETRIEVAL_K if voice_mode else RETRIEVAL_K
    context_budget = VOICE_CONTEXT_CHAR_BUDGET if voice_mode else CONTEXT_CHAR_BUDGET

    conv = resolve_conversation_context(
        query, conversation_history=conversation_history, chapter=chapter
    )
    retrieval_query = conv.retrieval_query or query

    cached = None if voice_mode else await get_cached_answer(collection_name, chapter_ids, query)
    if cached:
        answer, _ = deserialize_tutor_cache(cached)
        # Images are never stored in cache — always retrieve fresh so they match
        # this specific query rather than a previous one with a similar answer.
        imgs: list[dict] = []
        docs_for_imgs, scope, _ = retrieve_for_tutor_query(
            retrieval_query,
            collection_name=collection_name,
            chapter_ids=chapter_ids,
            k=retrieval_k,
        )
        img_allowed = should_retrieve_images(
            conv, chapter_ids=chapter_ids, heading_scope_kind=scope.kind
        )
        if chapter_ids and img_allowed:
            img_top_n = _image_top_n_for_scope(scope, docs=docs_for_imgs)
            if docs_for_imgs:
                try:
                    imgs = await asyncio.wait_for(
                        asyncio.to_thread(
                            _fetch_related_images,
                            scope,
                            collection_name=collection_name,
                            chapter_ids=chapter_ids,
                            chapter_names=chapter_names or [],
                            chapter=chapter,
                            query=query,
                            docs=docs_for_imgs,
                            conversation_history=conversation_history,
                            top_n=img_top_n,
                        ),
                        timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
                    )
                except Exception as exc:
                    logger.warning("Image retrieval on cache hit failed: %s", exc)
        _log_image_stage("cache_hit_emit", imgs)
        if emit_related_images:
            await emit_related_images(imgs)
        yield answer
        return

    docs, scope, section_instruction = retrieve_for_tutor_query(
        retrieval_query,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        k=retrieval_k,
    )
    img_allowed = should_retrieve_images(
        conv, chapter_ids=chapter_ids, heading_scope_kind=scope.kind
    )
    img_top_n = _image_top_n_for_scope(scope, docs=docs)
    if not docs:
        if emit_related_images:
            await emit_related_images([])
        yield "The answer is not found in the document."
        return

    if emit_related_images and not img_allowed:
        await emit_related_images([])

    context = _join_context_within_budget(docs, char_budget=context_budget)

    if voice_mode:
        from app.services.voice_tutor import (
            TutorState,
            UnderstandingScores,
            LearnerProfileSnapshot,
            build_voice_mistral_messages,
            voice_expand_requested,
        )

        scores = UnderstandingScores(**understanding_scores) if understanding_scores else UnderstandingScores()
        learner = LearnerProfileSnapshot(**learner_snapshot) if learner_snapshot else None
        try:
            state = TutorState(tutor_state)
        except ValueError:
            state = TutorState.TEACHING
        messages = build_voice_mistral_messages(
            query,
            context,
            class_level=class_level,
            board=board,
            subject_name=subject_name,
            chapter=chapter,
            student_name=student_name,
            conversation_history=conversation_history,
            tutor_state=state,
            understanding=scores,
            learner=learner,
            expand_deep=voice_expand_requested(query),
        )
    else:
        messages = _build_chat_messages(
            query,
            context,
            class_level=class_level,
            board=board,
            subject_name=subject_name,
            chapter=chapter,
            student_name=student_name,
            section_instruction=section_instruction,
            heading_scope=scope,
        )

    last_imgs: list[dict] = []
    images_emitted = False

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
                    conversation_history=conversation_history,
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
    if chapter_ids and img_allowed:
        early_img_task = asyncio.create_task(
            asyncio.to_thread(
                early_related_images_for_query,
                chapter_ids,
                docs,
                query,
                top_n=img_top_n,
                conversation_history=conversation_history,
                chapter_single=chapter,
            )
        )
        img_task = asyncio.create_task(_retrieve_images_for_answer(bootstrap_ctx))

    async def _maybe_emit_bootstrap_images(answer_so_far: str = "") -> None:
        nonlocal images_emitted, last_imgs
        if images_emitted or not emit_related_images or not img_allowed:
            return
        from app.config import EARLY_IMAGE_MIN_CHARS

        min_chars = 60 if voice_mode else EARLY_IMAGE_MIN_CHARS
        if len(answer_so_far) < min_chars:
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
        async for token in _stream_mistral_async(messages):
            full_answer.append(token)
            await _maybe_emit_bootstrap_images("".join(full_answer))
            yield token
        answer_text = "".join(full_answer)
        if not voice_mode:
            original_len = len(answer_text)
            try:
                answer_text = await _expand_short_answer(
                    messages, answer_text, query, heading_scope=scope
                )
                if len(answer_text) > original_len:
                    supplement = answer_text[original_len:]
                    if supplement:
                        yield supplement
            except Exception as exc:
                logger.warning("Stream answer expansion failed: %s", exc)
        final_imgs = await _retrieve_images_for_answer(answer_text)
        last_imgs = final_imgs
        _log_image_stage("final_emit", final_imgs)
        if emit_related_images:
            await emit_related_images(final_imgs)
        if answer_text and not voice_mode:
            await set_cached_answer(
                collection_name,
                chapter_ids,
                query,
                answer_text,
                related_images=last_imgs,
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
