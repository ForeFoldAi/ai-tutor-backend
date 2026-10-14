"""Answer / question type detection and agent-mode routing."""
from __future__ import annotations

import re
from typing import Any

logger = __import__("logging").getLogger(__name__)


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
    # Calculate/solve needs room for steps — never treat as 160-token direct answer
    if _PROBLEM_SOLVING_PATTERNS.search(q):
        return "stepwise"
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


# Define letters/symbols the first time they appear (R, P, A, B, t, …).
_DEFINE_SYMBOLS_RULE = (
    "Whenever you use a letter, symbol, or short form in an explanation or formula "
    "(for example R, P, A, B, t, F, V), say what it means in plain words the first time "
    "you use it — e.g. \"R is the puffing rate\", \"P is pressure\", \"A is area\". "
    "Do not leave students guessing what a letter stands for."
)


_SUBJECT_GUIDELINES: dict[str, str] = {
    "Science": (
        "Emphasize evidence, observations, experiments, and cause-effect relationships.\n"
        f"- {_DEFINE_SYMBOLS_RULE}"
    ),
    "Mathematics": (
        "You are an expert Mathematics Tutor for school students.\n"
        "Your goal is not just to give answers but to help students understand mathematical "
        "concepts and solve similar problems independently.\n\n"
        "Additional rules:\n"
        "- Use simple, student-friendly language adapted to the student's grade level.\n"
        "- Never provide only the final answer.\n"
        "- Explain the reasoning behind every operation.\n"
        f"- {_DEFINE_SYMBOLS_RULE}\n"
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
    f"- {_DEFINE_SYMBOLS_RULE}\n"
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
    f"- {_DEFINE_SYMBOLS_RULE}\n"
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

    Voice uses live conversational teaching by default. Full written format only
    when the student explicitly asks for more detail or a complete written solution.
    """
    if not voice_mode:
        return True
    _ = heading_scope
    from app.services.voice_tutor import voice_expand_requested, voice_wants_full_written_answer

    scores = understanding_scores or {}
    if scores.get("wants_expansion") or voice_expand_requested(query):
        return True
    if voice_wants_full_written_answer(query):
        return True
    return False


_MATH_DIALOGUE_TYPES = frozenset({"greeting", "affirmation", "personal-response", "clarification"})


_MATH_SHORT_TYPES = frozenset({"brief", "one-word"})


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


_FULL_STRUCTURE_TYPES = frozenset({"exam-format"})


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
        "MANDATORY: Write at least {min_words} words. Use the FULL five-section exam format with **bold** headers (NO emoji): "
        "**Concept Overview**, **Detailed Explanation**, **Real-Life Example**, "
        "**Key Points to Remember** (3-5 bullets), **Quick Check**."
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
        "The student confirmed they understood, thanked you, or briefly acknowledged "
        "your PREVIOUS explanation (see conversation above). "
        "Write 2-3 short sentences only (about 35-55 words). "
        "Do NOT teach new facts or repeat the prior explanation. "
        "Do NOT say 'Welcome back' or 'What would you like to learn today?'. "
        "Use this shape: warm acknowledgment (e.g. 'Great!' / 'You're welcome!') + "
        "'Now that you know what [topic] is…' + ONE question with clear choices: "
        "real-life example, quick quiz, next part of this topic (name it briefly), "
        "or explore other topics / wrap up. "
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
        "The student is greeting or making small talk. Reply WARMLY and BRIEFLY "
        "(2-3 sentences max) in a friendly teacher tone. "
        "If conversation history shows you were already teaching, briefly acknowledge "
        "and offer a next step on that topic — do NOT say 'Welcome back' or restart "
        "with 'What would you like to learn today?'. "
        "Only on a fresh session opener: greet by first name and invite a topic from "
        "{subject} or {chapter}. "
        "Never become a social chatbot. Always stay as a helpful tutor."
    ),
}


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
    dialogue_act: str | None = None,
) -> str:
    """Shared answer-type routing for prompts, token limits, and post-processing."""
    from app.services.chat_service.dialogue import (
        _is_accepting_tutor_continue_offer,
        _is_affirmation_followup,
        _is_personal_dialogue_response,
    )
    from app.services.conversation_context import (
        ResponseMode,
        answer_type_for_followup,
        is_clarification_followup,
        resolve_conversation_context,
    )
    from app.services.conversation_intent_classifier import FollowupType

    act = (dialogue_act or "").strip().lower()
    if act == "intro":
        return "greeting"
    if act in ("closing", "ack"):
        return "affirmation"

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

