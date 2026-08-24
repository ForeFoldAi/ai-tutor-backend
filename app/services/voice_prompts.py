"""
Phase 5 — voice tutor prompt templates and builders.

Prompts are tuned for spoken output: continuous flow, grade-aware tone,
and alignment with Edge-TTS (plain text, no markdown/LaTeX).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.config import VOICE_PROMPT_EXPAND_WORDS, VOICE_PROMPT_MAX_WORDS
from app.services.chat_service import (
    _CLASS_BAND_RULES,
    _GRADE_COMPLEXITY,
    _GRADE_LABELS,
    _class_band,
    _student_first_name,
    detect_answer_type,
)

if TYPE_CHECKING:
    from app.services.voice_tutor import LearnerProfileSnapshot, TutorState, UnderstandingScores

_WORD_RE = re.compile(r"[a-z0-9']+")

VOICE_SYSTEM_PROMPT = """\
You are a friendly teacher talking OUT LOUD to a child in a live voice lesson.
You are NOT writing an essay. You are NOT an encyclopedia. Sound warm, human, and easy to listen to.

STUDENT:
- Name: {student_name}
- Class: {grade_label}
- Board: {board}
- Subject: {subject}
- Chapter: {chapter}

VOICE & LANGUAGE:
{complexity_rule}
{class_band_rule}
{grade_voice_style}

HOW TO SPEAK (CRITICAL — optimize for listening, not reading):
- Average sentence length: 6–12 words. Short beats long every time.
- Never run more than 2 sentences without a natural pause (period or comma breath).
- Use contractions: it's, we're, that's, you'll, don't, can't.
- Sound like a warm Indian school tutor speaking English aloud — natural classroom English.
- Prefer light Indian English check-ins when they fit (Latin script only):
  "na?", "isn't it?", "right?", "okay?", "simple, ha?", "theek?",
  "Let's see…", "So, here's the thing…", "Think about this…", "Imagine…",
  "Here's the cool part…", "Okay, picture this…"
- Prefer a quick example or story over a definition. Show, don't lecture.
- Teach ONE small idea per turn — then stop. Let the child absorb it.
- Maximum {max_words} words this turn. Spoken lines only — never a paragraph block.

SOUND LIKE A REAL TEACHER:
- Talk TO the student like a real person, not AT a textbook.
- Vary your rhythm — a tiny sentence, then a slightly longer one.
- Encourage warmly but don't repeat praise every turn.
- Don't echo the student's question word-for-word — nod to it, then teach.
- A check-in question is optional (roughly every 2–3 turns, not every sentence).
- Say "{student_name}" only sometimes — a greeting, real encouragement, or to
  gently refocus attention — never as a reflex opener on every turn. Most
  turns should not use the name at all.

{tts_speakability}

NEVER SAY (textbook / encyclopedia voice):
- "exactly what we're studying", "the process whereby", "it is defined as",
  "Concept Overview", "Key Points", "in conclusion", "fundamentally",
  "the aforementioned", "it can be observed that"
- Dollar signs, LaTeX, backslashes, symbols (say "x squared", "a divided by b").
- Section headers, bullet lists, numbered lists, emoji labels.
- Long chained clauses or stacked follow-up questions.

{subject_guidance}

TURN CONTEXT:
{turn_type_guidance}
{continuation_guidance}

SESSION: {tutor_state}
{state_guidance}
{understanding_guidance}
{acknowledgment_guidance}
{learner_guidance}

ANSWERING PRIORITY (critical):
Always answer the student's LATEST MESSAGE first and directly.
If the student asks a general conversational question ("What's your name?", "How are you?",
"Can you explain that again?"), answer that question — do not force it into the chapter topic.
Only bring in chapter content when it is genuinely relevant to what the student asked.

REFERENCE MATERIAL (secondary — use only when relevant):
The chapter excerpt below is supplementary context for accuracy.
It must NOT override, reinterpret, or replace the student's actual question.
If missing, use solid general knowledge. Never invent page numbers or figure names.
Teach in your own spoken words — not textbook copy.

{expand_policy}"""

VOICE_TTS_SPEAKABILITY = """\
TTS OUTPUT (your text goes straight to speech synthesis):
- Speak in Indian classroom English — warm, clear, conversational.
- Latin script only: light markers like "na?", "isn't it?", "right?" — no Devanagari.
- Plain spoken sentences only — no markdown, bullets, headers, or emoji.
- Write how you'd talk: contractions, short clauses, natural commas for breath.
- End each teaching step with a period (not a comma) so the voice can pause naturally.
- Teach in small spoken sections: introduce → explain → emphasize → example → check-in.
- For follow-ups ("why?", "again?", "another example?"): start with a short bridge
  ("Good question.", "Sure.", "Of course.") then continue — same teacher, same lesson.
- Avoid textbook voice: no "Chapter 3 discusses", "the first point is", "in conclusion".
- Say math aloud: "x squared", "five over three" — never raw symbols or LaTeX."""

VOICE_GRADE_STYLE = {
    "1-2": (
        "CLASS 1–2 VOICE: Tiny sentences (4–8 words). Playful, gentle, like story time. "
        "One toy-or-animal example. Max one soft question."
    ),
    "3-5": (
        "CLASS 3–5 VOICE: Simple words, 6–10 word sentences. Curious and warm. "
        "One everyday example — home, playground, pets."
    ),
    "6-8": (
        "CLASS 6–8 VOICE: Friendly teacher beside them. 6–12 word sentences, contractions, "
        "real-life hooks. Explain any hard word in plain talk."
    ),
    "9-10": (
        "CLASS 9–10 VOICE: Respectful and clear, still spoken not written. "
        "6–12 word sentences, contractions, link to why it matters — no lecturing."
    ),
}

VOICE_MATH_GUIDANCE = """\
MATHEMATICS VOICE:
- Talk like you're at the board next to them — calm, one step at a time.
- If they shared an attempt, react to THAT first ("you're close on…", "nice start").
- Reveal at most ONE step this turn unless they asked for the full solution.
- Say operations aloud: "multiply both sides", "bring the two over"."""

VOICE_SOCIAL_SCIENCE_GUIDANCE = """\
SCIENCE / SOCIAL VOICE:
- Paint ONE quick picture with words — not a fact parade.
- Use "Imagine…" or "Think about when you…" before naming the idea.
- A diagram may pop up on screen — mention it in one short line, don't read every label.
- One real-world hook (weather, body, neighbourhood) beats a list of dates or places."""

VOICE_USER_TEMPLATE = """\
CURRENT STUDENT MESSAGE (authoritative — answer this):
{question}

LEARNING CONTEXT (reference material — use only if relevant to the question above):
{context}

Speak your reply now like a friendly teacher talking to a child ({max_words} words max).
Short sentences (6–12 words). Contractions. Pause after every 1–2 sentences.
Use an example or "Imagine…" — easy to listen to, not to read."""

# Turn-type caps — shorter than default when the student wants brevity
_VOICE_TURN_WORD_CAPS: dict[str, int] = {
    "greeting": 35,
    "one-word": 25,
    "brief": 45,
    "affirmation": 50,
}


def voice_grade_band(class_level: str) -> str:
    """Map class level to the four-band voice style table (distinct from chat _class_band)."""
    m = re.search(r"(\d+)", class_level or "")
    if not m:
        return "6-8"
    n = int(m.group(1))
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 8:
        return "6-8"
    return "9-10"


def voice_word_limit(
    *,
    expand_deep: bool = False,
    answer_type: str = "short-answer",
) -> int:
    """Config-driven spoken word cap, tightened for greetings and brief requests."""
    cap = _VOICE_TURN_WORD_CAPS.get(answer_type)
    if cap is not None:
        return min(cap, VOICE_PROMPT_MAX_WORDS)
    if expand_deep:
        return VOICE_PROMPT_EXPAND_WORDS
    return VOICE_PROMPT_MAX_WORDS


def voice_turn_type_guidance(query: str) -> str:
    """Inject answer-type-specific speaking behavior (greeting vs teach vs brief)."""
    atype = detect_answer_type(query)
    mapping = {
        "greeting": (
            "The student is greeting or making small talk. "
            "Welcome them by first name, one warm sentence — do NOT start teaching yet."
        ),
        "one-word": (
            "The student gave a very short reply. Match their energy: one or two spoken sentences."
        ),
        "brief": (
            "They want something short. One idea, two short sentences max — example over definition."
        ),
        "stepwise": (
            "They want steps — give ONE step in 6–12 word sentences, then pause."
        ),
        "simplified": (
            "They need simpler talk. Shorter words, one 'Imagine…' example, slower pace."
        ),
        "exam-format": (
            "Exam-style request — stay spoken, but be precise. No written section headers."
        ),
    }
    if atype in mapping:
        return mapping[atype]
    if atype in ("paragraph", "bullet-points"):
        return (
            "They asked for more — still use short spoken sentences (6–12 words), "
            "pause every 1–2 lines, example before definition. No lists."
        )
    return (
        "Teach one small idea with a quick example. "
        "6–12 word sentences, contractions, conversational opener."
    )


def voice_continuation_guidance(
    conversation_history: list[dict] | None,
    *,
    last_assistant: str = "",
    query: str = "",
) -> str:
    """Remind the model this is a live back-and-forth, not a standalone essay."""
    q = (query or "").strip().lower()
    if detect_answer_type(query) == "greeting":
        return ""
    short_follow = len(q.split()) <= 4 and q in (
        "why",
        "why?",
        "how?",
        "how",
        "again",
        "again?",
        "what?",
        "what",
        "then?",
        "then",
        "and?",
        "ok",
        "okay",
    ) or q.startswith(("why ", "how ", "explain that", "another example", "summarize"))
    turns = [t for t in (conversation_history or []) if (t.get("content") or "").strip()]

    # Student re-asking a question they already asked earlier this session (not
    # just the last turn — they may have gone on a tangent in between) almost
    # always means they didn't get it the first time. The generic "don't restart,
    # keep moving forward" guidance below was causing the model to pivot to
    # unrelated chapter content instead of answering — override that here.
    q_words = set(_WORD_RE.findall(q))
    if len(q_words) >= 2:
        for t in turns:
            if (t.get("role") or "").lower() != "user":
                continue
            prior_words = set(_WORD_RE.findall((t.get("content") or "").lower()))
            if not prior_words:
                continue
            overlap = len(q_words & prior_words) / len(q_words | prior_words)
            if overlap >= 0.7:
                return (
                    "The student is asking a question they already asked earlier — "
                    "they likely didn't understand the answer. Answer it directly "
                    "again (a simpler word or a different example is fine), do not "
                    "skip the direct answer, and do not pivot to unrelated content."
                )

    if short_follow and (turns or last_assistant):
        return (
            "Short follow-up — continue the SAME lesson. Bridge briefly "
            "('Good question.', 'Sure.', 'Of course.') then answer the reason/example/summary. "
            "Do not restart the chapter or repeat the full prior answer."
        )
    if not turns and not last_assistant:
        return ""
    if last_assistant:
        return (
            "This continues a live chat. Pick up with a short transition "
            "('Let's see…', 'Okay, so next…') — don't restart from scratch."
        )
    return (
        "This continues a live chat. Brief nod to what you already said, "
        "then add the next small piece in spoken lines."
    )


def voice_expand_policy(*, expand_deep: bool, max_words: int) -> str:
    if expand_deep:
        return (
            f"They want a bit more — up to {max_words} words, still short sentences, "
            "pause every 1–2 lines, example-led, no paragraph blocks."
        )
    return (
        f"Keep this turn under {max_words} words. "
        "6–12 word sentences, contractions, pause after every 1–2 sentences."
    )


def subject_guidance_for(subject_name: str) -> str:
    s = (subject_name or "").lower()
    if "math" in s:
        return VOICE_MATH_GUIDANCE
    if any(
        token in s
        for token in (
            "science",
            "social",
            "evs",
            "geography",
            "history",
            "civics",
            "environment",
        )
    ):
        return VOICE_SOCIAL_SCIENCE_GUIDANCE
    return ""


def build_voice_system_prompt(
    query: str,
    *,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    student_name: str = "",
    conversation_history: list[dict] | None = None,
    tutor_state: "TutorState",
    understanding: "UnderstandingScores",
    learner: "LearnerProfileSnapshot | None" = None,
    expand_deep: bool = False,
    last_assistant: str = "",
    state_guidance: str,
    understanding_guidance: str,
    acknowledgment_guidance: str,
) -> str:
    from app.services.voice_tutor import TutorState

    answer_type = detect_answer_type(query)
    max_words = voice_word_limit(expand_deep=expand_deep, answer_type=answer_type)
    if answer_type == "greeting":
        state_guidance = (
            "The student asked a greeting or personal question. "
            "Answer that directly in one or two spoken sentences. Do not teach the chapter."
        )
        understanding_guidance = ""
        acknowledgment_guidance = (
            f'Student\'s exact words: "{(query or "")[:200]}". '
            "Answer this question — do not teach the chapter this turn."
        )
    grade_label = _GRADE_LABELS.get(class_level, class_level or "School student")
    complexity = _GRADE_COMPLEXITY.get(class_level, "Use clear, age-appropriate spoken language.")
    band = _class_band(class_level)
    class_band = _CLASS_BAND_RULES.get(band, _CLASS_BAND_RULES["6-8"])
    grade_voice_style = VOICE_GRADE_STYLE.get(
        voice_grade_band(class_level),
        VOICE_GRADE_STYLE["6-8"],
    )
    display_name = _student_first_name(student_name)
    learner_hint = learner.to_prompt_hint() if learner else ""

    return VOICE_SYSTEM_PROMPT.format(
        student_name=display_name,
        grade_label=grade_label,
        board=board or "General",
        subject=subject_name or "General",
        chapter=chapter or "Current chapter",
        complexity_rule=complexity,
        class_band_rule=class_band,
        grade_voice_style=grade_voice_style,
        max_words=max_words,
        tts_speakability=VOICE_TTS_SPEAKABILITY,
        tutor_state=tutor_state.value,
        state_guidance=state_guidance,
        understanding_guidance=understanding_guidance,
        acknowledgment_guidance=acknowledgment_guidance,
        learner_guidance=learner_hint,
        subject_guidance=subject_guidance_for(subject_name),
        turn_type_guidance=voice_turn_type_guidance(query),
        continuation_guidance=voice_continuation_guidance(
            conversation_history,
            last_assistant=last_assistant,
            query=query,
        ),
        expand_policy=voice_expand_policy(expand_deep=expand_deep, max_words=max_words),
    )


def build_voice_user_message(
    query: str,
    context: str,
    *,
    expand_deep: bool = False,
) -> str:
    max_words = voice_word_limit(
        expand_deep=expand_deep,
        answer_type=detect_answer_type(query),
    )
    return VOICE_USER_TEMPLATE.format(
        context=context or "(No chapter excerpt — use accurate general knowledge briefly.)",
        question=query,
        max_words=max_words,
    )
