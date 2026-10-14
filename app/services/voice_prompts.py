"""
Phase 5 — voice tutor prompt templates and builders.

Prompts are tuned for spoken output: continuous flow, grade-aware tone,
and alignment with Edge-TTS (plain text, no markdown/LaTeX).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

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
    from app.services.voice_tutor import (
        LearnerProfileSnapshot,
        ReplyIntent,
        TutorState,
        UnderstandingScores,
    )

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
- Light check-ins are fine ("right?", "okay?") — do not open with "Here's the cool part",
  "So, here's the thing", or "Imagine you're looking at…" unless that image is in the chapter context.
- Teach ONE idea per turn for a simple question. For a broad question, give a short overview.
- Maximum {max_words} words this turn. Spoken lines only — never a paragraph block.

SOUND LIKE A REAL TEACHER:
- Talk TO the student like a real person, not AT a textbook.
- Vary your rhythm — a tiny sentence, then a slightly longer one.
- Encourage warmly but don't repeat praise every turn.
- Don't echo the student's question word-for-word — nod to it, then teach.
- A check-in question is optional (roughly every 2–3 turns, not every sentence).

NAME RULE (strict — follow every turn):
- Default: do NOT say "{student_name}" at all. Teach without using their name.
- Say the name ONLY when truly needed: a greeting/welcome, calming them when
  confused, or celebrating a correct quiz answer.
- Never open a normal teaching answer with their name (no "Okay {student_name}, …").

{tts_speakability}

NEVER SAY (textbook / encyclopedia voice):
- "In everyday terms", "In your chapter", "According to the chapter",
  "According to the textbook", "the chapter says", "as per the textbook",
  "exactly what we're studying", "the process whereby", "it is defined as",
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
{reply_intent_guidance}
{learner_guidance}

ANSWERING PRIORITY (critical):
Always answer the student's LATEST MESSAGE first and directly — like a good teacher
in a voice call: natural, clear, helpful. No scripted openers. No label templates.
If the student asks a general conversational question ("What's your name?", "How are you?"),
answer that question — do not force it into the chapter topic.
If they thank you, praise an example, introduce themselves, or wrap up: acknowledge briefly
from conversation history — do NOT restart with "Welcome back" or re-teach the chapter.
If AUTHORITATIVE CHAPTER CONTEXT is empty or says (none), do not invent chapter facts —
reply conversationally only (ack, offer next step, or greet).

FACTUAL GROUNDING (critical — controls facts, not wording):
Priority: (1) retrieved chapter context, (2) this conversation if it already used that context,
(3) brief conversational language. Never (4) extra historical/scientific facts from memory.
The user message includes AUTHORITATIVE CHAPTER CONTEXT. For any historical, scientific,
geographical, or textbook fact:
- Use only that context. You may simplify, paraphrase, and reorder supported facts.
- Still say the chapter's key concept words out loud (names, terms, formulas in words) so
  the student hears the actual vocabulary — short answers must not drop every content word.
- If the student plants an off-chapter claim: say this chapter does not cover that, redirect
  to what it does teach, and do NOT repeat the off-topic term they planted.
- Do not contradict it. Do not add names, dates, battles, events, quotes, causes,
  examples, page numbers, or relationships that the context does not state.
- Never invent a quotation. Never invent why something happened (geography, alliances,
  rivers, strategy) unless the context states that reason.
- If the context is missing or too thin, say so naturally
  ("I don't see that detail in the textbook section I'm using.") and stop.
  Do NOT fill the gap from general knowledge. Do NOT guess a historically plausible answer.
- Do not say "according to the chapter/textbook" — just teach the supported facts.
- Never use fixed phrases like "In everyday terms" or "In your chapter".

ANSWER SHAPE (spoken, not a template to read aloud):
- Simple fact: 1–2 short sentences.
- Normal explanation: 2–5 short sentences.
- Broad question: big picture, then the major points that ARE in the context, then a brief close.
  Name only people, places, and events that appear in the context.
- "Explain simply": simpler words, same meaning.
- Follow-up ("why?", "did it…?", "where did they…?"): stay on that topic; use the context.
- If the student only laughed, said okay/wow/interesting, or acknowledged: one short reaction. Do not teach.

{expand_policy}

{affect_persona_block}"""

VOICE_TTS_SPEAKABILITY = """\
TTS OUTPUT (your text goes straight to speech synthesis):
- Speak in Indian classroom English — warm, clear, conversational.
- Latin script only: light markers like "na?", "isn't it?", "right?" — no Devanagari.
- Plain spoken sentences only — no markdown, bullets, headers, or emoji.
- Write how you'd talk: contractions, short clauses, natural commas for breath.
- End each teaching step with a period (not a comma) so the voice can pause naturally.
- Teach in small spoken sections: introduce → explain → emphasize → check-in.
- Give an example only if that example is already in the chapter context.
- For follow-ups ("why?", "again?", "another example?"): continue the same lesson
  directly — no "Good question.", "Sure!", or "Of course!" openers.
- Avoid textbook voice: no "Chapter 3 discusses", "According to the chapter",
  "the first point is", "in conclusion".
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
- State the idea in plain talk — do not invent a story around it.
- If the chapter has a map or diagram for this, invite them to look at it in one
  short line ("have a look at the map") — do not read a list of figure numbers aloud.
  Never claim you have put something on their screen; you cannot see their screen.
- Only name a figure number if it is in the chapter context and you need one pointer.
- Use a real-world hook only if that hook is in the chapter context."""

VOICE_USER_TEMPLATE = """\
CURRENT STUDENT MESSAGE (answer this):
{question}

AUTHORITATIVE CHAPTER CONTEXT (primary source for facts):
{context}

Speak your reply now like a friendly teacher in a voice call ({max_words} words max).
Natural spoken English — short sentences, contractions, easy to listen to.
Answer directly. Use the chapter context first. Do not contradict it. Do not add unsupported facts.
If a name, place, battle, date, quote, or reason is not in the context, do not say it.
If the context does not contain enough information, say so — do not invent an answer.
Never invent quotations, battles, dates, or reasons that are not in the context.
Never say "In everyday terms", "In your chapter", or "According to the chapter".
Do not use the student's name unless this is a greeting or they need reassurance."""

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


# A spoken answer may run somewhat longer than its source, since simplifying
# for a child costs words ("resisted the expansion of" -> "stood up to the
# Sultanate when it tried to move south"). Past this multiple of the retrieved
# context there is nothing left to say, and asking anyway is what makes the
# model reach into general knowledge to hit the target.
#
# Calibrated on the live grounding cases in tests/test_voice_grounding_cases.py:
# a 24-word Hoysala context asked for 160 words produced invented temples at
# Belur and Halebidu; the same context capped near 40 words stays clean, while
# a 75-word political-map context still gets its full ~110-word overview.
_CONTEXT_WORD_RATIO = 1.5
_CONTEXT_FLOOR_WORDS = 40


def _context_word_ceiling(context: str) -> int | None:
    """Word cap the retrieved context can actually support, or None if unknown."""
    n = len((context or "").split())
    if not n:
        return None
    return max(_CONTEXT_FLOOR_WORDS, int(n * _CONTEXT_WORD_RATIO))


def voice_word_limit(
    *,
    expand_deep: bool = False,
    answer_type: str = "short-answer",
    query: str = "",
    context: str = "",
) -> int:
    """Config-driven spoken word cap, tightened for greetings and brief requests.

    Also tightened to what the retrieved context can support: a word target the
    context cannot fill is an instruction to invent.
    """
    turn_cap = _VOICE_TURN_WORD_CAPS.get(answer_type)
    if turn_cap is not None:
        cap = min(turn_cap, VOICE_PROMPT_MAX_WORDS)
    elif expand_deep:
        cap = VOICE_PROMPT_EXPAND_WORDS
    else:
        from app.services.section_retrieval import query_breadth

        cap = (
            VOICE_PROMPT_EXPAND_WORDS
            if query and query_breadth(query) in ("broad", "chapter")
            else VOICE_PROMPT_MAX_WORDS
        )
    ceiling = _context_word_ceiling(context)
    return min(cap, ceiling) if ceiling else cap


def voice_turn_type_guidance(query: str) -> str:
    """Inject answer-type-specific speaking behavior (greeting vs teach vs brief)."""
    atype = detect_answer_type(query)
    mapping = {
        "greeting": (
            "The student is greeting or making small talk. "
            "This is one of the rare turns where you MAY use their first name once. "
            "Welcome them warmly in one sentence — do NOT start teaching yet."
        ),
        "one-word": (
            "The student gave a very short reply. Match their energy: one or two spoken "
            "sentences. Do not use their name."
        ),
        "brief": (
            "They want something short. One clear spoken answer, two sentences max. "
            "Do not use their name."
        ),
        "stepwise": (
            "They want steps — give ONE step in 6–12 word sentences, then pause. "
            "Do not use their name."
        ),
        "simplified": (
            "They need simpler talk. Shorter words, same chapter facts, slower pace. "
            "Do not use their name unless they sound confused."
        ),
        "exam-format": (
            "Exam-style request — stay spoken, but be precise. No written section headers. "
            "Do not use their name."
        ),
    }
    if atype in mapping:
        return mapping[atype]
    if atype in ("paragraph", "bullet-points"):
        return (
            "They asked for more — still use short spoken sentences (6–12 words), "
            "pause every 1–2 lines. Natural answer, no lists, no name."
        )
    # short-answer / definition / "what is X"
    return (
        "Answer naturally: the meaning first, using chapter facts, "
        "then one short supporting detail — no labels, no name."
    )


def voice_continuation_guidance(
    conversation_history: list[dict] | None,
    *,
    last_assistant: str = "",
    query: str = "",
    explained_points: list[str] | None = None,
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

    # Rule 5: about to re-teach a topic already covered this session (even if
    # phrased differently from the exact re-ask check above) — don't repeat
    # the same explanation verbatim. Compared as substantive terms on both
    # sides (explained_points is already stopword-filtered) so question
    # scaffolding ("can you tell me about") doesn't dilute the overlap score.
    if explained_points:
        from app.services.chapter_scope import substantive_query_terms

        q_terms = substantive_query_terms(q)
        for point in explained_points:
            point_words = set(point.split())
            if not point_words or len(q_terms) < 1:
                continue
            overlap = len(q_terms & point_words) / len(q_terms | point_words)
            if overlap >= 0.5:
                return (
                    "You already explained this topic earlier this session — do not repeat "
                    "the same explanation. Give a one-sentence recap, explain it a different "
                    "way (a new example or angle), or ask directly whether they'd like it "
                    "explained differently or something specific is unclear."
                )

    if short_follow and (turns or last_assistant):
        return (
            "Short follow-up — continue the SAME lesson and topic. "
            "Answer the reason/example/meaning directly. No 'Good question' / 'Sure' opener. "
            "Do not restart the chapter or repeat the full prior answer."
        )
    if not turns and not last_assistant:
        return ""
    if last_assistant:
        return (
            "This continues a live chat. Pick up the same topic. "
            "Don't restart from scratch, and don't open with a filler phrase."
        )
    return (
        "This continues a live chat. Brief nod to what you already said, "
        "then add the next small piece in spoken lines."
    )


def reply_intent_guidance(
    intent: "ReplyIntent",
    *,
    quiz_pending: bool,
    quiz_question: str,
    quiz_attempts: int,
    dialogue_act: str | None = None,
) -> str:
    """Rules 3, 4 & 6: how to respond given the classified reply intent and
    the state of any pending quiz question."""
    from app.services.voice_tutor import ReplyIntent

    act = (dialogue_act or "").strip().lower()
    if act == "intro":
        return (
            "The student is introducing themselves (name/place), not asking a chapter topic. "
            "Greet them warmly in 1–2 short sentences and invite a question about the current "
            "chapter. Do NOT say the topic is 'not covered' or show an a/b chapter menu. "
            "Do not teach chapter content this turn."
        )
    if act == "ack":
        if quiz_pending and quiz_question:
            return (
                f'The student reacted briefly (laugh/okay/praise), not answered. '
                f'Acknowledge in ONE short sentence, then gently keep your open question alive '
                f'("{quiz_question[:120]}"). Do not teach a new concept.'
            )
        return (
            "The student acknowledged or praised your last reply (e.g. 'okay', 'nice example'). "
            "Brief warm acknowledgment only — do NOT teach new facts or start a new concept. "
            "Offer ONE next step: another example, quick quiz, next part, or wrap up."
        )
    if intent == ReplyIntent.CLOSING or act == "closing":
        if quiz_pending:
            return (
                "The student wants to wrap up, but your last question was never resolved. "
                "In ONE short sentence, reveal the correct answer yourself, THEN warmly "
                "close out or offer what's next. Do not repeat your earlier explanation."
            )
        return (
            "The student is wrapping up or saying thanks. Do not repeat earlier explanation — "
            "acknowledge warmly, then offer to move to the next topic, try a quick quiz, "
            "or wrap up. Never say 'Welcome back' or 'What would you like to learn today?' "
            "mid-session. Never dump chapter categories again."
        )
    if intent == ReplyIntent.DONT_KNOW:
        hint = (
            'The student doesn\'t know / is unsure — this is NOT a wrong guess, so never say '
            '"Not quite" or similar. Give a hint, a simpler rephrasing, or a quick example '
            "instead of just handing over the answer."
        )
        if quiz_pending and quiz_attempts >= 1:
            hint += (
                " They've already had one chance on this question — reveal the correct "
                "answer now in one warm sentence, then move on."
            )
        return hint
    if intent == ReplyIntent.WRONG_ANSWER and quiz_pending:
        if quiz_attempts >= 1:
            return (
                "Wrong answer, and this is their 2nd+ try. Briefly say it's not quite right, "
                "explain why in one short line, THEN reveal the correct answer clearly, then "
                "move on — do not ask the same question again."
            )
        return (
            "Wrong answer — briefly and warmly say it's not quite right, explain why in one "
            "short line. This is only their first try, so do NOT reveal the answer yet — "
            "invite another attempt or give a small hint."
        )
    if intent == ReplyIntent.NEW_QUESTION and quiz_pending:
        return (
            f'You still have an open question pending ("{quiz_question[:120]}"). Briefly '
            "acknowledge or resolve it in one short line before answering their new question."
        )
    if intent == ReplyIntent.UNCLEAR:
        return (
            "Their message is unclear or may be a mis-hearing. If a word could be a "
            'mis-transcribed chapter term, ask a short confirming question ("did you mean '
            '___?") instead of guessing or saying it\'s off-topic.'
        )
    return ""


def voice_expand_policy(*, expand_deep: bool, max_words: int) -> str:
    if expand_deep:
        return (
            f"They want a bit more — up to {max_words} words, still short sentences, "
            "pause every 1–2 lines. Extra detail only from the chapter context, no new examples."
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


def persona_mode_for_affect(
    *,
    primary: str = "neutral",
    wants_quiz: bool = False,
) -> str:
    if wants_quiz or primary == "curious":
        return "teacher"
    if primary in ("excited", "personal", "affirmation", "bored"):
        return "friend"
    return "mentor"


_PERSONA_GUIDANCE = {
    "friend": (
        "PERSONA — FRIEND: Casual, warm, like a helpful older sibling. "
        "Light check-ins, no lecture tone."
    ),
    "mentor": (
        "PERSONA — MENTOR: Warm guide who believes in them. Patient, clear, encouraging."
    ),
    "teacher": (
        "PERSONA — TEACHER: Slightly more structured but still spoken, not essay-like. "
        "Good for quizzes and step-by-step."
    ),
}

_NEST_INTENT_GUIDANCE = {
    "simplify": (
        "The student asked to simplify — use simpler words for the SAME chapter facts. "
        "Do not add new facts."
    ),
    "example": (
        "The student wants an example — use one from the chapter context if present. "
        "Do not invent an unsupported story."
    ),
    "quiz": "The student wants to be quizzed — ask ONE short oral question from the chapter.",
    "repeat": "The student wants you to repeat — say it again more simply, same topic, same facts.",
}


def build_affect_persona_block(
    *,
    student_affect: Any | None = None,
    nest_intent: str | None = None,
    filler_phrase_played: str | None = None,
) -> str:
    parts: list[str] = []
    if student_affect is not None:
        hint = getattr(student_affect, "to_hint", lambda: "")()
        if hint:
            parts.append(f"AFFECT GUIDANCE:\n{hint}")
        persona = persona_mode_for_affect(
            primary=getattr(student_affect, "primary", "neutral"),
            wants_quiz=bool(getattr(student_affect, "wants_quiz", False)),
        )
        parts.append(_PERSONA_GUIDANCE.get(persona, _PERSONA_GUIDANCE["mentor"]))
        parts.append(
            "Affect changes tone, patience, and sentence complexity only — never facts."
        )
    if nest_intent and nest_intent in _NEST_INTENT_GUIDANCE:
        parts.append(f"NEST INTENT:\n{_NEST_INTENT_GUIDANCE[nest_intent]}")
    if filler_phrase_played:
        parts.append(
            f'FILLER ALREADY SPOKEN: "{filler_phrase_played[:120]}" — '
            "do not repeat this opener or paraphrase it. Continue naturally from the next sentence."
        )
    return "\n\n".join(parts)


def build_voice_system_prompt(
    query: str,
    *,
    # Only for sizing the word cap — the context itself goes in the user
    # message, so the model still sees it exactly once.
    context: str = "",
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
    reply_intent: "ReplyIntent | None" = None,
    quiz_pending: bool = False,
    quiz_question: str = "",
    quiz_attempts: int = 0,
    explained_points: list[str] | None = None,
    student_affect: Any | None = None,
    nest_intent: str | None = None,
    dialogue_act: str | None = None,
    filler_phrase_played: str | None = None,
) -> str:
    from app.services.voice_tutor import ReplyIntent, TutorState

    act = (dialogue_act or "").strip().lower()
    answer_type = detect_answer_type(query)
    if act in ("closing", "ack"):
        answer_type = "affirmation"
    elif act == "intro":
        answer_type = "greeting"
    max_words = voice_word_limit(
        expand_deep=expand_deep, answer_type=answer_type, query=query, context=context
    )
    if answer_type == "greeting" or act == "intro":
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
        reply_intent_guidance=reply_intent_guidance(
            reply_intent or ReplyIntent.NEW_QUESTION,
            quiz_pending=quiz_pending,
            quiz_question=quiz_question,
            quiz_attempts=quiz_attempts,
            dialogue_act=dialogue_act,
        ),
        learner_guidance=learner_hint,
        subject_guidance=subject_guidance_for(subject_name),
        turn_type_guidance=voice_turn_type_guidance(query),
        continuation_guidance=voice_continuation_guidance(
            conversation_history,
            last_assistant=last_assistant,
            query=query,
            explained_points=explained_points,
        ),
        expand_policy=voice_expand_policy(expand_deep=expand_deep, max_words=max_words),
        affect_persona_block=build_affect_persona_block(
            student_affect=student_affect,
            nest_intent=nest_intent,
            filler_phrase_played=filler_phrase_played,
        ),
    )


def build_voice_user_message(
    query: str,
    context: str,
    *,
    expand_deep: bool = False,
    dialogue_act: str | None = None,
) -> str:
    act = (dialogue_act or "").strip().lower()
    answer_type = detect_answer_type(query)
    if act in ("closing", "ack"):
        answer_type = "affirmation"
    elif act == "intro":
        answer_type = "greeting"
    max_words = voice_word_limit(
        expand_deep=expand_deep,
        answer_type=answer_type,
        query=query,
        context=context,
    )
    if act in ("closing", "ack", "intro"):
        empty_ctx = (
            "(none — conversational turn only. Do not teach chapter facts. "
            "Use conversation history.)"
        )
    else:
        empty_ctx = (
            "(No matching chapter excerpt. Do not invent facts. "
            "Say you don't have enough information in this chapter.)"
        )
    return VOICE_USER_TEMPLATE.format(
        context=context or empty_ctx,
        question=query,
        max_words=max_words,
    )
