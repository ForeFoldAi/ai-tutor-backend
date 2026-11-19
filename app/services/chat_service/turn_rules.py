"""Per-turn rules shared by the text and voice tutors.

Detects requests the answer-type router can't express (an exact sentence count, a standing
"keep it short", the student posing their own MCQ, "prove it from my textbook", "teach me
something new") and turns them, plus the always-on conduct rules, into one prompt block.
Keep the student-posed / short-answer patterns in sync with
ai-tutor-backend-voice/src/modules/tutor/reply-intent.ts.
"""
from __future__ import annotations

import re

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUM = r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"

_COUNT_RE = re.compile(
    rf"\b(?:in|within|using|with|into)\s+(?:just\s+|only\s+|exactly\s+)?{_NUM}\s+"
    r"(sentences?|lines?|points?|bullets?|bullet\s+points?)\b",
    re.I,
)

_QUIZ_COUNT_RE = re.compile(
    r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|a\s+few|a\s+couple\s+of)\s+"
    r"(?:(?:quick|short|simple|small|more|mcq|multiple[\s-]choice|practice)\s+)*(?:questions?|mcqs?)\b",
    re.I,
)

_SHORT_PREFERENCE_RE = re.compile(
    r"\b(?:(?:stop|no\s+more|don'?t|do\s+not|avoid)\s+(?:giving\s+|making\s+|using\s+)?(?:such\s+|so\s+|very\s+)?"
    r"long(?:er)?\s+(?:answers?|replies|explanations?)"
    r"|(?:keep|make)\s+(?:it|them|your\s+answers?)\s+(?:short|shorter|brief)"
    r"|(?:shorter|short|brief)\s+answers?\s+please"
    r"|too\s+long|answers?\s+(?:are|is)\s+(?:too\s+)?long)\b",
    re.I,
)

_STUDENT_POSED_RE = re.compile(
    r"\b(?:true\s+or\s+false|choose\s+(?:the\s+)?(?:best|correct|right)|select\s+all"
    r"|which\s+of\s+(?:the\s+)?(?:following|these|this\s+following)"
    r"|(?:best|correct|right)\s+(?:and\s+correct\s+)?answer"
    r"|i\s+will\s+give\s+(?:you\s+)?(?:the\s+)?options?)\b",
    re.I,
)
_OPTION_TOKEN_RE = re.compile(r"\b(?:option\s+[a-e1-4t]|\(?[a-d]\)|[a-d]\.)\s", re.I)

_CITE_RE = re.compile(
    r"\b(?:prove\s+(?:it|that|this)|(?:from|in)\s+(?:my|the|your)\s+(?:text\s*book|book)"
    r"|where\s+(?:does|is)\s+(?:it|that)\s+(?:say|said|written)|which\s+page|show\s+me\s+where"
    r"|cite\s+(?:it|the))\b",
    re.I,
)
_VISUAL_WORDS_RE = re.compile(r"\b(?:images?|pictures?|figures?|fig\.?|diagrams?|maps?)\b", re.I)

_NEW_CONTENT_RE = re.compile(
    r"\b(?:something\s+new|already\s+know|teach\s+me\s+(?:something\s+)?(?:new|else|more)"
    r"|new\s+(?:thing|topic|fact)|what\s+else)\b",
    re.I,
)


def _to_int(token: str) -> int | None:
    tok = token.strip().lower()
    if tok.isdigit():
        return int(tok)
    if tok.startswith("a few"):
        return 3
    if tok.startswith("a couple"):
        return 2
    return _NUMBER_WORDS.get(tok)


def requested_count(query: str) -> tuple[int, str] | None:
    """("in 3 sentences") -> (3, "sentences"); None when the student set no count."""
    m = _COUNT_RE.search(query or "")
    if not m:
        return None
    n = _to_int(m.group(1))
    if not n:
        return None
    unit = m.group(2).lower()
    unit = "points" if "point" in unit or "bullet" in unit else ("lines" if unit.startswith("line") else "sentences")
    return min(n, 10), unit


def quiz_count(query: str) -> int | None:
    """("ask me 5 questions") -> 5, clamped to 1-10; None when unstated."""
    m = _QUIZ_COUNT_RE.search(query or "")
    if not m:
        return None
    n = _to_int(m.group(1))
    return max(1, min(10, n)) if n else None


def asks_for_short_answers(query: str) -> bool:
    return bool(_SHORT_PREFERENCE_RE.search(query or ""))


def is_student_posed_question(query: str) -> bool:
    """The student is asking their own MCQ / true-false item, not requesting a quiz."""
    q = query or ""
    if _STUDENT_POSED_RE.search(q):
        return True
    return len(_OPTION_TOKEN_RE.findall(q)) >= 2


def is_cite_request(query: str) -> bool:
    q = query or ""
    return bool(_CITE_RE.search(q)) and not _VISUAL_WORDS_RE.search(q)


def wants_new_content(query: str) -> bool:
    return bool(_NEW_CONTENT_RE.search(query or ""))


_COMPLAINT_RE = re.compile(
    r"\b(?:you\s+(?:are|'re)\s+(?:wrong|repeating|not\s+listening|confusing|using\s+difficult)"
    r"|confus(?:ing|e)\s+me|you\s+(?:already|just)\s+(?:said|asked|told)"
    r"|(?:you\s+are\s+)?telling\s+(?:me\s+)?different|let\s+me\s+finish|don'?t\s+wanna\s+answer"
    r"|(?:stop|quit)\s+(?:asking|repeating|giving)|that'?s\s+(?:not\s+what\s+i|wrong)|i\s+didn'?t\s+ask"
    r"|why\s+(?:are|do)\s+you\s+(?:keep|asking|repeating)|this\s+is\s+(?:boring|confusing))\b",
    re.I,
)
_META_REQUEST_RE = re.compile(
    r"\b(?:explain|teach|tell\s+me|what\s+(?:is|are|was|were|does|did)|why|how|who|when|where"
    r"|can\s+you|could\s+you|give\s+me|show\s+me|summari[sz]e|recap|example)\b",
    re.I,
)


def is_complaint(query: str) -> bool:
    return bool(_COMPLAINT_RE.search(query or ""))


def is_non_answer(query: str) -> bool:
    """While a quiz is open: the message is a complaint, a request or the student's own question."""
    q = (query or "").strip()
    if not q:
        return True
    if _COMPLAINT_RE.search(q) or is_student_posed_question(q) or asks_for_short_answers(q):
        return True
    return bool(_META_REQUEST_RE.search(q)) and q.rstrip().endswith("?")


_CORE_RULES = """\
TURN RULES (always):
- FALSE STATEMENTS: If the student states something false ("India became independent in 1956", \
"states and districts are the same", "every state has the same language"), correct it kindly in one \
line with the right fact and one short reason, then continue. A statement is not a quiz answer — never \
reply "try again" or "not quite yet" to it.
- RELATED QUESTIONS: If the question is close to the chapter's subject (modern states or maps for a \
chapter on how India's political map changed) but the chapter context doesn't cover it, do not just refuse: \
answer in one or two accurate sentences starting "This isn't in your chapter, but", then link back to \
the chapter in one line. Only a question from a different subject altogether gets "that's not in this book".
- PUSHBACK: If the student doubts you ("are you sure?"), stay calm and restate the correct fact in one \
line, saying whether it comes from the chapter or is general knowledge. If they say their teacher explained \
something different, ask what the teacher said so you can compare. If you said something wrong earlier, \
say so plainly and correct it.
- MISHEARD WORDS: The message may come from speech-to-text. If a word doesn't fit but a chapter word \
sounds similar ("terrorists" → territories, "density" → dynasty, "stats" → states, "invited" → invented, \
"option T" → option D), answer the intended word and say once "I think you meant ...". Only when a word \
really was misheard — never use "I think you meant" to rephrase a clear question.
- MEMORY: Speak as "I" and about the conversation as "we" — never "the tutor". If the student asks about \
"the second point" and you gave no numbered list, use your last explanation and say which point you mean.
- Never say "your chapter covers/focuses on X" unless the CHAPTER CONTEXT shows X.
- NEVER say the context, chapter or textbook is unavailable. For follow-ups (example, again, simpler), \
build on your previous explanation."""


def build_turn_rules(
    query: str,
    *,
    prefers_short: bool = False,
    explained_points: list[str] | None = None,
    voice: bool = False,
) -> str:
    """Always-on conduct rules plus whatever this message specifically asks for."""
    lines = [_CORE_RULES]
    count = requested_count(query)
    if count:
        n, unit = count
        lines.append(
            f"- LENGTH (the student's own request, mandatory): exactly {n} {unit}. "
            "No headings, no extra bullets, no question at the end."
        )
    elif prefers_short or asks_for_short_answers(query):
        cap = 40 if voice else 60
        lines.append(
            f"- LENGTH PREFERENCE: the student asked for short answers. Keep this reply under {cap} words, "
            "one idea, no headings, no follow-up question."
        )
    if is_student_posed_question(query):
        lines.append(
            "- STUDENT'S OWN QUESTION: the student posed their own multiple-choice or true/false question. "
            "Answer it directly: name the correct option (or True/False) and give a one-line reason. "
            "Do not start a new quiz and do not grade the student."
        )
    if is_cite_request(query):
        lines.append(
            "- TEXTBOOK PROOF: quote or closely paraphrase the line(s) from the CHAPTER CONTEXT that support "
            "your previous answer, introduced with \"Your chapter says:\". If the chapter doesn't cover that "
            "fact, say so honestly in one line and restate the fact as general knowledge. Never invent page numbers."
        )
    points = [p for p in (explained_points or []) if p.strip()][-12:]
    if wants_new_content(query):
        known = "; ".join(points) if points else "the ideas already discussed in this conversation"
        lines.append(
            f"- NEW MATERIAL: the student already knows: {known}. Teach ONE named idea from the CHAPTER CONTEXT "
            "(a ruler, kingdom, event or term) that is not in that list, with a fresh example. Do not repeat "
            "an earlier example and do not drift to topics outside the chapter."
        )
    elif points:
        lines.append(
            f"- ALREADY EXPLAINED this session: {'; '.join(points)}. If you return to one, use a new example "
            "or angle instead of repeating."
        )
    return "\n".join(lines)
