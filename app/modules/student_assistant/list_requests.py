"""Turn plans for "give me 50 questions" style requests in Ask Learn Mastro.

The default length rules (40-120 words, at most 3 bullets) made the model refuse long lists
and the 512-token cap would cut them anyway. Text gets the full numbered list with a token
budget sized to it; voice turns it into a one-question-at-a-time quiz (reading 50 Q&As
aloud takes minutes) or five items at a time for non-question lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import ASSISTANT_MAX_TOKENS

LIST_CAP = 50
_TOKENS_PER_ITEM = 35
_LIST_MAX_TOKENS = 2400
_VOICE_BATCH = 5

_NUMBER_WORDS = {
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "twenty five": 25,
    "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "a hundred": 100,
}
_NUM = r"(\d{1,3}|" + "|".join(sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True)) + r")"

_QUESTION_NOUN = r"questions?|mcqs?|quiz(?:zes)?|q\s*&\s*as?|q\s*and\s*as?|qnas?"
_OTHER_NOUN = (
    r"facts?|examples?|items?|points?|words?|names?|tips?|things?|ideas?|sentences?|lines?|"
    r"comparisons?|problems?|exercises?|pairs?|sums?|fractions?"
)

_COUNT_RE = re.compile(
    rf"\b{_NUM}\s+(?:[a-z'-]+\s+){{0,4}}?({_QUESTION_NOUN}|{_OTHER_NOUN})\b",
    re.I,
)
_LIST_N_RE = re.compile(rf"\blist(?:ing)?\s+{_NUM}\b", re.I)
# "give me twenty more" / "20 more please" — same list shape as the last assistant list.
_MORE_RE = re.compile(rf"\b{_NUM}\s+more\b", re.I)
_REQUEST_VERB_RE = re.compile(
    r"\b(?:give|list|listing|generate|write|tell|share|make|create|prepare|ask|need|want|"
    r"show|provide|send|quiz|test|can\s+you|could\s+you|please)\b",
    re.I,
)
# "in 3 sentences" / "in 5 lines" is a length request, not a list.
_LENGTH_ONLY_RE = re.compile(rf"\b(?:in|within|using|into)\s+(?:just\s+|only\s+)?{_NUM}\s+(?:sentences?|lines?|words?)\b", re.I)

_QUIZ_PROGRESS_RE = re.compile(r"\bQuestion\s+(\d{1,3})\s+of\s+(\d{1,3})\b", re.I)
_STOP_RE = re.compile(r"\b(?:stop|enough|that'?s\s+all|no\s+more|end\s+(?:the\s+)?quiz|quit)\b", re.I)
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*(\d{1,3})[.)]\s+\S")


@dataclass(frozen=True)
class ListTurnPlan:
    prompt_block: str = ""
    max_tokens: int = ASSISTANT_MAX_TOKENS
    continue_prompt: str = ""
    # Text lists only: complete once items 1..N are all present (gaps count as incomplete).
    expected_items: int = 0

    def list_incomplete(self, answer: str) -> bool:
        if self.expected_items <= 0:
            return False
        numbers = {int(x) for x in _NUMBERED_ITEM_RE.findall(answer or "")}
        needed = set(range(1, self.expected_items + 1))
        return not needed.issubset(numbers)


def _to_int(tok: str) -> int | None:
    t = re.sub(r"\s+", " ", tok.strip().lower())
    if t.isdigit():
        return int(t)
    return _NUMBER_WORDS.get(t)


def _last_assistant_was_questions(history: list[dict[str, str]] | None) -> bool:
    for turn in reversed(history or []):
        if (turn.get("role") or "") != "assistant":
            continue
        content = turn.get("content") or ""
        if _QUIZ_PROGRESS_RE.search(content):
            return True
        if re.search(r"(?i)\b(?:question|mcq|q\s*&\s*a|answer\s*:)", content):
            return True
        if _NUMBERED_ITEM_RE.search(content):
            return False
        return False
    return False


def requested_list(
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> tuple[int, bool] | None:
    """(count, is_questions) when the student asks for a list of more than 3 items."""
    q = query or ""
    if _LENGTH_ONLY_RE.search(q) or not _REQUEST_VERB_RE.search(q):
        return None
    m = _COUNT_RE.search(q)
    if m:
        n = _to_int(m.group(1))
        is_questions = bool(re.fullmatch(_QUESTION_NOUN, m.group(2), re.I))
    else:
        m = _LIST_N_RE.search(q)
        if m:
            n = _to_int(m.group(1))
            is_questions = bool(re.search(_QUESTION_NOUN, q, re.I))
        else:
            m = _MORE_RE.search(q)
            if not m:
                return None
            n = _to_int(m.group(1))
            is_questions = _last_assistant_was_questions(conversation_history)
    if not n or n <= 3:
        return None
    return n, is_questions


def _quiz_in_progress(history: list[dict[str, str]] | None) -> tuple[int, int] | None:
    for turn in reversed(history or []):
        if (turn.get("role") or "") != "assistant":
            continue
        hits = _QUIZ_PROGRESS_RE.findall(turn.get("content") or "")
        if not hits:
            return None
        k, n = (int(x) for x in hits[-1])
        return (k, n) if 0 < k < n else None
    return None


def plan_list_turn(
    query: str,
    conversation_history: list[dict[str, str]] | None,
    *,
    voice_mode: bool,
) -> ListTurnPlan:
    asked = requested_list(query, conversation_history)
    if asked:
        requested, is_questions = asked
        n = min(requested, LIST_CAP)
        over = (
            f"They asked for {requested}; give the first {n} and offer to continue with the rest."
            if requested > n
            else ""
        )
        if voice_mode and is_questions:
            return ListTurnPlan(
                prompt_block=(
                    "VOICE QUIZ (overrides the length rules above for this reply):\n"
                    f"- The student asked for {n} questions by voice. Reading them all aloud is too long, "
                    "so run them as a quiz, one question at a time.\n"
                    f"- Say in one short sentence that you will ask them one by one, then ask only "
                    f'"Question 1 of {n}: ..." and stop. Do not give the answer yet.\n'
                    "- Never say you cannot give that many questions."
                ),
            )
        if voice_mode:
            return ListTurnPlan(
                prompt_block=(
                    "VOICE LIST (overrides the length rules above for this reply):\n"
                    f"- The student asked for {n} items by voice. Give items 1 to {min(_VOICE_BATCH, n)} now, "
                    "numbered, one short sentence each, then ask if they want the next five.\n"
                    "- Never say you cannot give that many."
                ),
            )
        item = (
            'the question, then " — Answer: " and a short answer, on the same line'
            if is_questions
            else "one short line"
        )
        return ListTurnPlan(
            prompt_block=(
                "LIST REQUEST (overrides the length and bullet limits above for this reply):\n"
                f"- The student asked for {n} items. Give all {n}, numbered 1 to {n} with no gaps, one per line.\n"
                f"- Each item: {item}. Keep each item short and right for their grade.\n"
                "- At most one short intro line and one short closing line. No extra explanation.\n"
                "- Never say you cannot list that many in one message."
                + (f"\n- {over}" if over else "")
            ),
            max_tokens=min(_LIST_MAX_TOKENS, max(ASSISTANT_MAX_TOKENS, 150 + n * _TOKENS_PER_ITEM)),
            continue_prompt=(
                f"Continue the numbered list and fill any missing numbers so the final list is "
                f"contiguous 1 to {n} with no gaps. Do not repeat items you already wrote correctly."
            ),
            expected_items=n,
        )

    progress = _quiz_in_progress(conversation_history) if voice_mode else None
    if progress and not _STOP_RE.search(query or ""):
        k, n = progress
        return ListTurnPlan(
            prompt_block=(
                "VOICE QUIZ IN PROGRESS (overrides the length rules above for this reply):\n"
                f"- You just asked Question {k} of {n}. If the student's reply is an answer, say in one short "
                "sentence whether it is right (give the correct answer if not), then ask "
                f'"Question {k + 1} of {n}: ..." and stop. Never repeat an earlier question.\n'
                "- If they ask something else or want to stop, answer that instead and end the quiz."
            ),
        )
    return ListTurnPlan()
