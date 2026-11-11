"""Text-chat quiz state: parse the questions the tutor asked, grade the student's reply, build feedback.

Same rules as the voice tutor (ai-tutor-backend-voice tutor-orchestrator applyQuizGrades):
CORRECT / PARTIAL close the item, a first INCORRECT earns a hint and a retry, a second reveals the
answer, UNCERTAIN costs no attempt, and NOT_AN_ANSWER leaves the quiz untouched.
The state is a plain dict so the client can hold it between turns:
  {"items": [{"question", "attempts", "done", "revealed", "outcome"}], "context": str}
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

GRADES = ("CORRECT", "PARTIAL", "INCORRECT", "UNCERTAIN", "NOT_AN_ANSWER")
MAX_ATTEMPTS = 2

_QUESTION_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:Q(?:uestion)?\s*)?(\d{1,2})\s*[.):\-]\s*(?:\*\*)?\s*(.+?)\s*$",
    re.I,
)
_OPTION_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?\(?([a-dA-D])[).]\s+(.+?)\s*$")
_GRADE_ROW_RE = re.compile(
    r"^\s*(\d{1,2})\s*[:.)\-]\s*(CORRECT|PARTIAL(?:LY_CORRECT)?|INCORRECT|UNCERTAIN|NOT_AN_ANSWER)\b\s*[|:\-]?\s*(.*)$",
    re.I,
)


def parse_quiz_questions(answer_text: str) -> list[str]:
    """Numbered questions (with their a)-d) options) from a tutor quiz reply."""
    questions: list[str] = []
    for raw in (answer_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        q = _QUESTION_LINE_RE.match(line)
        if q and (q.group(2).rstrip("*").strip().endswith("?") or len(q.group(2).split()) >= 4):
            questions.append(q.group(2).replace("**", "").strip())
            continue
        opt = _OPTION_LINE_RE.match(line)
        if opt and questions:
            questions[-1] += f" {opt.group(1).lower()}) {opt.group(2).strip()}"
    return questions[:10]


def new_quiz_state(answer_text: str, context: str = "") -> dict | None:
    questions = parse_quiz_questions(answer_text)
    if not questions:
        return None
    return {
        "items": [
            {"question": q, "attempts": 0, "done": False, "revealed": False, "outcome": None}
            for q in questions
        ],
        "context": (context or "")[:2000],
    }


def _items(state: dict | None) -> list[dict]:
    items = (state or {}).get("items")
    return items if isinstance(items, list) else []


def open_indexes(state: dict | None) -> list[int]:
    return [i for i, it in enumerate(_items(state)) if isinstance(it, dict) and not it.get("done")]


def has_open_questions(state: dict | None) -> bool:
    return bool(open_indexes(state))


def quiz_score(state: dict | None) -> float:
    score = 0.0
    for it in _items(state):
        if it.get("outcome") == "CORRECT":
            score += 1
        elif it.get("outcome") == "PARTIAL" and not it.get("revealed"):
            score += 0.5
    return score


def spoken_score(score: float) -> str:
    whole = int(score)
    if score - whole >= 0.5:
        return f"{whole} and a half" if whole else "half"
    return str(whole)


def parse_grades(raw: str, open_idx: list[int]) -> list[dict]:
    """Grader rows "1: CORRECT | note" -> [{"index", "grade", "note"}], numbered over open questions."""
    out: list[dict] = []
    for line in (raw or "").splitlines():
        m = _GRADE_ROW_RE.match(line)
        if not m:
            continue
        pos = int(m.group(1)) - 1
        if pos < 0 or pos >= len(open_idx):
            continue
        grade = m.group(2).upper()
        if grade.startswith("PARTIAL"):
            grade = "PARTIAL"
        out.append({"index": open_idx[pos], "grade": grade, "note": m.group(3).strip()})
    return out


def is_not_an_answer(grades: list[dict]) -> bool:
    return bool(grades) and all(g.get("grade") == "NOT_AN_ANSWER" for g in grades)


async def grade_quiz_answer(state: dict, answer: str) -> list[dict]:
    """LLM-grade the student's reply against every open question. Empty list on failure."""
    from app.services import llm_client

    idx = open_indexes(state)
    if not idx:
        return []
    items = _items(state)
    numbered = "\n".join(f"{n + 1}. {items[i]['question']}" for n, i in enumerate(idx))
    context = (state.get("context") or "").strip()
    prompt = (
        "You grade a school student's reply to quiz questions from their textbook chapter.\n"
        f"CHAPTER CONTEXT:\n{context or '(use general school knowledge of this chapter)'}\n\n"
        f"OPEN QUESTIONS:\n{numbered}\n\n"
        f"STUDENT REPLY: {answer}\n\n"
        "Rules: grade by meaning, not wording. Accept answers like \"1-b, 2 a\" for options. "
        "PARTIAL = right idea but missing a key part. UNCERTAIN = the student says they're unsure or "
        "don't know. NOT_AN_ANSWER = the reply is a complaint, a request, or the student's own question, "
        "not an attempt at any question. A question the reply doesn't address: skip it.\n"
        "Output one line per question the reply addresses, exactly: <number>: <CORRECT|PARTIAL|INCORRECT|"
        "UNCERTAIN|NOT_AN_ANSWER> | <the correct answer in under 12 words>"
    )
    for attempt in range(2):
        try:
            raw = await llm_client.complete(
                [{"role": "user", "content": prompt}], feature="chat", max_tokens=60 + 30 * len(idx)
            )
            grades = parse_grades(raw, idx)
            if grades:
                return grades
        except Exception as exc:
            logger.warning("quiz grading attempt %d failed: %s", attempt + 1, exc)
    return []


def apply_grades(state: dict, grades: list[dict]) -> dict:
    """Return a new state with the grades applied (NOT_AN_ANSWER rows change nothing)."""
    items = [dict(it) for it in _items(state)]
    for g in grades:
        i, grade = g.get("index"), g.get("grade")
        if not isinstance(i, int) or i >= len(items) or items[i].get("done") or grade == "NOT_AN_ANSWER":
            continue
        it = items[i]
        it["outcome"] = grade
        if grade in ("CORRECT", "PARTIAL"):
            it["done"] = True
        elif grade == "INCORRECT":
            it["attempts"] = int(it.get("attempts") or 0) + 1
            if it["attempts"] >= MAX_ATTEMPTS:
                it["done"] = True
                it["revealed"] = True
    return {**state, "items": items}


def build_feedback_block(state: dict, grades: list[dict]) -> str:
    """Prompt block telling the tutor exactly what to say about each graded question."""
    items = _items(state)
    lines = ["QUIZ GRADES (already decided — do not re-grade, do not ask a new question):"]
    for g in grades:
        i = g.get("index")
        if not isinstance(i, int) or i >= len(items) or g.get("grade") == "NOT_AN_ANSWER":
            continue
        it, note = items[i], (g.get("note") or "").strip()
        q = f"Q{i + 1} ({it['question'][:120]})"
        grade = g["grade"]
        if grade == "CORRECT":
            lines.append(f"- {q}: correct. Praise briefly in a few words.")
        elif grade == "PARTIAL":
            lines.append(f"- {q}: partly correct. Credit the right part, then add what was missing: {note}")
        elif grade == "UNCERTAIN":
            lines.append(f"- {q}: the student is unsure. Give one encouraging hint (not the answer) and invite a try.")
        elif it.get("revealed"):
            lines.append(f"- {q}: wrong again. Give the answer kindly in one line: {note}")
        else:
            lines.append(f"- {q}: not right yet. Give one hint (do not reveal the answer) and ask them to try again.")
    remaining = open_indexes(state)
    total = len(items)
    if remaining:
        nums = ", ".join(f"Q{i + 1}" for i in remaining)
        lines.append(f"Still open: {nums}. End by inviting an answer to those only.")
    else:
        lines.append(
            f"The quiz is finished. Say the score: {spoken_score(quiz_score(state))} out of {total}. "
            "Then offer one next step in a sentence. Do not ask a new quiz question."
        )
    lines.append("Plain prose, no headings, no quotation marks around answers.")
    return "\n".join(lines)
