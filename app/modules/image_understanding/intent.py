"""Infer student intent from message + image understanding."""

from __future__ import annotations

import re

from app.modules.image_understanding.schemas import ImageUnderstandingResult, coerce_intent

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(check|correct|is\s+my\s+answer|did\s+i\s+get|verify)\b", re.I), "check_answer"),
    (re.compile(r"\b(solve|find\s+x|calculate|compute|work\s+out)\b", re.I), "solve"),
    (re.compile(r"\b(summarize|summarise|summary|in\s+brief|in\s+short)\b", re.I), "summarize"),
    (re.compile(r"\b(simplif(?:y|ied|ication)|make\s+it\s+easier)\b", re.I), "simplify"),
    (re.compile(r"\b(translat)\w*\b", re.I), "translate"),
    (re.compile(r"\b(extract|read\s+(the\s+)?text|what\s+does\s+it\s+say)\b", re.I), "extract_text"),
    (re.compile(r"\b(compare|difference|vs\.?)\b", re.I), "compare"),
    (re.compile(r"\b(example|examples)\b", re.I), "generate_examples"),
    (re.compile(r"\b(quiz|practice\s+question|generate\s+question)\b", re.I), "generate_questions"),
    (re.compile(r"\b(diagram|what\s+does\s+this\s+(figure|diagram))\b", re.I), "explain_diagram"),
    (re.compile(r"\b(graph|plot|axis)\b", re.I), "explain_graph"),
    (re.compile(r"\b(table|tabular)\b", re.I), "explain_table"),
    (re.compile(r"\b(explain|why|how|what\s+is|describe|mean|simple\s+words)\b", re.I), "explain"),
    (re.compile(r"\b(answer|what\s+is\s+the\s+answer)\b", re.I), "answer"),
]


def detect_intent(student_message: str, result: ImageUnderstandingResult) -> str:
    msg = (student_message or "").strip()
    if msg:
        for pat, intent in _PATTERNS:
            if pat.search(msg):
                return intent
        # Question in message with image of problem → answer/explain
        if "?" in msg:
            return "answer"

    # Prefer question-level intent from vision
    for q in result.questions:
        qi = coerce_intent(q.intent)
        if qi != "unknown":
            return qi

    t = result.image_type
    if t in ("math_problem",) or result.mathematical_content.detected:
        return "solve"
    if t in ("diagram", "science_diagram"):
        return "explain_diagram"
    if t in ("graph", "chart"):
        return "explain_graph"
    if t in ("answer_sheet", "handwritten_homework") and result.question_detected:
        return "check_answer"
    if result.question_detected:
        return "answer"
    return "explain"
