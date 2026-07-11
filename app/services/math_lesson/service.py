"""Interactive mathematics lesson prompt, detection, and JSON extraction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.math_lesson.fallbacks import (
    build_fallback_math_lesson,
    merge_catalog_visualization,
    query_requests_assessment,
    query_requests_hints,
    query_requests_practice_mode,
)
from app.services.math_lesson.schemas import MathLesson

logger = logging.getLogger(__name__)

_MATH_LESSON_FENCE_RE = re.compile(
    r"```(?:math-lesson|json:math-lesson|math_lesson)\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)

_INCOMPLETE_MATH_LESSON_RE = re.compile(
    r"\n?```(?:math-lesson|json:math-lesson|math_lesson)\s*\n[\s\S]*$",
    re.IGNORECASE,
)

_MARKDOWN_FENCE_RE = re.compile(
    r"```markdown\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)

_INTERACTIVE_LESSON_SKIP_TYPES = frozenset(
    {"greeting", "affirmation", "personal-response", "one-word", "brief"}
)

_CONCEPT_TEACHING_TYPES = frozenset(
    {
        "conceptual",
        "analytical",
        "factual",
        "concept",
        "definition",
        "paragraph",
        "short-answer",
        "stepwise",
        "simplified",
        "bullet-points",
        "exam-format",
    }
)


_MATH_VISUALIZATION_APPENDIX = """\
INTERACTIVE VISUALIZATION (MANDATORY for every mathematics answer — append after the prose answer):

IMPORTANT RULES:
- Interactive Exploration is REQUIRED for every mathematics question (except greetings / one-word replies).
- Write the FULL standard mathematics tutor answer FIRST using every required section \
(**To Find**, **Given Information**, **Concept Behind It**, **The Formula**, **Solution**, \
**Quick Check**, **Final Answer**, **Key Takeaway**, **Practice Question**) exactly as before.
- Do NOT wrap the answer in ```markdown or any code fence — use **bold** headings directly in plain text.
- Do NOT replace the standard sections with "Concept Name" / "Learning Objective" headings.
- AFTER **Practice Question**, append ONE complete ```math-lesson``` JSON block for the interactive visualization.
- The JSON must be valid, complete, and closed with ``` — keep strings concise so the block is not cut off.
- Keep the math-lesson JSON focused on the interactive visualization only.
- Do NOT include practiceMode, commonMistakes, aiHints, or assessment unless the student \
explicitly asked for them (see OPTIONAL FIELDS below).

```math-lesson
{
  "conceptName": "...",
  "classLevel": "Class N",
  "learningObjective": "...",
  "conceptExplanation": "one-sentence summary for the viz panel",
  "visualization": {
    "visualizationType": "counting|number-line|place-value|multiplication-grid|bar-model|decimal-blocks|percent-circle|ratio-bar|integer-line|clock-time|money|pattern|shapes-basic|matchstick-squares|symmetry|factor-tree|pythagoras|trig-basic|concept-explorer|algebra-tiles|factor-rectangle|geometry-basics|angle-explorer|parallel-transversal|linear-graph|line-intersection|triangle-explorer|triangle-angle-sum|quadrilateral-morph|statistics-lab|mensuration-cube|mensuration-cylinder|area-resizer|probability-dice|probability-coin|circle-tangent|geometry-construction|fractions|circle|coordinate|graph|probability|statistics|mensuration|generic",
    "title": "...",
    "description": "what the student should explore",
    "sliders": [{"id": "turnSlider", "label": "Quarter turns", "min": 0, "max": 4, "step": 1, "default": 1, "unit": ""}],
    "liveCalculations": [{"id": "angle", "label": "Angle", "formula": "turnSlider * 90", "unit": "°"}],
    "buttons": [{"id": "reset", "label": "Reset", "action": "reset"}],
    "colors": {"primary": "#3B82F6", "secondary": "#10B981", "accent": "#F59E0B", "background": "#F8FAFC", "text": "#1E293B"},
    "studentInteractions": [{"id": "s1", "type": "slide", "description": "Move the slider", "expectedObservation": "..."}]
  },
  "guidedExploration": ["What happens if..."]
}
```

Visualization rules (Class 1–10 — always pick the most student-friendly interactive type):
- Class 1–2: counting, number-line, shapes-basic, matchstick-squares, clock-time, money, pattern
- Class 3–5: multiplication-grid, fractions, place-value, bar-model (division), perimeter/area-resizer
- Class 6–8: integer-line, ratio-bar, percent-circle, linear-graph, angle-explorer, statistics-lab
- Class 9–10: algebra-tiles, factor-rectangle, parallel-transversal, pythagoras, trig-basic, circle-tangent
- Lines / rays / segments: use geometry-basics (drag points, switch Segment / Ray / Line)
- ANY concept: use concept-explorer with relevant sliders only if no exact match
- Always include sliders OR draggables OR animate button — never static-only
- Match complexity to the student's class level from the prompt context
- Include at least one slider OR draggable object OR interactive button.
- liveCalculations must use slider id names as variables (e.g. turnSlider, r, numerator, denominator).
- Never use static diagrams when interaction is possible."""


def should_use_interactive_math_lesson(
    *,
    subject_name: str,
    answer_type: str,
    query: str,
    heading_scope: Any | None = None,
    question_type: str = "conceptual",
) -> bool:
    """True when the tutor should append an interactive math visualization JSON block."""
    from app.services.section_heading import HeadingScope

    del question_type  # all substantive maths questions get Interactive Exploration
    if not _is_mathematics_subject(subject_name):
        return False
    if not (query or "").strip():
        return False
    if answer_type in _INTERACTIVE_LESSON_SKIP_TYPES:
        return False
    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        return False
    return True


def _is_mathematics_subject(subject_name: str) -> bool:
    s = (subject_name or "").lower()
    keywords = ("math", "algebra", "geometry", "calculus", "arithmetic", "trigonometry")
    return any(kw in s for kw in keywords)


_OPTIONAL_HINTS_APPENDIX = """\
OPTIONAL — student asked for HINTS:
Add "aiHints": [["easy hint", "medium hint", "hard hint"]] to the math-lesson JSON \
(3 progressive levels). Do not add assessment unless also requested."""

_OPTIONAL_ASSESSMENT_APPENDIX = """\
OPTIONAL — student asked for ASSESSMENT / QUIZ:
Add "assessment": [{"question": "...", "type": "conceptual"}, ...] with 3–5 short questions \
to the math-lesson JSON. Do not add aiHints unless also requested."""

_OPTIONAL_PRACTICE_APPENDIX = """\
OPTIONAL — student asked for PRACTICE:
Add "practiceMode": {"easy": "...", "medium": "...", "hard": "...", "challenge": "..."} \
to the math-lesson JSON."""


_MATH_VISUALIZATION_APPENDIX_ELEMENTARY = """\
INTERACTIVE VISUALIZATION (MANDATORY for every elementary mathematics answer):
- Write a SHORT kid-friendly prose answer FIRST (60–120 words, no section headers).
- You MUST append ONE complete ```math-lesson``` JSON block for the interactive visual.
- Do NOT use **To Find**, **Given Information**, **The Formula**, or other exam-style section headers.
- The JSON must be valid, complete, and closed with ``` — keep strings concise so the block is not cut off.
- Do NOT include practiceMode, commonMistakes, aiHints, or assessment unless the student \
explicitly asked for them (see OPTIONAL FIELDS below).

```math-lesson
{
  "conceptName": "...",
  "classLevel": "Class N",
  "learningObjective": "...",
  "conceptExplanation": "one-sentence summary for the viz panel",
  "visualization": {
    "visualizationType": "shapes-basic|matchstick-squares|counting|number-line|multiplication-grid|fractions|concept-explorer|generic",
    "title": "...",
    "description": "what the student should explore",
    "sliders": [{"id": "sides", "label": "Sides", "min": 3, "max": 8, "step": 1, "default": 4}],
    "buttons": [{"id": "animate", "label": "▶ Draw shape", "action": "animate"}]
  },
  "guidedExploration": ["What happens if..."]
}
```"""


def get_visualization_appendix_prompt(query: str = "", *, elementary: bool = False) -> str:
    """Prompt add-on: keep standard math format, append visualization JSON at the end."""
    parts = [_MATH_VISUALIZATION_APPENDIX_ELEMENTARY if elementary else _MATH_VISUALIZATION_APPENDIX]
    if query_requests_hints(query):
        parts.append(_OPTIONAL_HINTS_APPENDIX)
    if query_requests_assessment(query):
        parts.append(_OPTIONAL_ASSESSMENT_APPENDIX)
    if query_requests_practice_mode(query):
        parts.append(_OPTIONAL_PRACTICE_APPENDIX)
    return "\n\n".join(parts)


def apply_math_lesson_display_policy(
    lesson: dict[str, Any] | None,
    query: str,
) -> dict[str, Any] | None:
    """Strip optional lesson panels unless the student explicitly asked for them."""
    if not lesson:
        return None
    out = dict(lesson)
    if not query_requests_hints(query):
        out["aiHints"] = []
    if not query_requests_assessment(query):
        out["assessment"] = []
    if not query_requests_practice_mode(query):
        out["practiceMode"] = {"easy": "", "medium": "", "hard": "", "challenge": ""}
        out["commonMistakes"] = []
    return out


def _try_parse_lesson(raw: str) -> MathLesson | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return MathLesson.model_validate(data)
    except Exception as exc:
        logger.debug("Math lesson JSON parse failed: %s", exc)
        return None


def clean_tutor_answer_text(answer: str) -> str:
    """Remove markdown fences and incomplete math-lesson blocks from displayed answer."""
    text = answer or ""
    text = _MARKDOWN_FENCE_RE.sub(r"\1", text)
    text = _INCOMPLETE_MATH_LESSON_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def finalize_math_answer(
    answer: str,
    query: str,
    *,
    class_level: str = "",
    subject_name: str = "",
    existing_lesson: dict[str, Any] | None = None,
    allow_fallback: bool = True,
    conversation_history: list[dict[str, str]] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Extract math-lesson JSON from answer, or synthesize a fallback for math subjects."""
    clean, lesson = extract_math_lesson_from_answer(answer)
    if lesson is None:
        lesson = existing_lesson
    if not _is_mathematics_subject(subject_name):
        return clean, lesson
    if lesson is None and not allow_fallback:
        return clean, None
    catalog = (
        build_fallback_math_lesson(
            query, class_level, conversation_history=conversation_history
        )
        if allow_fallback
        else None
    )
    if lesson is None:
        if catalog is None:
            return clean, None
        logger.info("Using Class 1-10 math visualization for query=%r", query[:80])
        return clean, apply_math_lesson_display_policy(catalog, query)
    if catalog is None:
        return clean, apply_math_lesson_display_policy(lesson, query)
    merged = merge_catalog_visualization(lesson, catalog)
    if merged is not lesson:
        logger.info(
            "Merged catalog viz %s for query=%r",
            catalog.get("visualization", {}).get("visualizationType"),
            query[:80],
        )
    return clean, apply_math_lesson_display_policy(merged, query)


def extract_math_lesson_from_answer(answer: str) -> tuple[str, dict[str, Any] | None]:
    """Extract math-lesson JSON from answer; return (clean_answer, lesson_dict)."""
    clean, lesson = strip_math_lesson_block(answer)
    clean = clean_tutor_answer_text(clean)
    if lesson is None:
        # No valid JSON — still strip any broken fence from the raw answer
        return clean_tutor_answer_text(answer), None
    try:
        validated = MathLesson.model_validate(lesson)
        return clean, validated.model_dump(mode="json")
    except Exception as exc:
        logger.warning("Math lesson validation failed, returning raw dict: %s", exc)
        return clean, lesson


def strip_math_lesson_block(answer: str) -> tuple[str, dict[str, Any] | None]:
    """Remove math-lesson fenced block from answer text."""
    match = _MATH_LESSON_FENCE_RE.search(answer or "")
    if not match:
        return clean_tutor_answer_text(answer or ""), None
    lesson = _try_parse_lesson(match.group(1))
    clean = (answer[: match.start()] + answer[match.end() :]).strip()
    clean = clean_tutor_answer_text(clean)
    if lesson is None:
        return clean, None
    return clean, lesson.model_dump(mode="json")
