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
- AFTER **Practice Question**, append ONE complete ```math-lesson``` JSON block.
- The JSON must be valid and closed with ```.
- Do NOT invent a per-lesson `colors` object. Choose `paletteId` from: \
"primary-1to2" | "primary-3to5" | "middle-6to8" | "technical-9to10" \
(match the student's class band).
- Do NOT include practiceMode, commonMistakes, aiHints, or assessment unless the student asked.

WORKED EXAMPLE — fractions (2d):
```math-lesson
{
  "conceptName": "Fractions",
  "classLevel": "Class 5",
  "visualization": {
    "visualizationType": "fractions",
    "title": "Pizza Fractions",
    "description": "Change numerator and denominator",
    "paletteId": "primary-3to5",
    "renderMode": "2d",
    "sliders": [
      {"id": "numerator", "label": "Numerator", "min": 0, "max": 8, "step": 1, "default": 3},
      {"id": "denominator", "label": "Denominator", "min": 1, "max": 8, "step": 1, "default": 4}
    ],
    "liveCalculations": [{"id": "pct", "label": "Percent", "formula": "numerator / denominator * 100", "unit": "%"}]
  },
  "guidedExploration": ["What happens if the denominator increases?"]
}
```

WORKED EXAMPLE — algebra-stepper (solve 2x + 5 = 15):
```math-lesson
{
  "conceptName": "Linear Equations",
  "classLevel": "Class 8",
  "visualization": {
    "visualizationType": "algebra-stepper",
    "title": "Solve 2x + 5 = 15",
    "paletteId": "middle-6to8",
    "renderMode": "2d",
    "algebraSteps": [
      {"id": "s0", "expressionBefore": "2x + 5 = 15", "expressionAfter": "2x + 5 = 15", "operation": "Start", "highlightTerms": ["2x", "5"]},
      {"id": "s1", "expressionBefore": "2x + 5 = 15", "expressionAfter": "2x = 10", "operation": "Subtract 5 from both sides", "highlightTerms": ["5"]},
      {"id": "s2", "expressionBefore": "2x = 10", "expressionAfter": "x = 5", "operation": "Divide both sides by 2", "highlightTerms": ["2"]}
    ]
  }
}
```

WORKED EXAMPLE — compound-interest-visual (3-year CI):
```math-lesson
{
  "conceptName": "Compound Interest",
  "classLevel": "Class 8",
  "visualization": {
    "visualizationType": "compound-interest-visual",
    "title": "CI Growth",
    "paletteId": "middle-6to8",
    "financeSpec": {"principal": 10000, "rate": 8, "timeYears": 3, "mode": "compound-interest", "compoundingFrequency": "annually"},
    "sliders": [
      {"id": "P", "label": "Principal", "min": 1000, "max": 50000, "step": 1000, "default": 10000},
      {"id": "r", "label": "Rate %", "min": 1, "max": 20, "step": 0.5, "default": 8},
      {"id": "n", "label": "Years", "min": 1, "max": 10, "step": 1, "default": 3}
    ]
  }
}
```

WORKED EXAMPLE — quadratic grapher (x² − 5x + 6 = 0):
```math-lesson
{
  "conceptName": "Quadratic Equations",
  "classLevel": "Class 10",
  "visualization": {
    "visualizationType": "linear-graph",
    "title": "y = x² − 5x + 6",
    "paletteId": "technical-9to10",
    "curveType": "quadratic",
    "coefficients": [1, -5, 6],
    "sliders": [
      {"id": "a", "label": "a", "min": -3, "max": 3, "step": 0.5, "default": 1},
      {"id": "b", "label": "b", "min": -8, "max": 8, "step": 0.5, "default": -5},
      {"id": "c", "label": "c", "min": -8, "max": 8, "step": 0.5, "default": 6}
    ]
  }
}
```

WORKED EXAMPLE — 3d cube (renderMode 3d + scene required):
```math-lesson
{
  "conceptName": "Cube",
  "classLevel": "Class 9",
  "visualization": {
    "visualizationType": "mensuration-cube",
    "title": "Cube Explorer",
    "paletteId": "technical-9to10",
    "renderMode": "3d",
    "scene": {
      "groundGrid": true,
      "camera": {"position": [4.8, 3.2, 5.6], "target": [0, 0.65, 0], "fov": 40},
      "objects": [{"id": "cube", "type": "box", "position": [0, 0, 0], "scaleDrivenBy": "s", "color": "primary", "wireframeAccent": true}]
    },
    "sliders": [{"id": "s", "label": "Side", "min": 1, "max": 10, "step": 1, "default": 5}]
  }
}
```

WORKED EXAMPLE — composite cone-on-cylinder (Class 10 solids):
```math-lesson
{
  "conceptName": "Combination of Solids",
  "classLevel": "Class 10",
  "visualization": {
    "visualizationType": "mensuration-cylinder",
    "title": "Cone on Cylinder",
    "paletteId": "technical-9to10",
    "renderMode": "3d",
    "scene": {
      "groundGrid": true,
      "objects": [{
        "id": "combo",
        "type": "composite",
        "position": [0, 0, 0],
        "children": [
          {"id": "cyl", "type": "cylinder", "position": [0, 0.6, 0], "color": "primary"},
          {"id": "cone", "type": "cone", "position": [0, 1.6, 0], "color": "accent"}
        ]
      }]
    }
  }
}
```

WORKED EXAMPLE — heights-and-distances (3d):
```math-lesson
{
  "conceptName": "Heights and Distances",
  "classLevel": "Class 10",
  "visualization": {
    "visualizationType": "heights-distances-scene",
    "title": "Angle of Elevation",
    "paletteId": "technical-9to10",
    "renderMode": "3d",
    "scene": {
      "groundGrid": true,
      "objects": [
        {"id": "tower", "type": "box", "position": [4, 1.2, 0], "scale": [0.5, 2.4, 0.5], "color": "primary"},
        {"id": "eye", "type": "sphere", "position": [0, 0.2, 0], "scale": [0.2, 0.2, 0.2], "color": "accent"}
      ]
    },
    "sliders": [
      {"id": "angle", "label": "Angle °", "min": 15, "max": 75, "step": 1, "default": 30},
      {"id": "distance", "label": "Distance m", "min": 10, "max": 100, "step": 5, "default": 40}
    ]
  }
}
```

Type rules:
- Solving / simplify / rationalise: "algebra-stepper" with algebraSteps (SymPy-verified steps may be injected server-side — keep concise).
- CI / discount / tax: "compound-interest-visual" + financeSpec.
- Quadratic: "linear-graph" + curveType "quadratic" + coefficients [a,b,c].
- Solids / heights-distances: renderMode "3d" with a non-empty scene.objects array.
- Never use static-only diagrams; include sliders, buttons, or scene scaleDrivenBy.
- liveCalculations must use slider id names as variables."""


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
- Do NOT use exam-style section headers.
- Do NOT invent a `colors` object — set paletteId to "primary-1to2" (Class 1–2) or "primary-3to5" (Class 3–5).
- Do NOT include practiceMode / aiHints / assessment unless asked.

```math-lesson
{
  "conceptName": "...",
  "classLevel": "Class N",
  "learningObjective": "...",
  "conceptExplanation": "one-sentence summary for the viz panel",
  "visualization": {
    "visualizationType": "shape-lab|matchstick-squares|counting|number-line|multiplication-grid|fractions|concept-explorer|generic",
    "title": "...",
    "description": "what the student should explore",
    "paletteId": "primary-3to5",
    "renderMode": "2d",
    "interactiveObjects": [{"id": "shape", "label": "Square", "type": "square"}],
    "sliders": [{"id": "s", "label": "Side length", "min": 1, "max": 10, "step": 1, "default": 4}],
    "buttons": [{"id": "animate", "label": "▶ Draw square", "action": "animate"}]
  },
  "guidedExploration": ["What happens if..."]
}
```"""


def algebra_steps_from_math_engine(query: str, class_level: str = "") -> list[dict[str, Any]]:
    """Map SymPy MathEngineResult steps → algebraSteps for algebra-stepper grounding."""
    try:
        from app.services.math_engine import try_solve
    except Exception:
        return []
    try:
        result = try_solve(query, class_level=class_level)
    except Exception as exc:
        logger.debug("try_solve for algebraSteps failed: %s", exc)
        return []
    if not result or not getattr(result, "solved", False):
        return []

    steps_out: list[dict[str, Any]] = []
    engine_steps = list(getattr(result, "steps", None) or [])
    sol_lines = list(getattr(result, "solution_latex", None) or [])

    if engine_steps:
        prev = ""
        for i, step in enumerate(engine_steps):
            lines = list(getattr(step, "latex_lines", None) or [])
            expr = (lines[-1] if lines else "").replace("$$", "").strip()
            before = prev or (lines[0].replace("$$", "").strip() if lines else expr)
            after = expr or before
            steps_out.append(
                {
                    "id": f"sympy-{i}",
                    "expressionBefore": before or after,
                    "expressionAfter": after or before,
                    "operation": getattr(step, "label", "") or getattr(step, "note", "") or "",
                    "highlightTerms": [],
                }
            )
            prev = after
    elif sol_lines:
        cleaned = [ln.replace("$$", "").strip() for ln in sol_lines if ln.strip()]
        for i, line in enumerate(cleaned):
            before = cleaned[i - 1] if i else line
            steps_out.append(
                {
                    "id": f"sympy-{i}",
                    "expressionBefore": before,
                    "expressionAfter": line,
                    "operation": "Verified step" if i else "Start",
                    "highlightTerms": [],
                }
            )

    final = str(getattr(result, "final_answer", "") or "").strip()
    if final and steps_out and final not in steps_out[-1].get("expressionAfter", ""):
        steps_out.append(
            {
                "id": f"sympy-final",
                "expressionBefore": steps_out[-1]["expressionAfter"],
                "expressionAfter": final.replace("$$", "").strip(),
                "operation": "Final answer",
                "highlightTerms": [],
            }
        )
    return steps_out


def inject_sympy_algebra_steps(
    lesson: dict[str, Any] | None,
    query: str,
    class_level: str = "",
) -> dict[str, Any] | None:
    """If lesson is algebra-stepper and algebraSteps empty, fill from SymPy."""
    if not lesson:
        return lesson
    viz = lesson.get("visualization") or {}
    vtype = str(viz.get("visualizationType") or "")
    if vtype != "algebra-stepper":
        return lesson
    existing = viz.get("algebraSteps") or []
    if existing:
        return lesson
    steps = algebra_steps_from_math_engine(query, class_level)
    if not steps:
        return lesson
    viz = dict(viz)
    viz["algebraSteps"] = steps
    out = dict(lesson)
    out["visualization"] = viz
    return out


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
    from app.services.math_lesson.elementary_catalog import apply_shape_lab_guide

    out = dict(lesson)
    if not query_requests_hints(query):
        out["aiHints"] = []
    if not query_requests_assessment(query):
        out["assessment"] = []
    if not query_requests_practice_mode(query):
        out["practiceMode"] = {"easy": "", "medium": "", "hard": "", "challenge": ""}
        out["commonMistakes"] = []
    return apply_shape_lab_guide(out, query)


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


def safe_finalize_math_answer(
    answer: str,
    query: str,
    **kwargs: Any,
) -> tuple[str, dict[str, Any] | None]:
    """finalize_math_answer with fail-open fallback — never crash the chat path."""
    try:
        clean, lesson = finalize_math_answer(answer, query, **kwargs)
    except Exception as exc:
        logger.exception("finalize_math_answer failed: %s", exc)
        return clean_tutor_answer_text(answer or ""), None
    if lesson is not None:
        try:
            lesson = json.loads(json.dumps(lesson, default=str))
        except Exception as exc:
            logger.warning("math_lesson JSON sanitize failed: %s", exc)
            lesson = None
    return clean, lesson


def finalize_math_answer(
    answer: str,
    query: str,
    *,
    class_level: str = "",
    subject_name: str = "",
    existing_lesson: dict[str, Any] | None = None,
    allow_fallback: bool = True,
    conversation_history: list[dict[str, str]] | None = None,
    allow_llm_pass2: bool = True,
) -> tuple[str, dict[str, Any] | None]:
    """Extract/build math-lesson grounded on the answer; suppress if type/numbers mismatch."""
    from app.services.interactive_grounding import match_text, resolve_grounded_panel

    clean, lesson = extract_math_lesson_from_answer(answer)
    if lesson is None:
        lesson = existing_lesson
    if not _is_mathematics_subject(subject_name):
        return clean, lesson
    if not allow_fallback and lesson is None:
        return clean, None

    text = match_text(clean, query)
    catalog = (
        build_fallback_math_lesson(
            text, class_level, conversation_history=conversation_history
        )
        if allow_fallback
        else None
    )
    # Soft-merge only when first fence is weak — resolve_grounded_panel is source of truth.
    if lesson is not None and catalog is not None:
        lesson = merge_catalog_visualization(lesson, catalog)

    grounded = resolve_grounded_panel(
        answer=clean,
        query=query,
        kind="math",
        class_level=class_level,
        first_panel=lesson,
        catalog_panel=catalog,
        allow_llm_pass2=allow_llm_pass2 and allow_fallback,
    )
    if grounded is None:
        return clean, None
    lesson = apply_math_lesson_display_policy(grounded, query)
    return clean, inject_sympy_algebra_steps(lesson, query, class_level)


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
