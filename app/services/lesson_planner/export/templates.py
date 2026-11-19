from __future__ import annotations

from typing import Any


def _lines_from_list(items: list[Any], *, prefix: str = "• ") -> list[str]:
    return [f"{prefix}{item}" for item in items if item]


def render_lesson_plan_section(content: dict[str, Any]) -> list[str]:
    if content.get("format") == "markdown" and content.get("markdown"):
        return str(content["markdown"]).splitlines()
    lines = [content.get("title", "Lesson Plan"), ""]
    for obj in content.get("objectives") or []:
        lines.append(f"• Objective: {obj}")
    lines.append("")
    for act in content.get("activities") or []:
        if isinstance(act, dict):
            lines.append(f"— {act.get('title', 'Activity')} ({act.get('duration_minutes', '?')} min)")
            lines.append(f"  {act.get('description', '')}")
        else:
            lines.append(str(act))
    if content.get("assessment_plan"):
        lines.extend(["", "Assessment:", str(content["assessment_plan"])])
    if content.get("revision_plan"):
        lines.extend(["", "Revision:", str(content["revision_plan"])])
    return lines


def render_teaching_notes_section(content: dict[str, Any]) -> list[str]:
    if content.get("format") == "markdown" and content.get("markdown"):
        return str(content["markdown"]).splitlines()
    lines = ["Teaching Notes", ""]
    if content.get("introduction_script"):
        lines.extend(["Introduction:", content["introduction_script"], ""])
    if content.get("teacher_explanation"):
        lines.extend(["Explanation:", content["teacher_explanation"], ""])
    mistakes = content.get("common_mistakes") or []
    if mistakes:
        lines.append("Common mistakes:")
        lines.extend(_lines_from_list(mistakes))
    return lines


def render_examples_section(content: dict[str, Any]) -> list[str]:
    if content.get("format") == "markdown" and content.get("markdown"):
        return str(content["markdown"]).splitlines()
    lines = ["Examples", ""]
    for bucket in ("easy", "medium", "hard", "real_world", "visual_examples"):
        for item in content.get(bucket) or []:
            if isinstance(item, dict):
                lines.append(f"• {item.get('title', 'Example')}: {item.get('explanation', '')}")
    return lines


def render_worksheet_section(content: dict[str, Any]) -> list[str]:
    if content.get("format") == "markdown" and content.get("markdown"):
        return str(content["markdown"]).splitlines()
    lines = ["Worksheet", ""]
    for section, key in (
        ("Fill in the blanks", "fill_blanks"),
        ("True / False", "true_false"),
        ("Short answer", "short_answer"),
        ("Long answer", "long_answer"),
    ):
        items = content.get(key) or []
        if not items:
            continue
        lines.append(section + ":")
        for i, item in enumerate(items, 1):
            q = item.get("question") or item.get("statement") if isinstance(item, dict) else str(item)
            lines.append(f"  {i}. {q}")
        lines.append("")
    return lines


def render_homework_section(content: dict[str, Any]) -> list[str]:
    if content.get("format") == "markdown" and content.get("markdown"):
        return str(content["markdown"]).splitlines()
    lines = ["Homework", ""]
    for key in ("practice_questions", "observation_tasks", "project_work", "reading_assignment", "revision_tasks"):
        for item in content.get(key) or []:
            if isinstance(item, dict):
                text = item.get("question") or item.get("text") or item.get("task") or item.get("description") or ""
                if text:
                    lines.append(f"• {text}")
    return lines


def render_quiz_section(content: dict[str, Any]) -> list[str]:
    if content.get("format") == "markdown" and content.get("markdown"):
        return str(content["markdown"]).splitlines()
    lines = ["Quiz", ""]
    for i, item in enumerate(content.get("mcq") or [], 1):
        q = item.get("question", "")
        opts = item.get("options") or []
        lines.append(f"{i}. {q}")
        for opt in opts:
            lines.append(f"   - {opt}")
    answers = content.get("answers") or {}
    if answers:
        lines.extend(["", "Answer key:", str(answers)])
    return lines


def render_ppt_outline_section(content: dict[str, Any]) -> list[str]:
    if content.get("format") == "markdown" and content.get("markdown"):
        return str(content["markdown"]).splitlines()
    lines = ["Presentation Outline", ""]
    for item in content.get("slides") or []:
        if isinstance(item, dict):
            lines.append(f"Slide {item.get('number', '?')}: {item.get('title', 'Slide')}")
            for bullet in item.get("bullets") or []:
                lines.append(f"  • {bullet}")
            lines.append("")
    return lines


def ppt_slides_from_markdown(md: str) -> list[dict[str, Any]]:
    """Backward-compatible wrapper — structured parse lives in deck_schema."""
    from app.services.lesson_planner.export.deck_schema import ppt_slides_from_markdown as _parse

    return _parse(md)


def render_artifact_lines(artifact_key: str, content: Any) -> list[str]:
    if not isinstance(content, dict):
        return [str(content)]
    renderers = {
        "lesson_plan": render_lesson_plan_section,
        "teaching_notes": render_teaching_notes_section,
        "examples": render_examples_section,
        "worksheet": render_worksheet_section,
        "quiz": render_quiz_section,
        "homework": render_homework_section,
        "ppt_outline": render_ppt_outline_section,
    }
    renderer = renderers.get(artifact_key)
    if renderer:
        return renderer(content)
    return [artifact_key.replace("_", " ").title(), "", str(content)[:6000]]
