from __future__ import annotations

import re
import textwrap
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ARTIFACT_TITLES: dict[str, str] = {
    "lesson_plan": "Lesson Plan",
    "teaching_notes": "Teaching Notes",
    "examples": "Examples",
    "worksheet": "Worksheet",
    "quiz": "Quiz",
    "homework": "Homework",
    "ppt_outline": "Presentation Outline",
}

_MARGIN_X = 50
_MARGIN_TOP = 50
_MARGIN_BOTTOM = 55
_LINE_LEADING = 14


def _page_size() -> tuple[float, float]:
    return A4


def _strip_md_bold(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text)


def _is_table_sep(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.match(r"^:?-{2,}:?$", c) for c in cells if c)


def _wrap_draw(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    *,
    font: str,
    size: int,
    max_chars: int = 95,
) -> float:
    c.setFont(font, size)
    for part in textwrap.wrap(_strip_md_bold(text), width=max_chars) or [""]:
        if y < _MARGIN_BOTTOM:
            c.showPage()
            _, height = _page_size()
            y = height - _MARGIN_TOP
            c.setFont(font, size)
        c.drawString(x, y, part)
        y -= _LINE_LEADING
    return y


def draw_markdown_pdf(c: canvas.Canvas, markdown: str, y: float) -> float:
    """Render markdown content onto a reportlab canvas; returns final y."""
    _, height = _page_size()
    for raw in markdown.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            y -= 8
            continue

        if stripped == "---":
            y -= 6
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            if _is_table_sep(stripped):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            stripped = "  |  ".join(cells)

        if stripped.startswith("# "):
            y -= 6
            y = _wrap_draw(c, stripped[2:], _MARGIN_X, y, font="Helvetica-Bold", size=16)
            y -= 4
            continue
        if stripped.startswith("## "):
            y -= 4
            y = _wrap_draw(c, stripped[3:], _MARGIN_X, y, font="Helvetica-Bold", size=13)
            y -= 2
            continue
        if stripped.startswith("### "):
            y -= 2
            y = _wrap_draw(c, stripped[4:], _MARGIN_X, y, font="Helvetica-Bold", size=11)
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            y = _wrap_draw(c, f"• {stripped[2:]}", _MARGIN_X + 8, y, font="Helvetica", size=10)
            continue
        if re.match(r"^\d+\.\s", stripped):
            y = _wrap_draw(c, stripped, _MARGIN_X + 4, y, font="Helvetica", size=10)
            continue
        if stripped.startswith("**") and "**" in stripped[2:]:
            y = _wrap_draw(c, stripped, _MARGIN_X, y, font="Helvetica-Bold", size=10)
            continue

        y = _wrap_draw(c, stripped, _MARGIN_X, y, font="Helvetica", size=10)

    return y


def export_pdf_document(
    path: Any,
    *,
    title: str,
    subtitle: str,
    sections: list[tuple[str, Any]],
) -> None:
    """Write a formatted PDF. sections: [(artifact_key, content), ...]."""
    from app.services.lesson_planner.export.templates import render_artifact_lines

    c = canvas.Canvas(str(path), pagesize=_page_size())
    _, height = _page_size()
    y = height - _MARGIN_TOP

    y = _wrap_draw(c, title, _MARGIN_X, y, font="Helvetica-Bold", size=18, max_chars=60)
    y -= 4
    y = _wrap_draw(c, subtitle, _MARGIN_X, y, font="Helvetica", size=10, max_chars=100)
    y -= 16

    for artifact_key, content in sections:
        section_title = ARTIFACT_TITLES.get(artifact_key, artifact_key.replace("_", " ").title())
        if y < _MARGIN_BOTTOM + 40:
            c.showPage()
            y = height - _MARGIN_TOP

        if len(sections) > 1:
            y = _wrap_draw(c, section_title, _MARGIN_X, y, font="Helvetica-Bold", size=14)
            y -= 8

        if isinstance(content, dict) and content.get("format") == "markdown" and content.get("markdown"):
            y = draw_markdown_pdf(c, str(content["markdown"]), y)
        else:
            for line in render_artifact_lines(artifact_key, content):
                y = _wrap_draw(c, line, _MARGIN_X + 4, y, font="Helvetica", size=10)
        y -= 12

    c.save()
