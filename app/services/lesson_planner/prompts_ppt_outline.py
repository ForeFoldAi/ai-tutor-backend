from __future__ import annotations

import json

from app.services.lesson_planner.prompt_focus import topics_focus_block
from app.services.lesson_planner.state import PlannerState

PPT_OUTLINE_SYSTEM = """You are an expert instructional designer for K-12 classroom PowerPoint decks.

Create a STUDENT-FRIENDLY presentation that becomes a professional .pptx.
Write for the back of the room: big ideas, short lines, clear side headings.

HARD RULES:
1. Do NOT return JSON. Markdown only. No code fences.
2. Produce EXACTLY the requested number of slides — no more, no fewer.
3. Max 5 bullets per slide; ~8–12 words per bullet.
4. One idea per slide. Easy language for the grade.
5. Use textbook content as the primary source.
6. Slide 1 title MUST be the chapter/lesson name — NEVER write "Title Slide", "Welcome", or "Introduction".
7. Every non-title slide MUST have real Slide Content bullets (at least 2). Never leave a slide empty.
8. Callout text must be plain words — no markdown (** or *).
9. Prefer Icon labels. Include at most one figure slide if figures are listed.

OUTPUT HEADER:

# Presentation Outline: <Chapter Name>

Class:
Subject:
Chapter:
Estimated Slides: <exact number>

## Presentation Overview
One short paragraph.

Then EACH slide MUST use this block format:

### Slide <N>: <Meaningful Title — never "Title Slide">

**Layout:** <title|section|bullets|two_column|steps|big_idea|summary|figure>
**Side Heading:** <short eyebrow, e.g. Concept / Try this / Remember>
**Icon:** <engage|example|try|check|diagram|formula|remember>
**Figure:** <file_name — caption>   (only for layout=figure)
**Slide Content:**
- short bullet
**Right:**   (only for two_column)
- short bullet
**Callout:** <one plain memorable line>
**Speaker Notes:** <what the teacher says>

Separate slides with a line containing only ---

For EXACTLY 8 slides use this flow:
1) title (chapter name)  2) big questions OR objectives  3) core concept
4) two_column concept|example  5) steps OR instruments  6) real-life / why it matters
7) practice OR case study  8) summary checklist

For 12 slides, expand with one section divider, one figure (if available), one extra concept, one practice.
For 16 slides, add more worked examples and a short closing thank-you summary.

Layout guide:
- title: opening only — title text = chapter name
- section: short divider with a real part name (use sparingly)
- bullets: teaching points
- two_column: Concept | Example
- steps: 2–4 real numbered steps (never 1)
- big_idea: one large takeaway sentence in Slide Content
- figure: teaching points + textbook diagram
- summary: 3–5 checklist points (not the word "Summary Checklist")
"""

_SUBJECT_HINTS: dict[str, str] = {
    "math": "Use formula slides, step-by-step examples, and visual models.",
    "science": "Use experiment steps, diagrams, and cause-effect.",
    "social": "Use timelines, case studies, and compare columns.",
    "english": "Use vocabulary + example sentence columns.",
    "language": "Use vocabulary, grammar steps, and speaking prompts.",
    "computer": "Use algorithm steps and example columns.",
    "history": "Use timeline steps and source vs interpretation columns.",
    "geography": "Use map concepts and data compare columns.",
}


def _subject_hint(subject: str) -> str:
    key = (subject or "").lower()
    for token, hint in _SUBJECT_HINTS.items():
        if token in key:
            return hint
    return "Adapt layouts to the subject."


def build_ppt_outline_markdown_prompt(state: PlannerState) -> tuple[str, str]:
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""
    chapter = state.get("chapter_name") or ""
    duration = state.get("duration_minutes") or 45
    objectives = state.get("learning_objectives") or ""
    context = state.get("chapter_context") or "No textbook context available."
    template = state.get("ppt_template") or "clean_academic"
    slide_count = int(state.get("ppt_slide_count") or 12)
    figures = state.get("figures") or []

    user = (
        f"Class / Grade: {grade}\n"
        f"Subject: {subject}\n"
        f"Chapter / lesson title: {chapter}\n"
        f"Lesson duration: {duration} minutes\n"
        f"PPT visual template: {template}\n"
        f"REQUIRED SLIDE COUNT: exactly {slide_count} slides. "
        f"Estimated Slides must be {slide_count}. Do not create {slide_count + 1} or more.\n"
        f"Slide 1 title must be exactly: {chapter}\n"
        f"{topics_focus_block(state)}"
        f"Learning objectives:\n{objectives or 'Derive from selected topics / chapter.'}\n\n"
        f"Subject focus: {_subject_hint(subject)}\n\n"
        f"Primary textbook context:\n{context[:14000]}\n"
    )
    if figures:
        compact = [
            {
                "file_name": f.get("file_name"),
                "caption": f.get("caption"),
                "page": f.get("page"),
            }
            for f in figures[:6]
            if isinstance(f, dict)
        ]
        user += (
            "\nAvailable textbook figures (optional one figure slide):\n"
            f"{json.dumps(compact, ensure_ascii=False)[:2200]}\n"
        )
    user += (
        f"\nWrite the complete presentation outline in Markdown now with EXACTLY {slide_count} "
        "### Slide blocks. Use Layout / Side Heading / Icon / Callout. No JSON. "
        'Never title a slide "Title Slide".'
    )
    return PPT_OUTLINE_SYSTEM, user
