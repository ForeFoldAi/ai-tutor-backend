from __future__ import annotations

import json

from app.services.lesson_planner.prompt_focus import topics_focus_block
from app.services.lesson_planner.state import PlannerState

EXAMPLES_SYSTEM = """You are an expert K-12 educator and instructional designer.

Generate Examples for the given chapter.

IMPORTANT:
1. Do NOT generate worksheets, quizzes, homework, or lesson plans.
2. Do NOT return JSON.
3. Do NOT return code blocks wrapping the whole document.
4. Return beautifully formatted Markdown examples directly displayable in the UI.
5. Adapt examples according to subject and grade level.
6. Base examples on textbook content.

Generate these sections (omit sections that do not apply):

# Examples: <Chapter Name>

Class:
Subject:
Chapter:

## Concept Examples

## Worked Examples

## Real-Life Examples

## Visual Examples

## Guided Examples

## Challenge Examples

## Common Mistakes in Examples

Subject adaptation:
- Mathematics: solved problems, word problems, visual models, multiple solution methods
- Science: scientific situations, experiments, observations, cause and effect
- Social Studies: case studies, historical events, maps, current affairs
- English/Languages: grammar, vocabulary, reading passages, writing examples
- Computer Science: coding examples, algorithms, pseudocode

For worked examples show clear step-by-step solutions.
For common mistakes show the wrong approach and the correct approach.
Use age-appropriate language."""

_SUBJECT_HINTS: dict[str, str] = {
    "math": "Include solved problems, word problems, visual models, and alternate methods.",
    "science": "Include experiments, observations, and cause-effect situations.",
    "social": "Include case studies, maps, timelines, and current affairs links.",
    "english": "Include grammar, vocabulary, reading, and writing samples.",
    "language": "Include grammar, vocabulary, reading, and writing samples.",
    "computer": "Include coding snippets, algorithms, and pseudocode.",
    "history": "Include historical events and source-based examples.",
    "geography": "Include maps, data, and diagram suggestions.",
}


def _subject_hint(subject: str) -> str:
    key = (subject or "").lower()
    for token, hint in _SUBJECT_HINTS.items():
        if token in key:
            return hint
    return "Adapt example types to the subject naturally."


def build_examples_markdown_prompt(state: PlannerState) -> tuple[str, str]:
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""
    chapter = state.get("chapter_name") or ""
    objectives = state.get("learning_objectives") or ""
    context = state.get("chapter_context") or "No textbook context available."
    figures = state.get("figures") or []

    user = (
        f"Class / Grade: {grade}\n"
        f"Subject: {subject}\n"
        f"Chapter: {chapter}\n"
        f"{topics_focus_block(state)}"
        f"Learning objectives:\n{objectives or 'Derive from chapter content.'}\n\n"
        f"Subject focus: {_subject_hint(subject)}\n\n"
        f"Primary textbook context:\n{context[:14000]}\n"
    )
    if figures:
        user += f"\nTextbook figures to reference:\n{json.dumps(figures[:8], ensure_ascii=False)[:2500]}\n"
    user += "\nWrite complete Examples in Markdown now. Examples only — no worksheets or quizzes."
    return EXAMPLES_SYSTEM, user
