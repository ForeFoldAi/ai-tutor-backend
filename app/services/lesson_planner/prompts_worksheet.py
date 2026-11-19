from __future__ import annotations

import json

from app.services.lesson_planner.prompt_focus import topics_focus_block
from app.services.lesson_planner.state import PlannerState

WORKSHEET_SYSTEM = """You are an expert K-12 worksheet designer.

Generate a student worksheet for the given chapter.

IMPORTANT:
1. Do NOT include answers, answer keys, or solutions.
2. Do NOT generate quizzes, lesson plans, teaching notes, or homework assignments.
3. Do NOT return JSON.
4. Do NOT wrap the document in code blocks.
5. Return a beautifully formatted, printable Markdown worksheet.
6. Adapt questions to subject and grade level.
7. Base content on textbook material.

Generate:

# Worksheet: <Chapter Name>

Class:
Subject:
Chapter:

## Warm-Up Questions

## Practice Questions

## Application Questions

## Higher-Order Thinking Questions

## Activity-Based Questions

## Reflection Questions

Subject adaptation (include only relevant types):
- Mathematics: calculations, word problems, puzzles, reasoning
- Science: observations, diagrams to label, experiments, data interpretation
- Social Studies: map activities, timelines, case studies, source analysis
- English/Languages: grammar, vocabulary, reading comprehension, writing prompts
- Computer Science: tracing algorithms, coding exercises, debugging

Number questions clearly within each section.
Leave space cues like "Answer: _______________" where students write responses.
Do not reveal correct answers anywhere."""

_SUBJECT_HINTS: dict[str, str] = {
    "math": "Use calculations, word problems, puzzles, and reasoning questions.",
    "science": "Use observations, labeling diagrams, and experiment-style prompts.",
    "social": "Use maps, timelines, case studies, and source analysis.",
    "english": "Use grammar, vocabulary, comprehension, and short writing prompts.",
    "language": "Use grammar, vocabulary, comprehension, and short writing prompts.",
    "computer": "Use algorithm tracing, coding, and debugging exercises.",
}


def _subject_hint(subject: str) -> str:
    key = (subject or "").lower()
    for token, hint in _SUBJECT_HINTS.items():
        if token in key:
            return hint
    return "Adapt question types to the subject naturally."


def build_worksheet_markdown_prompt(state: PlannerState) -> tuple[str, str]:
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""
    chapter = state.get("chapter_name") or ""
    objectives = state.get("learning_objectives") or ""
    context = state.get("chapter_context") or "No textbook context available."

    user = (
        f"Class / Grade: {grade}\n"
        f"Subject: {subject}\n"
        f"Chapter: {chapter}\n"
        f"{topics_focus_block(state)}"
        f"Learning objectives:\n{objectives or 'Derive from chapter content.'}\n\n"
        f"Subject focus: {_subject_hint(subject)}\n\n"
        f"Primary textbook context:\n{context[:14000]}\n\n"
        "Write the complete student worksheet in Markdown now. "
        "Questions only — no answers, no answer key."
    )
    return WORKSHEET_SYSTEM, user
