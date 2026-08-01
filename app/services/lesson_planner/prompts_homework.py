from __future__ import annotations

from app.services.lesson_planner.prompt_focus import topics_focus_block
from app.services.lesson_planner.state import PlannerState

HOMEWORK_SYSTEM = """You are an expert K-12 educator.

Generate meaningful homework for the given chapter.

IMPORTANT:
1. Homework should promote independent learning.
2. Homework should NOT be a worksheet (no long lists of drill questions).
3. Homework should NOT repeat quiz-style MCQs.
4. Encourage observation, creativity, and real-world connections.
5. Do NOT return JSON.
6. Do NOT wrap the document in code blocks.
7. Return beautifully formatted Markdown directly displayable in the UI.
8. Tasks must be achievable at home, engaging, age-appropriate, and curriculum aligned.

Generate (include only relevant sections):

# Homework: <Chapter Name>

Class:
Subject:
Chapter:

## Practice Tasks

## Real-Life Tasks

## Observation Tasks

## Creative Tasks

## Mini Projects

## Research Tasks

## Reflection Tasks

Subject adaptation:
- Mathematics: practice problems, real-life calculations, puzzles (not repetitive drills)
- Science: observations, simple experiments, data collection at home
- Social Studies: timelines, map work, interviews, research activities
- English/Languages: reading, writing, vocabulary, speaking activities
- Computer Science: coding practice, debugging exercises, mini projects

Use bullet points with clear, actionable task descriptions."""

_SUBJECT_HINTS: dict[str, str] = {
    "math": "Include real-life calculations and puzzles, not worksheet-style drills.",
    "science": "Include home-safe observations and simple experiments.",
    "social": "Include timelines, map work, interviews, or short research.",
    "english": "Include reading, writing, vocabulary, and speaking tasks.",
    "language": "Include reading, writing, vocabulary, and speaking tasks.",
    "computer": "Include coding practice, debugging, and mini projects.",
}


def _subject_hint(subject: str) -> str:
    key = (subject or "").lower()
    for token, hint in _SUBJECT_HINTS.items():
        if token in key:
            return hint
    return "Adapt homework tasks to the subject naturally."


def build_homework_markdown_prompt(state: PlannerState) -> tuple[str, str]:
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
        "Write complete homework in Markdown now. "
        "Meaningful independent-learning tasks only — not a worksheet or quiz."
    )
    return HOMEWORK_SYSTEM, user
