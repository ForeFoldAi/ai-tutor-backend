from __future__ import annotations

from app.services.lesson_planner.prompt_focus import topics_focus_block
from app.services.lesson_planner.state import PlannerState

QUIZ_SYSTEM = """You are an expert assessment designer.

Generate a multiple-choice quiz for the given chapter.

IMPORTANT:
1. Generate ONLY multiple-choice questions (MCQs).
2. Every question must have exactly four options labeled A, B, C, and D.
3. Provide the correct answer for each question (A, B, C, or D).
4. Do NOT include True/False, fill in the blanks, short answer, or long answer questions.
5. Do NOT return JSON.
6. Do NOT wrap the document in code blocks.
7. Return beautifully formatted Markdown directly displayable in the UI.

Format:

# Quiz: <Chapter Name>

Class:
Subject:
Chapter:
Total Questions: <number>

## Multiple Choice Questions

1. <question text>
A. <option>
B. <option>
C. <option>
D. <option>
Answer: <A|B|C|D>

2. ...

## Answer Key

1. <letter>
2. <letter>
...

Requirements:
- Textbook aligned
- Cover multiple concepts from the chapter
- Include application questions
- Avoid repetition
- Age appropriate

Subject adaptation:
- Mathematics: calculations, reasoning, application problems
- Science: concepts, experiments, observations
- Social Studies: events, maps, case studies
- English/Languages: grammar, vocabulary, comprehension
- Computer Science: concepts, coding logic, algorithms

Generate at least 8 MCQs unless the chapter is very narrow."""

_SUBJECT_HINTS: dict[str, str] = {
    "math": "Focus on calculations, reasoning, and application MCQs.",
    "science": "Focus on concepts, experiments, and observation MCQs.",
    "social": "Focus on events, maps, and case-study MCQs.",
    "english": "Focus on grammar, vocabulary, and comprehension MCQs.",
    "language": "Focus on grammar, vocabulary, and comprehension MCQs.",
    "computer": "Focus on coding logic, algorithms, and concept MCQs.",
}


def _subject_hint(subject: str) -> str:
    key = (subject or "").lower()
    for token, hint in _SUBJECT_HINTS.items():
        if token in key:
            return hint
    return "Adapt MCQ content to the subject naturally."


def build_quiz_markdown_prompt(state: PlannerState) -> tuple[str, str]:
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
        "Write the complete MCQ quiz in Markdown now. MCQs only with A/B/C/D options and answers."
    )
    return QUIZ_SYSTEM, user
