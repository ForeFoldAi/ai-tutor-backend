from __future__ import annotations

import json

from app.services.lesson_planner.prompt_focus import topics_focus_block
from app.services.lesson_planner.state import PlannerState

TEACHING_NOTES_SYSTEM = """You are an expert instructional coach, curriculum designer, teacher trainer, and K-12 education specialist.

Your task is to create PROFESSIONAL TEACHING NOTES for teachers.

The Teaching Notes are NOT a lesson plan.
They explain HOW TO TEACH the chapter, WHAT TO EMPHASIZE, and HOW TO HELP STUDENTS UNDERSTAND DIFFICULT CONCEPTS.
The response should feel like a teacher handbook or teacher guide.

IMPORTANT RULES:
1. Do NOT return JSON.
2. Do NOT return code blocks.
3. Do NOT create a lesson schedule.
4. Do NOT include timings.
5. Do NOT include lesson phases (Engage, Explore, Explain, Elaborate, Evaluate).
6. Do NOT include Lesson Flow, Activities Timeline, Assessment Schedule, or Homework Schedule.
7. Do NOT repeat the lesson plan.
8. Return beautifully formatted Markdown directly displayable in the UI.
9. Use textbook content as the primary source.
10. Only include sections relevant to the subject and chapter.

OUTPUT FORMAT:

# Teaching Notes: <Chapter Name>

Class:
Subject:
Chapter:

## Chapter at a Glance

## Key Teaching Takeaways

## Essential Concepts to Emphasize

## Teacher Preparation Guide

## Materials & Resources

## Background Knowledge for Teachers

## Student-Friendly Explanations

## Teaching Strategies

## Suggested Classroom Dialogue

## Questions to Stimulate Thinking

## Anticipated Student Responses

## Common Misconceptions & Remedies

## Difficult Areas and Scaffolding Strategies

## Real-World Connections

## Cross-Curricular Connections

## Differentiation Suggestions

### Support for Struggling Learners

### Extension for Advanced Learners

## Teaching Tips & Best Practices

## Interesting Facts & Extensions

## Digital Teaching Resources

## Post-Lesson Reflection Questions

Subject adaptation (include only what fits):
- Mathematics: formulas, worked examples, visual models, common calculation mistakes, alternative methods
- Science: experiments, observations, scientific reasoning, safety precautions
- Social Studies: maps, timelines, case studies, primary sources, discussions
- English/Languages: grammar, vocabulary, pronunciation, reading/writing/speaking strategies
- Computer Science: algorithms, coding examples, debugging tips, practical applications

For Common Misconceptions use:
Misconception:
Why students think this:
Correction Strategy:

For Classroom Dialogue use Teacher / Student / Teacher Follow-up lines.

The final response must be completely different from a lesson plan."""

_SUBJECT_HINTS: dict[str, str] = {
    "math": "Emphasize formulas, worked examples, visual models, and common calculation mistakes.",
    "science": "Emphasize experiments, observations, scientific reasoning, and safety.",
    "social": "Emphasize maps, timelines, case studies, and source-based discussion.",
    "english": "Emphasize grammar, vocabulary, reading/writing/speaking strategies.",
    "language": "Emphasize grammar, vocabulary, pronunciation, and communication skills.",
    "computer": "Emphasize algorithms, coding examples, debugging, and practical use.",
    "history": "Emphasize timelines, source analysis, and historical significance.",
    "geography": "Emphasize maps, diagrams, and data interpretation.",
    "civics": "Emphasize civic concepts, case studies, and real governance examples.",
    "economics": "Emphasize real-world applications and data interpretation.",
}


def _subject_hint(subject: str) -> str:
    key = (subject or "").lower()
    for token, hint in _SUBJECT_HINTS.items():
        if token in key:
            return hint
    return "Adapt emphasis and strategies to the subject naturally."


def build_teaching_notes_markdown_prompt(state: PlannerState) -> tuple[str, str]:
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""
    chapter = state.get("chapter_name") or ""
    objectives = state.get("learning_objectives") or ""
    context = state.get("chapter_context") or "No textbook context available."
    figures = state.get("figures") or []
    experiments = state.get("experiments") or []
    pedagogy = (state.get("metadata") or {}).get("pedagogy") or {}
    misconceptions = pedagogy.get("misconceptions") or []

    user = (
        f"Class / Grade: {grade}\n"
        f"Subject: {subject}\n"
        f"Chapter: {chapter}\n"
        f"{topics_focus_block(state)}"
        f"Learning objectives from teacher:\n{objectives or 'Derive appropriate focus areas from the chapter.'}\n\n"
        f"Subject focus: {_subject_hint(subject)}\n\n"
    )
    if misconceptions:
        user += f"Known misconceptions to address:\n{json.dumps(misconceptions[:8], ensure_ascii=False)}\n\n"
    user += (
        f"Pedagogy analysis:\n{json.dumps(pedagogy, ensure_ascii=False)[:3000]}\n\n"
        f"Primary textbook context (ground all content here):\n{context[:14000]}\n"
    )
    if figures:
        user += f"\nAvailable textbook figures:\n{json.dumps(figures[:8], ensure_ascii=False)[:2500]}\n"
    if experiments:
        user += f"\nInteractive experiments:\n{json.dumps(experiments[:3], ensure_ascii=False)[:2000]}\n"
    user += (
        "\nWrite complete Teaching Notes in Markdown now. "
        "No JSON. No code fences wrapping the whole document. "
        "No lesson schedule, timings, or 5E phases."
    )
    return TEACHING_NOTES_SYSTEM, user
