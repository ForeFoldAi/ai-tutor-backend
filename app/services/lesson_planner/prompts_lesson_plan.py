from __future__ import annotations

import json

from app.services.lesson_planner.prompt_focus import topics_focus_block
from app.services.lesson_planner.state import PlannerState

LESSON_PLAN_SYSTEM = """You are an expert curriculum designer, instructional coach, and K-12 lesson planning specialist.

Create a professional, teacher-ready lesson plan that can work for ANY subject:
Mathematics, Science, Social Studies, English, Languages, Computer Science, EVS, General Knowledge, Commerce, History, Geography, Civics, Economics.

IMPORTANT RULES:
1. Do NOT return JSON.
2. Do NOT return code blocks.
3. Return a beautifully formatted lesson plan in Markdown.
4. The lesson should be directly displayable in the Lesson Planner UI.
5. Adapt the lesson structure based on the subject and chapter.
6. Include only sections that are relevant to the subject.
7. Follow modern pedagogy and the 5E model whenever appropriate: Engage, Explore, Explain, Elaborate, Evaluate.
8. Use age-appropriate language.
9. Use textbook content as the primary source.
10. Include activities, assessments, and differentiated instruction.

The response MUST contain these sections (omit subsections that do not apply):

# Lesson Plan: <Chapter Name>

Class:
Subject:
Chapter:
Duration:

## Learning Objectives

## Prerequisite Knowledge

## Materials Required

## Key Vocabulary

## Lesson Flow

For each lesson phase include:
- Duration
- Teacher Activity
- Student Activity
- Guiding Questions
- Expected Student Responses
- Assessment Checkpoints
- Resources Needed
- Expected Outcome

## Real-Life Connections

## Common Misconceptions

## Assessment
- Formative Assessment
- Summative Assessment
- Exit Ticket
- Rubric

## Differentiated Instruction
- Support for Struggling Learners
- Support for Advanced Learners

## Homework

## Teacher Reflection

## Suggested Digital Resources
- Images
- Animations
- Simulations
- Videos
- Interactive Activities

Subject-specific guidance (include only what fits the subject):
- Science: experiments, safety instructions, observations, lab activities
- Mathematics: worked examples, step-by-step problem solving, visual models, practice problems, common calculation errors
- Social Studies: maps, timelines, historical sources, case studies, discussions and debates
- English/Languages: reading, vocabulary, grammar, speaking, writing, creative exercises
- Computer Science: coding activities, algorithms, projects, practical exercises
- Geography: maps, diagrams, data interpretation
- History: historical events, timelines, source analysis, role play

Only include sections relevant to the current subject. Write like a professional teacher handbook."""

_SUBJECT_HINTS: dict[str, str] = {
    "science": "Emphasize experiments, safety, observations, and lab activities.",
    "mathematics": "Emphasize worked examples, visual models, step-by-step solving, and common errors.",
    "math": "Emphasize worked examples, visual models, step-by-step solving, and common errors.",
    "social": "Emphasize maps, timelines, sources, case studies, and debates.",
    "english": "Emphasize reading, vocabulary, grammar, speaking, writing, and creative tasks.",
    "language": "Emphasize reading, vocabulary, grammar, speaking, writing, and creative tasks.",
    "computer": "Emphasize coding activities, algorithms, projects, and hands-on practice.",
    "history": "Emphasize timelines, source analysis, historical events, and role play.",
    "geography": "Emphasize maps, diagrams, and data interpretation.",
    "civics": "Emphasize case studies, discussions, and real-world governance examples.",
    "economics": "Emphasize data interpretation, case studies, and real-life applications.",
}


def _subject_hint(subject: str) -> str:
    key = (subject or "").lower()
    for token, hint in _SUBJECT_HINTS.items():
        if token in key:
            return hint
    return "Adapt activities and resources to the subject naturally."


def build_lesson_plan_markdown_prompt(state: PlannerState) -> tuple[str, str]:
    grade = state.get("grade") or ""
    subject = state.get("subject") or ""
    chapter = state.get("chapter_name") or ""
    duration = state.get("duration_minutes") or 45
    objectives = state.get("learning_objectives") or ""
    context = state.get("chapter_context") or "No textbook context available."
    figures = state.get("figures") or []
    experiments = state.get("experiments") or []
    pedagogy = (state.get("metadata") or {}).get("pedagogy") or {}

    user = (
        f"Class / Grade: {grade}\n"
        f"Subject: {subject}\n"
        f"Chapter: {chapter}\n"
        f"Duration: {duration} minutes\n"
        f"{topics_focus_block(state)}"
        f"Learning objectives from teacher:\n{objectives or 'Derive appropriate objectives from the selected topics / chapter.'}\n\n"
        f"Subject focus: {_subject_hint(subject)}\n\n"
        f"Pedagogy analysis:\n{json.dumps(pedagogy, ensure_ascii=False)[:3000]}\n\n"
        f"Primary textbook context (ground all content here):\n{context[:14000]}\n"
    )
    if figures:
        user += f"\nAvailable textbook figures to reference:\n{json.dumps(figures[:8], ensure_ascii=False)[:2500]}\n"
    if experiments:
        user += f"\nInteractive experiments to reference:\n{json.dumps(experiments[:3], ensure_ascii=False)[:2000]}\n"
    user += "\nWrite the complete lesson plan in Markdown now. No JSON. No code fences wrapping the whole document."
    return LESSON_PLAN_SYSTEM, user
