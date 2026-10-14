from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ActivityItem(BaseModel):
    title: str
    duration_minutes: int = 10
    description: str = ""
    activity_type: str = "instruction"


class LessonPlanOutput(BaseModel):
    title: str
    duration: int
    objectives: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    activities: list[ActivityItem] = Field(default_factory=list)
    assessment_plan: str = ""
    revision_plan: str = ""
    visuals: list[str] = Field(default_factory=list)


class TeachingNotesOutput(BaseModel):
    introduction_script: str = ""
    teacher_explanation: str = ""
    common_mistakes: list[str] = Field(default_factory=list)
    real_life_connections: list[str] = Field(default_factory=list)
    questions_to_ask: list[str] = Field(default_factory=list)
    blackboard_flow: list[str] = Field(default_factory=list)


class ExamplesOutput(BaseModel):
    easy: list[dict[str, Any]] = Field(default_factory=list)
    medium: list[dict[str, Any]] = Field(default_factory=list)
    hard: list[dict[str, Any]] = Field(default_factory=list)
    real_world: list[dict[str, Any]] = Field(default_factory=list)
    visual_examples: list[dict[str, Any]] = Field(default_factory=list)


class WorksheetOutput(BaseModel):
    fill_blanks: list[dict[str, Any]] = Field(default_factory=list)
    true_false: list[dict[str, Any]] = Field(default_factory=list)
    match_following: list[dict[str, Any]] = Field(default_factory=list)
    short_answer: list[dict[str, Any]] = Field(default_factory=list)
    long_answer: list[dict[str, Any]] = Field(default_factory=list)
    application_questions: list[dict[str, Any]] = Field(default_factory=list)


class QuizOutput(BaseModel):
    mcq: list[dict[str, Any]] = Field(default_factory=list)
    short_answer: list[dict[str, Any]] = Field(default_factory=list)
    hots_questions: list[dict[str, Any]] = Field(default_factory=list)
    assertion_reason: list[dict[str, Any]] = Field(default_factory=list)
    answers: dict[str, Any] = Field(default_factory=dict)


class HomeworkOutput(BaseModel):
    practice_questions: list[dict[str, Any]] = Field(default_factory=list)
    observation_tasks: list[dict[str, Any]] = Field(default_factory=list)
    project_work: list[dict[str, Any]] = Field(default_factory=list)
    reading_assignment: list[dict[str, Any]] = Field(default_factory=list)
    revision_tasks: list[dict[str, Any]] = Field(default_factory=list)


class PptSlideOutput(BaseModel):
    number: int
    title: str
    bullets: list[str] = Field(default_factory=list)
    speaker_notes: str = ""


class PptOutlineOutput(BaseModel):
    slides: list[PptSlideOutput] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    animations: list[str] = Field(default_factory=list)
    speaker_notes: list[str] = Field(default_factory=list)


ARTIFACT_SCHEMAS = {
    "lesson_plan": LessonPlanOutput,
    "teaching_notes": TeachingNotesOutput,
    "examples": ExamplesOutput,
    "worksheet": WorksheetOutput,
    "quiz": QuizOutput,
    "homework": HomeworkOutput,
    "ppt_outline": PptOutlineOutput,
}
