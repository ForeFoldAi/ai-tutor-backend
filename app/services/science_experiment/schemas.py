"""Pydantic schemas for interactive science experiments (three-view + lab model)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SliderSpec(BaseModel):
    id: str
    label: str
    min: float = 0
    max: float = 100
    step: float = 1
    default: float = 50
    unit: str = ""


class ButtonSpec(BaseModel):
    id: str
    label: str
    action: str = "reset"


class LiveCalculationSpec(BaseModel):
    id: str
    label: str
    formula: str
    unit: str = ""


class AnimationSpec(BaseModel):
    id: str
    trigger: str = "onLoad"
    description: str = ""
    duration: float = 1.0


class StudentInteractionSpec(BaseModel):
    id: str
    type: str
    description: str
    expectedObservation: str = ""


class ColorSpec(BaseModel):
    primary: str = "#2d70b3"
    secondary: str = "#388c46"
    accent: str = "#e08a2b"
    background: str = "#fafafa"
    text: str = "#1a1a1f"


class ExperimentViewSpec(BaseModel):
    title: str = ""
    description: str = ""
    narration: str = ""
    equation: str = ""
    labels: list[str] = Field(default_factory=list)


class ThreeViewSpec(BaseModel):
    realWorld: ExperimentViewSpec = Field(default_factory=ExperimentViewSpec)
    microscopic: ExperimentViewSpec = Field(default_factory=ExperimentViewSpec)
    scientific: ExperimentViewSpec = Field(default_factory=ExperimentViewSpec)


class ExperimentStep(BaseModel):
    id: str
    instruction: str
    safetyNote: str | None = None
    durationSeconds: float | None = None


class ExperimentSpec(BaseModel):
    experimentType: str
    title: str
    description: str = ""
    gradeTier: str = "middle"
    subject: Literal["physics", "chemistry", "biology", "evs"] | str = "evs"
    kind: Literal["concept", "experiment"] | str = "concept"
    aim: str = ""
    apparatus: list[str] = Field(default_factory=list)
    safetyLevel: Literal["none", "caution", "adult-supervision"] | str = "none"
    threeViews: ThreeViewSpec = Field(default_factory=ThreeViewSpec)
    sliders: list[SliderSpec] = Field(default_factory=list)
    buttons: list[ButtonSpec] = Field(default_factory=list)
    animations: list[AnimationSpec] = Field(default_factory=list)
    liveCalculations: list[LiveCalculationSpec] = Field(default_factory=list)
    colors: ColorSpec = Field(default_factory=ColorSpec)
    studentInteractions: list[StudentInteractionSpec] = Field(default_factory=list)
    safetyNotes: list[str] = Field(default_factory=list)
    hypothesisPrompt: str = ""
    procedure: list[str] = Field(default_factory=list)
    procedureSteps: list[ExperimentStep] = Field(default_factory=list)
    expectedObservation: str = ""
    explanation: str = ""
    relatedVisualizationType: str | None = None


class PracticeModeSpec(BaseModel):
    easy: str = ""
    medium: str = ""
    hard: str = ""
    challenge: str = ""


class AssessmentQuestion(BaseModel):
    question: str
    type: str = "conceptual"


class ScienceExperiment(BaseModel):
    conceptName: str
    classLevel: str = ""
    subject: Literal["physics", "chemistry", "biology", "evs"] | str = "evs"
    kind: Literal["concept", "experiment"] | str = "concept"
    learningObjective: str = ""
    conceptExplanation: str = ""
    experiment: ExperimentSpec
    guidedExploration: list[str] = Field(default_factory=list)
    practiceMode: PracticeModeSpec = Field(default_factory=PracticeModeSpec)
    commonMistakes: list[str] = Field(default_factory=list)
    aiHints: list[list[str]] = Field(default_factory=list)
    assessment: list[AssessmentQuestion] = Field(default_factory=list)
