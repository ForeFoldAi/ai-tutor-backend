"""Pydantic schemas for interactive science experiments (three-view synchronized model)."""

from __future__ import annotations

from typing import Any

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
    primary: str = "#0EA5E9"
    secondary: str = "#10B981"
    accent: str = "#F59E0B"
    background: str = "#F0F9FF"
    text: str = "#0F172A"


class ExperimentViewSpec(BaseModel):
    """One synchronized view: real-world, microscopic, or scientific."""

    title: str = ""
    description: str = ""
    narration: str = ""
    equation: str = ""
    labels: list[str] = Field(default_factory=list)


class ThreeViewSpec(BaseModel):
    realWorld: ExperimentViewSpec = Field(default_factory=ExperimentViewSpec)
    microscopic: ExperimentViewSpec = Field(default_factory=ExperimentViewSpec)
    scientific: ExperimentViewSpec = Field(default_factory=ExperimentViewSpec)


class ExperimentSpec(BaseModel):
    experimentType: str
    title: str
    description: str = ""
    gradeTier: str = "middle"
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
    learningObjective: str = ""
    conceptExplanation: str = ""
    experiment: ExperimentSpec
    guidedExploration: list[str] = Field(default_factory=list)
    practiceMode: PracticeModeSpec = Field(default_factory=PracticeModeSpec)
    commonMistakes: list[str] = Field(default_factory=list)
    aiHints: list[list[str]] = Field(default_factory=list)
    assessment: list[AssessmentQuestion] = Field(default_factory=list)
