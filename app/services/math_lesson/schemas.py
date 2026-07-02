"""Pydantic schemas for interactive mathematics lessons."""

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


class DraggableObjectSpec(BaseModel):
    id: str
    label: str
    type: str = "point"
    initialX: float = 0
    initialY: float = 0
    color: str = "#3B82F6"


class InteractiveObjectSpec(BaseModel):
    id: str
    label: str
    type: str = "shape"
    properties: dict[str, Any] = Field(default_factory=dict)


class AnimationSpec(BaseModel):
    id: str
    trigger: str = "onLoad"
    description: str = ""
    duration: float = 1.0


class LiveCalculationSpec(BaseModel):
    id: str
    label: str
    formula: str
    unit: str = ""


class LabelSpec(BaseModel):
    id: str
    text: str
    x: float | None = None
    y: float | None = None


class ColorSpec(BaseModel):
    primary: str = "#3B82F6"
    secondary: str = "#10B981"
    accent: str = "#F59E0B"
    background: str = "#F8FAFC"
    text: str = "#1E293B"


class StudentInteractionSpec(BaseModel):
    id: str
    type: str
    description: str
    expectedObservation: str = ""


class VisualizationSpec(BaseModel):
    visualizationType: str
    title: str
    description: str = ""
    interactiveObjects: list[InteractiveObjectSpec] = Field(default_factory=list)
    draggableObjects: list[DraggableObjectSpec] = Field(default_factory=list)
    sliders: list[SliderSpec] = Field(default_factory=list)
    buttons: list[ButtonSpec] = Field(default_factory=list)
    animations: list[AnimationSpec] = Field(default_factory=list)
    liveCalculations: list[LiveCalculationSpec] = Field(default_factory=list)
    labels: list[LabelSpec] = Field(default_factory=list)
    colors: ColorSpec = Field(default_factory=ColorSpec)
    studentInteractions: list[StudentInteractionSpec] = Field(default_factory=list)


class PracticeModeSpec(BaseModel):
    easy: str = ""
    medium: str = ""
    hard: str = ""
    challenge: str = ""


class AssessmentQuestion(BaseModel):
    question: str
    type: str = "conceptual"


class MathLesson(BaseModel):
    conceptName: str
    classLevel: str = ""
    learningObjective: str = ""
    conceptExplanation: str = ""
    visualization: VisualizationSpec
    guidedExploration: list[str] = Field(default_factory=list)
    practiceMode: PracticeModeSpec = Field(default_factory=PracticeModeSpec)
    commonMistakes: list[str] = Field(default_factory=list)
    aiHints: list[list[str]] = Field(default_factory=list)
    assessment: list[AssessmentQuestion] = Field(default_factory=list)
