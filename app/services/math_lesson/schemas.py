"""Pydantic schemas for interactive mathematics lessons."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.services.math_lesson.math_tokens import default_colors

PaletteId = Literal["primary-1to2", "primary-3to5", "middle-6to8", "technical-9to10"]


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
    color: str = ""


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
    """Legacy optional fields; prefer paletteId. Kept so old lessons still render."""

    primary: str = Field(default_factory=lambda: default_colors()["primary"])
    secondary: str = Field(default_factory=lambda: default_colors()["secondary"])
    accent: str = Field(default_factory=lambda: default_colors()["accent"])
    background: str = Field(default_factory=lambda: default_colors()["background"])
    text: str = Field(default_factory=lambda: default_colors()["text"])


class StudentInteractionSpec(BaseModel):
    id: str
    type: str
    description: str
    expectedObservation: str = ""


class AlgebraStep(BaseModel):
    """One symbolic-manipulation step (solve / factorise / simplify)."""

    id: str
    expressionBefore: str
    expressionAfter: str
    operation: str = ""
    highlightTerms: list[str] = Field(default_factory=list)


class AlgebraStepSpec(BaseModel):
    """Legacy stub shape — normalized into AlgebraStep by VisualizationSpec."""

    expression: str = ""
    explanation: str = ""


class CameraSpec(BaseModel):
    position: list[float] = Field(default_factory=lambda: [3.5, 2.8, 4.5])
    target: list[float] = Field(default_factory=lambda: [0.0, 0.5, 0.0])
    fov: float = 45.0


class SceneObject(BaseModel):
    id: str
    type: Literal["box", "cylinder", "cone", "sphere", "composite"] = "box"
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    scaleDrivenBy: str | None = None
    color: str | None = None  # palette token key, e.g. "primary" — not raw hex
    roughness: float = 0.35
    metalness: float = 0.15
    wireframeAccent: bool = True
    children: list[SceneObject] = Field(default_factory=list)  # for composite


class SceneSpec(BaseModel):
    camera: CameraSpec = Field(default_factory=CameraSpec)
    objects: list[SceneObject] = Field(default_factory=list)
    groundGrid: bool = True
    labels: list[dict[str, Any]] = Field(default_factory=list)


class FinanceSpec(BaseModel):
    principal: float = 10000
    rate: float = 8
    timeYears: float = 5
    mode: Literal[
        "compound-interest",
        "simple-interest-compare",
        "discount",
        "tax",
    ] = "compound-interest"
    compoundingFrequency: Literal["annually", "half-yearly", "quarterly"] = "annually"


class VisualizationSpec(BaseModel):
    visualizationType: str
    title: str
    description: str = ""
    renderMode: Literal["2d", "3d"] = "2d"
    scene: SceneSpec | None = None
    paletteId: PaletteId | None = None
    curveType: Literal["linear", "quadratic"] | None = None
    coefficients: list[float] = Field(default_factory=list)
    algebraSteps: list[AlgebraStep] = Field(default_factory=list)
    financeSpec: FinanceSpec | None = None
    # Legacy aliases (normalized away in model_validator)
    graphMode: Literal["", "linear", "quadratic"] = ""
    steps: list[AlgebraStepSpec] = Field(default_factory=list)
    interactiveObjects: list[InteractiveObjectSpec] = Field(default_factory=list)
    draggableObjects: list[DraggableObjectSpec] = Field(default_factory=list)
    sliders: list[SliderSpec] = Field(default_factory=list)
    buttons: list[ButtonSpec] = Field(default_factory=list)
    animations: list[AnimationSpec] = Field(default_factory=list)
    liveCalculations: list[LiveCalculationSpec] = Field(default_factory=list)
    labels: list[LabelSpec] = Field(default_factory=list)
    colors: ColorSpec = Field(default_factory=ColorSpec)
    studentInteractions: list[StudentInteractionSpec] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        # graphMode → curveType
        gm = (out.get("graphMode") or "").strip().lower()
        if not out.get("curveType") and gm in ("linear", "quadratic"):
            out["curveType"] = gm
        # legacy steps → algebraSteps
        if not out.get("algebraSteps") and out.get("steps"):
            converted: list[dict[str, Any]] = []
            raw_steps = out["steps"]
            if isinstance(raw_steps, list):
                for i, s in enumerate(raw_steps):
                    if not isinstance(s, dict):
                        continue
                    expr = str(s.get("expression") or "")
                    expl = str(s.get("explanation") or "")
                    converted.append(
                        {
                            "id": str(s.get("id") or f"s{i}"),
                            "expressionBefore": expr,
                            "expressionAfter": expr,
                            "operation": expl,
                            "highlightTerms": s.get("highlightTerms") or [],
                        }
                    )
            if converted:
                out["algebraSteps"] = converted
        return out

    @model_validator(mode="after")
    def _require_scene_for_3d(self) -> VisualizationSpec:
        if self.renderMode == "3d":
            if self.scene is None or not self.scene.objects:
                raise ValueError(
                    "renderMode='3d' requires a non-null scene with at least one object"
                )
        return self


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
