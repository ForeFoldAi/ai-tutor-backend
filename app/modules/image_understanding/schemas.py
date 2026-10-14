"""Pydantic schemas for multimodal student image understanding."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ImageType = Literal[
    "textbook_page",
    "textbook_question",
    "handwritten_notes",
    "handwritten_homework",
    "math_problem",
    "diagram",
    "graph",
    "chart",
    "map",
    "science_diagram",
    "question_paper",
    "answer_sheet",
    "mixed_educational_content",
    "unknown",
]

StudentIntent = Literal[
    "explain",
    "solve",
    "summarize",
    "answer",
    "check_answer",
    "translate",
    "simplify",
    "extract_text",
    "explain_diagram",
    "explain_graph",
    "explain_table",
    "compare",
    "generate_examples",
    "generate_questions",
    "unknown",
]

_IMAGE_TYPES = set(ImageType.__args__)  # type: ignore[attr-defined]
_INTENTS = set(StudentIntent.__args__)  # type: ignore[attr-defined]


class DetectedQuestion(BaseModel):
    text: str = ""
    intent: str = "unknown"


class VisualElement(BaseModel):
    type: str = "unknown"
    description: str = ""
    labels: list[str] = Field(default_factory=list)
    measurements: list[str] = Field(default_factory=list)


class MathematicalContent(BaseModel):
    detected: bool = False
    expressions: list[str] = Field(default_factory=list)


class ImageQualityResult(BaseModel):
    score: float = 1.0
    is_usable: bool = True
    too_small: bool = False
    too_dark: bool = False
    too_bright: bool = False
    blurry: bool = False
    mostly_blank: bool = False
    reason: str = ""


AnalysisSource = Literal["vision", "tesseract", "hybrid", "none"]
AnalysisStatus = Literal["ok", "unclear"]
# Set by our pipeline only — stripped from model JSON so the vision LLM cannot spoof them.
_PIPELINE_FIELDS = (
    "source",
    "status",
    "requires_clearer_image",
    "reason",
    "ocr_confidence",
    "image_quality",
)


class ImageUnderstandingResult(BaseModel):
    image_type: str = "unknown"
    language: str = "English"
    detected_subject: str = ""
    ocr_text: str = ""
    content_summary: str = ""
    educational_topic: str = ""
    question_detected: bool = False
    questions: list[DetectedQuestion] = Field(default_factory=list)
    visual_elements: list[VisualElement] = Field(default_factory=list)
    mathematical_content: MathematicalContent = Field(default_factory=MathematicalContent)
    handwriting_detected: bool = False
    confidence: float = 0.0
    unclear_regions: list[str] = Field(default_factory=list)
    source: AnalysisSource = "vision"
    status: AnalysisStatus = "ok"
    requires_clearer_image: bool = False
    reason: str = ""
    ocr_confidence: float | None = None
    image_quality: ImageQualityResult | None = None

    @field_validator("image_type", mode="before")
    @classmethod
    def _coerce_image_type(cls, v: Any) -> str:
        s = str(v or "unknown").strip().lower().replace(" ", "_")
        return s if s in _IMAGE_TYPES else "unknown"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("questions", mode="before")
    @classmethod
    def _coerce_questions(cls, v: Any) -> list:
        if not isinstance(v, list):
            return []
        out = []
        for item in v[:12]:
            if isinstance(item, (dict, DetectedQuestion)):
                out.append(item)
            elif isinstance(item, str) and item.strip():
                out.append({"text": item.strip(), "intent": "unknown"})
        return out


class UnderstandingBundle(BaseModel):
    result: ImageUnderstandingResult
    intent: str = "unknown"
    retrieval_query: str = ""
    tutor_prompt_block: str = ""
    math_prompt_block: str = ""
    low_confidence: bool = False
    provider_model: str = ""
    routing_task: str = ""


class ImageUploadResponse(BaseModel):
    image_id: str
    content_hash: str
    expires_at: float
    mime_type: str
    width: int
    height: int


class ImageAnalyzeRequest(BaseModel):
    image_ids: list[str] = Field(min_length=1)
    message: str = ""
    class_level: str = ""
    subject_name: str = ""
    board: str = ""


class ImageAnalyzeResponse(BaseModel):
    result: ImageUnderstandingResult
    intent: str
    retrieval_query: str
    low_confidence: bool


def sanitize_vision_payload(raw: dict[str, Any] | None) -> ImageUnderstandingResult:
    """Validate model JSON; never trust raw LLM output."""
    if not isinstance(raw, dict):
        return ImageUnderstandingResult(confidence=0.0)
    raw = {k: v for k, v in raw.items() if k not in _PIPELINE_FIELDS}
    try:
        return ImageUnderstandingResult.model_validate(raw)
    except Exception:
        # Best-effort partial extract
        partial: dict[str, Any] = {}
        for key in (
            "image_type",
            "language",
            "detected_subject",
            "ocr_text",
            "content_summary",
            "educational_topic",
            "question_detected",
            "handwriting_detected",
            "confidence",
            "unclear_regions",
        ):
            if key in raw:
                partial[key] = raw[key]
        if isinstance(raw.get("questions"), list):
            partial["questions"] = raw["questions"]
        if isinstance(raw.get("visual_elements"), list):
            partial["visual_elements"] = raw["visual_elements"]
        if isinstance(raw.get("mathematical_content"), dict):
            partial["mathematical_content"] = raw["mathematical_content"]
        try:
            return ImageUnderstandingResult.model_validate(partial)
        except Exception:
            return ImageUnderstandingResult(
                content_summary=str(raw.get("content_summary") or "")[:500],
                confidence=0.0,
            )


def coerce_intent(value: str | None) -> str:
    s = (value or "unknown").strip().lower().replace(" ", "_")
    return s if s in _INTENTS else "unknown"
