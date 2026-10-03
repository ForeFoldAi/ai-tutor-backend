"""Decide whether local OCR is enough or the image needs the vision model. Pure rules, no LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from app.config import (
    TESSERACT_MAX_LOW_CONF_RATIO,
    TESSERACT_MIN_CONFIDENCE,
    TESSERACT_MIN_TEXT_COVERAGE,
    TESSERACT_MIN_WORDS,
)
from app.modules.image_understanding.ocr.schemas import OCRResult
from app.modules.image_understanding.schemas import ImageQualityResult


class Task(str, Enum):
    TEXT_ONLY = "text_only"
    MATH = "math"
    HANDWRITING = "handwriting"
    DIAGRAM = "diagram"
    GRAPH = "graph"
    MAP = "map"
    TABLE = "table"
    MIXED = "mixed"
    UNKNOWN = "unknown"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class RoutingDecision:
    task: Task
    use_vision: bool
    reason: str


# Student message → the question is about something OCR text cannot represent.
_MESSAGE_VISUAL: list[tuple[re.Pattern[str], Task]] = [
    (re.compile(r"\b(graph|plot|axis|axes|slope|chart|bar\s+graph|pie)\b", re.I), Task.GRAPH),
    (re.compile(r"\b(map|maps|located|location|border|state\s+capital)\b", re.I), Task.MAP),
    (re.compile(r"\b(table|tabular|column|row)\b", re.I), Task.TABLE),
    (
        re.compile(
            r"\b(diagram|figure|fig\.?|picture|photo|image|drawing|draw|shape|arrow|label(?:led|ed|s)?|"
            r"triangle|circle|angle|parts?\s+of)\b",
            re.I,
        ),
        Task.DIAGRAM,
    ),
    (
        re.compile(
            r"\b(my\s+(answer|work|solution|notes?|handwriting|homework)|check|verify|"
            r"is\s+(this|it|my\s+\w+)\s+(right|correct)|did\s+i)\b",
            re.I,
        ),
        Task.HANDWRITING,
    ),
    (
        re.compile(r"\b(solve|calculate|compute|simplify|evaluate|find\s+(x|y|the\s+value)|equation)\b", re.I),
        Task.MATH,
    ),
]

# OCR text → mathematical layout that plain text loses (exponents, fractions, alignment).
_MATH_TEXT = re.compile(
    r"[=^√∫∑π÷×≤≥]"
    r"|\d\s*[+*/x]\s*\d"
    r"|\d\s+[\-−]\s+\d"  # spaced minus only, so year/page ranges like 1857-58 stay text
    r"|\b\d*[a-z]\s*[+\-−]\s*\d"
    r"|\b\d+[a-z]\b",
)


def _message_task(message: str) -> Task | None:
    for pattern, task in _MESSAGE_VISUAL:
        if pattern.search(message or ""):
            return task
    return None


def decide(ocr: OCRResult | None, quality: ImageQualityResult, student_message: str = "") -> RoutingDecision:
    if not quality.is_usable:
        return RoutingDecision(Task.UNCLEAR, False, quality.reason or "unusable_image")
    if ocr is None:
        return RoutingDecision(Task.UNKNOWN, True, "ocr_disabled")
    if ocr.error:
        return RoutingDecision(Task.UNKNOWN, True, f"ocr_error:{ocr.error}")

    asked = _message_task(student_message)
    if asked is not None and asked is not Task.MATH:
        return RoutingDecision(asked, True, "student_asked_visual")
    if asked is Task.MATH or _MATH_TEXT.search(ocr.text):
        return RoutingDecision(Task.MATH, True, "math_layout")

    if not ocr.has_text or ocr.word_count < TESSERACT_MIN_WORDS:
        # Little readable text: a picture, diagram, or handwriting Tesseract cannot read.
        return RoutingDecision(Task.UNKNOWN, True, "little_text")
    # Low confidence has many causes (handwriting, photo angle, glare, unsupported script); vision decides which.
    if ocr.average_confidence < TESSERACT_MIN_CONFIDENCE:
        return RoutingDecision(Task.UNKNOWN, True, "low_ocr_confidence")
    if ocr.low_confidence_ratio > TESSERACT_MAX_LOW_CONF_RATIO:
        return RoutingDecision(Task.UNKNOWN, True, "unreliable_words")
    if ocr.text_coverage < TESSERACT_MIN_TEXT_COVERAGE:
        return RoutingDecision(Task.MIXED, True, "low_text_coverage")
    if quality.blurry or quality.too_dark or quality.too_bright:
        return RoutingDecision(Task.UNKNOWN, True, f"quality:{quality.reason}")
    return RoutingDecision(Task.TEXT_ONLY, False, "clean_printed_text")
