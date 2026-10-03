"""Validated OCR output. Never trust raw engine output."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

_WS = re.compile(r"\s+")


def _clamp01(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


class OCRWord(BaseModel):
    text: str
    confidence: float = 0.0
    left: int = Field(default=0, ge=0)
    top: int = Field(default=0, ge=0)
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)

    @field_validator("text", mode="before")
    @classmethod
    def _norm_text(cls, v: Any) -> str:
        return _WS.sub(" ", str(v or "")).strip()

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        return _clamp01(v)


class OCRResult(BaseModel):
    text: str = ""
    words: list[OCRWord] = Field(default_factory=list)
    average_confidence: float = 0.0
    text_coverage: float = 0.0
    # Share of words Tesseract itself doubts (<0.6): garbled/partial reads hide behind a decent average.
    low_confidence_ratio: float = 0.0
    word_count: int = Field(default=0, ge=0)
    has_text: bool = False
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    engine: str = "tesseract"
    language: str = ""
    error: str | None = None

    @field_validator("average_confidence", "text_coverage", "low_confidence_ratio", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        return _clamp01(v)
