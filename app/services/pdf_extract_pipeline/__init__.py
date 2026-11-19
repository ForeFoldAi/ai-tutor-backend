"""
Native multi-model PDF extraction pipeline for ai-tutor-backend.

Stages (full-page raster analysis):
  1. Layout detection  — DocLayout-YOLO
  2. Formula detection — YOLOv8 MFD (optional)
  3. Formula recognition — UniMERNet → LaTeX (optional)
  4. OCR — PaddleOCR with formula masking (optional)
  5. Table VLM — StructEqTable → LaTeX/Markdown/HTML (optional)

Enable via PDF_EXTRACTION_PIPELINE_ENABLED=true in .env.
"""

from app.services.pdf_extract_pipeline.cache import (
    clear_pipeline_cache,
    get_cached_pipeline_result,
    set_cached_pipeline_result,
)
from app.services.pdf_extract_pipeline.bridge import MlPipelineExtractionError
from app.services.pdf_extract_pipeline.config import pipeline_enabled
from app.services.pdf_extract_pipeline.pipeline import PdfExtractionPipeline
from app.services.pdf_extract_pipeline.types import PipelineResult

__all__ = [
    "MlPipelineExtractionError",
    "PdfExtractionPipeline",
    "PipelineResult",
    "clear_pipeline_cache",
    "get_cached_pipeline_result",
    "pipeline_enabled",
    "set_cached_pipeline_result",
]
