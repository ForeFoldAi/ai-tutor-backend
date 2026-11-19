"""Backend adapters — thin wrappers around real app.services callables."""

from __future__ import annotations

from evaluation.core.adapters.pdf import extract_pdf, page_count
from evaluation.core.adapters.document import chunk_document
from evaluation.core.adapters.retrieval import embed_and_store, retrieve
from evaluation.core.adapters.tutor import ask_tutor
from evaluation.core.adapters.lesson import generate_artifact_offline
from evaluation.core.adapters.lia import sample_tutor_guidance

__all__ = [
    "ask_tutor",
    "chunk_document",
    "embed_and_store",
    "extract_pdf",
    "generate_artifact_offline",
    "page_count",
    "retrieve",
    "sample_tutor_guidance",
]
