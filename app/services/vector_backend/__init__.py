"""Vector backend selection (chroma default, qdrant opt-in)."""

from __future__ import annotations


def vector_backend_name() -> str:
    try:
        from app.config import VECTOR_BACKEND

        name = (VECTOR_BACKEND or "chroma").strip().lower()
    except Exception:
        name = "chroma"
    return name if name in ("chroma", "qdrant") else "chroma"


def use_qdrant() -> bool:
    return vector_backend_name() == "qdrant"
