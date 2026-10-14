"""Document chunking adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def chunk_document(
    pdf_path: Path,
    *,
    board: str,
    class_level: str,
    subject_name: str,
    textbook_upload_id: str,
) -> list[Any]:
    from app.services.document_service import process_document

    return process_document(
        str(pdf_path.resolve()),
        extra_metadata={
            "board": board,
            "class_level": class_level,
            "subject_name": subject_name,
            "textbook_upload_id": textbook_upload_id,
            "eval": True,
        },
    )


def chunks_to_serializable(chunks: list[Any]) -> list[dict[str, Any]]:
    out = []
    for i, c in enumerate(chunks):
        meta = dict(getattr(c, "metadata", None) or {})
        text = getattr(c, "page_content", "") or ""
        out.append(
            {
                "index": i,
                "chars": len(text),
                "preview": text[:300],
                "page": meta.get("page"),
                "section_hint": meta.get("section_hint"),
                "metadata": {k: meta[k] for k in meta if k not in ("pil_image",)},
            }
        )
    return out
