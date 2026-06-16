"""Load textbook text from the native PDF text layer (PyMuPDF fallback)."""

from __future__ import annotations

import logging
import os
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_MIN_ALNUM_CHARS = 80


def is_sparse_extracted_text(text: str) -> bool:
    """True when extracted page text is too thin for heading-aware RAG."""
    body = (text or "").strip()
    if not body:
        return True
    alnum = sum(1 for ch in body if ch.isalnum())
    return alnum < _MIN_ALNUM_CHARS


def load_pdf_page_documents(pdf_path: str, *, source_basename: str | None = None) -> list[Document]:
    """One LangChain Document per PDF page using the native text layer."""
    if not pdf_path or not os.path.isfile(pdf_path):
        return []
    try:
        import fitz
    except ImportError:
        logger.debug("PyMuPDF unavailable for pdf text layer")
        return []

    source = source_basename or os.path.basename(pdf_path) or "document.pdf"
    docs: list[Document] = []
    try:
        pdf = fitz.open(pdf_path)
        try:
            for page_index in range(len(pdf)):
                text = (pdf[page_index].get_text("text") or "").strip()
                if not text:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "page": page_index,
                            "extraction": "pdf_text_layer",
                        },
                    )
                )
        finally:
            pdf.close()
    except Exception as exc:
        logger.warning("pdf text layer read failed for %s: %s", pdf_path, exc)
        return []

    return docs


def merge_sparse_pages_with_pdf_text(
    docs: list[Document],
    pdf_path: str,
    *,
    source_basename: str | None = None,
) -> list[Document]:
    """
    Replace ML/OCR page bodies with PDF text layer content when page markdown is sparse.
    """
    layer_pages = {
        int(d.metadata.get("page", 0) or 0): d
        for d in load_pdf_page_documents(pdf_path, source_basename=source_basename)
    }
    if not layer_pages:
        return docs

    merged: list[Document] = []
    used_pages: set[int] = set()
    for doc in docs:
        meta = dict(doc.metadata or {})
        page = int(meta.get("page", 0) or 0)
        used_pages.add(page)
        if is_sparse_extracted_text(doc.page_content or ""):
            fallback = layer_pages.get(page)
            if fallback:
                meta["extraction"] = "pdf_text_layer_fallback"
                merged.append(
                    Document(page_content=fallback.page_content, metadata=meta)
                )
                continue
        merged.append(doc)

    for page, layer_doc in sorted(layer_pages.items()):
        if page in used_pages:
            continue
        meta = dict(layer_doc.metadata or {})
        meta["extraction"] = "pdf_text_layer_extra"
        merged.append(Document(page_content=layer_doc.page_content, metadata=meta))

    merged.sort(key=lambda d: int((d.metadata or {}).get("page", 0) or 0))
    return merged


def load_pdf_text_chunks_for_uploads(upload_ids: list[str]) -> list[Document]:
    """Load PDF text-layer documents for textbook upload IDs (query-time catalog fallback)."""
    if not upload_ids:
        return []

    from app.core.database import SessionLocal
    from app.modules.catalog.models import TextbookUpload

    docs: list[Document] = []
    try:
        import uuid

        uuids = []
        for raw in upload_ids:
            try:
                uuids.append(uuid.UUID(str(raw)))
            except Exception:
                continue
        if not uuids:
            return []

        with SessionLocal() as db:
            rows = db.query(TextbookUpload).filter(TextbookUpload.id.in_(uuids)).all()
            for row in rows:
                path = (row.file_path or "").strip()
                if not path or not os.path.isfile(path):
                    continue
                for doc in load_pdf_page_documents(path):
                    meta = dict(doc.metadata or {})
                    meta["textbook_upload_id"] = str(row.id)
                    docs.append(Document(page_content=doc.page_content, metadata=meta))
    except Exception as exc:
        logger.debug("load_pdf_text_chunks_for_uploads failed: %s", exc)
        return []

    docs.sort(key=lambda d: int((d.metadata or {}).get("page", 0) or 0))
    return docs
