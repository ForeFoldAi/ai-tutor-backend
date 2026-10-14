"""Convert pipeline page results to LangChain Documents for RAG."""

from __future__ import annotations

from langchain_core.documents import Document

from app.services.pdf_extract_pipeline.types import PipelineResult


def pipeline_to_documents(
    pipeline: PipelineResult,
    *,
    source_basename: str,
    pdf_path: str | None = None,
) -> list[Document]:
    docs: list[Document] = []
    for page in pipeline.pages:
        text = (page.markdown or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source_basename,
                    "page": page.page_no,
                    "extraction": "ml_pipeline",
                },
            )
        )
    if not docs and pipeline.full_text.strip():
        docs.append(
            Document(
                page_content=pipeline.full_text.strip(),
                metadata={"source": source_basename, "page": 0, "extraction": "ml_pipeline"},
            )
        )
    if pdf_path:
        from app.services.pdf_text_layer import merge_sparse_pages_with_pdf_text

        docs = merge_sparse_pages_with_pdf_text(docs, pdf_path, source_basename=source_basename)
    return docs
