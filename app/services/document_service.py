"""Unified document processing: extract text from PDF / Word, then chunk."""

from __future__ import annotations

import os
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS


def _token_length(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(text or "")))
    except Exception:
        return max(1, len(text or "") // 4)


def _section_hint(content: str) -> str | None:
    for raw in (content or "").strip().split("\n"):
        line = raw.strip()
        if not line:
            continue
        if len(line) > 140:
            return None
        if re.search(r"chapter\s*\d+", line, re.I):
            return line[:160]
        m = re.match(r"^(\d+(?:\.\d+)*)\s+(\S.+)$", line)
        if m:
            num = m.group(1)
            if "." not in num:
                try:
                    if int(num) >= 20:
                        continue
                except ValueError:
                    pass
            return line[:160]
        if len(line) <= 80 and line.replace(" ", "").isalpha() and line.isupper():
            return line[:160]
        return None


def _enrich_metadata(docs: list[Document], source: str, extra: dict | None = None) -> None:
    for d in docs:
        meta = dict(d.metadata or {})
        meta["source"] = source
        hint = _section_hint(d.page_content or "")
        if hint:
            meta["section_hint"] = hint
        if extra:
            meta.update(extra)
        d.metadata = meta


def _load_pdf(file_path: str) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    return loader.load()


def _load_docx(file_path: str) -> list[Document]:
    """
    Load a DOCX file, preserving heading and paragraph boundaries.

    Each heading starts a new logical "page" document so the splitter can
    respect chapter/section breaks.  Paragraphs are grouped by heading so
    the resulting chunks stay semantically coherent and metadata (section)
    is derived from the nearest heading above.
    """
    from docx import Document as DocxDocument

    doc = DocxDocument(file_path)
    source = os.path.basename(file_path)

    pages: list[Document] = []
    current_heading = ""
    current_lines: list[str] = []

    def _flush(heading: str, lines: list[str]) -> None:
        text = "\n\n".join(lines).strip()
        if text:
            meta: dict = {"source": source, "page": len(pages)}
            if heading:
                meta["section_hint"] = heading[:160]
            pages.append(Document(page_content=text, metadata=meta))

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name or "").lower()
        if "heading" in style_name or re.match(r"^(chapter|unit|lesson|section)\s", text, re.I):
            # Flush the previous section and start a new one
            _flush(current_heading, current_lines)
            current_heading = text
            current_lines = []
        else:
            current_lines.append(text)

    _flush(current_heading, current_lines)

    # If the DOCX had no headings at all, fall back to one big document
    if not pages:
        full_text = "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        pages = [Document(page_content=full_text, metadata={"source": source})]

    return pages


def _get_splitter() -> RecursiveCharacterTextSplitter:
    separators = ["\n\n", "\nCHAPTER ", "\nChapter ", "\n", ". ", " ", ""]
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_token_length,
        separators=separators,
        is_separator_regex=False,
    )


def process_document(
    file_path: str,
    *,
    extra_metadata: dict | None = None,
) -> list[Document]:
    """Load a PDF or Word document and return chunked Documents with metadata."""
    source = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        raw_docs = _load_pdf(file_path)
    elif ext in (".docx", ".doc"):
        raw_docs = _load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    splitter = _get_splitter()
    chunks = splitter.split_documents(raw_docs)
    _enrich_metadata(chunks, source, extra_metadata)
    from app.services.section_heading import enrich_chunks_with_section_metadata

    enrich_chunks_with_section_metadata(chunks)
    return chunks
