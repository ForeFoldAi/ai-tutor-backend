import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS


def _token_length_cl100k(text: str) -> int:
    """Token count for chunk sizing (cl100k is a common proxy; close enough for Mistral RAG)."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(text or "")))
    except Exception:
        # ~4 chars per English token on average
        return max(1, len(text or "") // 4)


def _section_hint_from_text(content: str) -> str | None:
    """Best-effort heading: chapter lines, numbered sections, or short all-caps titles."""
    for raw in (content or "").strip().split("\n"):
        line = raw.strip()
        if not line:
            continue
        if len(line) > 140:
            return None
        if re.search(r"chapter\s*\d+", line, re.I):
            return line[:160]
        if re.match(r"^\d+(\.\d+)*\s+\S", line):
            return line[:160]
        if len(line) <= 80 and line.replace(" ", "").isalpha() and line.isupper():
            return line[:160]
        return None


def _enrich_chunk_metadata(docs: list[Document], source_basename: str) -> None:
    for d in docs:
        meta = dict(d.metadata or {})
        meta.setdefault("source", source_basename)
        hint = _section_hint_from_text(d.page_content or "")
        if hint:
            meta["section_hint"] = hint
        d.metadata = meta


def process_pdf(file_path: str) -> list[Document]:
    """
    Load PDF, split with paragraph-first recursive splitting (~512 tokens, ~10% overlap),
    then tag chunks with source + optional section hint for retrieval prompts.
    """
    source_basename = os.path.basename(file_path) or "document.pdf"
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for d in documents:
        meta = dict(d.metadata or {})
        meta.setdefault("source", meta.get("source", source_basename))
        d.metadata = meta

    # Prefer paragraph / chapter line breaks before cutting inside sentences.
    separators = ["\n\n", "\nCHAPTER ", "\nChapter ", "\n", ". ", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_token_length_cl100k,
        separators=separators,
        is_separator_regex=False,
    )

    docs = splitter.split_documents(documents)
    _enrich_chunk_metadata(docs, source_basename)
    return docs
