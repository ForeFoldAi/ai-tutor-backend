"""
BGE embeddings for figure_context (BAAI/bge-base-en-v1.5).

Stored on TextbookImage.context_embedding_bge at extraction time.
Query-time cosine uses stored vectors when present; otherwise embeds on the fly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.catalog.models import TextbookImage, TextbookUpload
from app.services.textbook_image_extraction import build_figure_context

logger = logging.getLogger(__name__)

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def embedding_to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def embedding_from_bytes(raw: bytes | None) -> np.ndarray | None:
    if not raw:
        return None
    try:
        arr = np.frombuffer(raw, dtype=np.float32)
        if arr.size < 8:
            return None
        return arr
    except Exception:
        return None


def get_stored_context_embedding(im: TextbookImage) -> np.ndarray | None:
    return embedding_from_bytes(getattr(im, "context_embedding_bge", None))


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < 1e-9:
        return vec
    return vec / n


def cosine_100(a: np.ndarray, b: np.ndarray) -> float:
    an = _normalize(a)
    bn = _normalize(b)
    return max(0.0, float(np.dot(an, bn))) * 100.0


def embed_texts(texts: list[str]) -> list[np.ndarray | None]:
    if not texts:
        return []
    try:
        from app.services.vector_service import _get_embedding_model, is_embedding_model_loaded

        if not is_embedding_model_loaded():
            return [None] * len(texts)
        model = _get_embedding_model()
        if model is None:
            return [None] * len(texts)
        trimmed = [(t or "")[:2000] or " " for t in texts]
        raw = model.embed_documents(trimmed)
        return [np.array(v, dtype=np.float32) for v in raw]
    except Exception as exc:
        logger.warning("figure_context BGE embed batch failed: %s", exc)
        return [None] * len(texts)


def embed_query(text: str) -> np.ndarray | None:
    vecs = embed_texts([text])
    return vecs[0] if vecs else None


def _figure_context_text(im: TextbookImage) -> str:
    ctx = getattr(im, "figure_context", None)
    if ctx and str(ctx).strip():
        return str(ctx).strip()[:4000]
    return build_figure_context(
        caption=im.caption,
        nearby_before=getattr(im, "nearby_text_before_figure", None) or "",
        nearby_after=getattr(im, "nearby_text_after_figure", None) or "",
        section_title=getattr(im, "section_title", None),
        subsection_title=getattr(im, "subsection_title", None),
        chapter_title=getattr(im, "chapter_title", None),
        page_snippet=im.page_text_snippet,
    )


def figure_context_bge_score(
    query_text: str,
    im: TextbookImage,
    *,
    query_vec: np.ndarray | None = None,
) -> float:
    """Cosine similarity 0–100 between query embedding and figure_context embedding."""
    ctx = _figure_context_text(im)
    if not ctx.strip():
        return 0.0

    stored = get_stored_context_embedding(im)
    if stored is not None:
        qv = query_vec if query_vec is not None else embed_query(query_text)
        if qv is None:
            return 0.0
        return cosine_100(qv, stored)

    vecs = embed_texts([query_text, ctx])
    if vecs[0] is None or vecs[1] is None:
        return 0.0
    return cosine_100(vecs[0], vecs[1])


def index_figure_context_embeddings(db: Session, upload: TextbookUpload) -> int:
    """Compute and persist BGE embeddings for all images of an upload."""
    images = list(
        db.scalars(
            select(TextbookImage).where(TextbookImage.textbook_upload_id == upload.id)
        ).all()
    )
    if not images:
        return 0

    contexts: list[str] = []
    for im in images:
        ctx = getattr(im, "figure_context", None)
        if not ctx or not str(ctx).strip():
            ctx = build_figure_context(
                caption=im.caption,
                nearby_before=getattr(im, "nearby_text_before_figure", None) or "",
                nearby_after=getattr(im, "nearby_text_after_figure", None) or "",
                section_title=getattr(im, "section_title", None),
                subsection_title=getattr(im, "subsection_title", None),
                chapter_title=getattr(im, "chapter_title", None),
                page_snippet=im.page_text_snippet,
            )
            im.figure_context = ctx
        contexts.append((ctx or im.file_name or " ")[:2000])

    vectors = embed_texts(contexts)
    indexed = 0
    for im, vec in zip(images, vectors):
        if vec is None:
            im.context_bge_indexed = False
            continue
        im.context_embedding_bge = embedding_to_bytes(vec)
        im.context_bge_indexed = True
        if not im.embedding_model:
            im.embedding_model = BGE_MODEL_NAME
        indexed += 1
    db.commit()
    logger.info("Indexed %d/%d figure_context BGE embeddings for upload %s", indexed, len(images), upload.id)
    return indexed
