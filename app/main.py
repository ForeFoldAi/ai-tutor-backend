import json
import logging
import os
import re
from typing import Annotated

logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.pdf_service import process_pdf
from app.services.vector_service import create_vector_store, load_vector_store
from app.config import RETRIEVAL_K
from app.services.chat_service import chapter_aware_qa, chapter_aware_qa_stream, get_qa_chain
from app.services.query_match import document_page, keyword_match_score
from app.voice_api import router as voice_router
from app.voice_ws import ws_router
from app.modules.auth.router import router as auth_router
from app.modules.auth.dependencies import get_current_user
from app.modules.auth.bootstrap import seed_test_users_if_missing
from app.modules.catalog.router import router as catalog_router, student_router as student_catalog_router
from app.modules.users.models import User

app = FastAPI()

vectorstore = None

# Restrict origins in production: replace "*" with your frontend domain(s).
# Example: allow_origins=["https://app.yourdomain.com"]
_ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(voice_router)
app.include_router(ws_router)      # WebSocket: /ws/voice
app.include_router(auth_router)
app.include_router(catalog_router)
app.include_router(student_catalog_router)


@app.on_event("startup")
def _startup() -> None:
    seed_test_users_if_missing()
    import threading

    def _warm_tts():
        try:
            import asyncio
            from app.services.edge_tts_service import resolve_voice

            asyncio.run(resolve_voice())
            logger.info("Edge TTS voice pre-warmed and ready")
        except Exception as exc:
            logger.warning("TTS pre-warm failed (non-fatal): %s", exc)

    def _warm_multimodal():
        from app.config import USE_MULTIMODAL_IMAGE_RETRIEVAL, WARM_MULTIMODAL_ON_STARTUP

        if not USE_MULTIMODAL_IMAGE_RETRIEVAL or not WARM_MULTIMODAL_ON_STARTUP:
            return
        try:
            from app.services.image_service.multimodal_encoder import clip_model_available

            if clip_model_available():
                logger.info("CLIP multimodal model pre-warmed and ready")
            else:
                logger.warning(
                    "CLIP multimodal not available — image search uses keyword fallback. "
                    "Set HF_TOKEN in .env and restart."
                )
        except Exception as exc:
            logger.warning("CLIP pre-warm failed (non-fatal): %s", exc)

    threading.Thread(target=_warm_tts, daemon=True).start()
    threading.Thread(target=_warm_multimodal, daemon=True).start()


@app.get("/health/ai-models")
def ai_models_health():
    """
    Which AI models are configured for this deployment (text chat, RAG, images, voice).
    """
    from app.config import (
        HF_TOKEN,
        MISTRAL_API_KEY,
        MISTRAL_MODEL,
        MULTIMODAL_IMAGE_MODEL,
        USE_MULTIMODAL_IMAGE_RETRIEVAL,
    )
    from app.services.image_service.multimodal_encoder import clip_model_available, current_model_name
    from app.services.vector_service import is_embedding_model_loaded

    clip_ok = False
    if USE_MULTIMODAL_IMAGE_RETRIEVAL:
        try:
            clip_ok = clip_model_available()
        except Exception:
            clip_ok = False

    return {
        "models": {
            "text_chat": {
                "provider": "mistral_api",
                "model": MISTRAL_MODEL,
                "configured": bool(MISTRAL_API_KEY),
            },
            "text_rag_embeddings": {
                "model": "BAAI/bge-base-en-v1.5",
                "loaded": is_embedding_model_loaded(),
            },
            "image_retrieval": {
                "mode": "clip_chroma" if (USE_MULTIMODAL_IMAGE_RETRIEVAL and clip_ok) else "keyword_heuristic",
                "enabled": USE_MULTIMODAL_IMAGE_RETRIEVAL,
                "clip_model": current_model_name() if clip_ok else MULTIMODAL_IMAGE_MODEL,
                "clip_loaded": clip_ok,
                "hf_token_set": bool(HF_TOKEN),
            },
            "voice_tts": {
                "provider": "edge-tts",
                "voice": "en-IN-NeerjaNeural",
                "fallback_voice": "en-IN-PrabhatNeural",
                "format": "audio/mpeg",
            },
            "voice_stt": {
                "provider": "browser_webspeech",
            },
        }
    }


class ChatRequest(BaseModel):
    query: str


class ConversationTurn(BaseModel):
    role: str
    content: str


class ChapterChatRequest(BaseModel):
    query: str
    board: str
    class_level: str
    subject_name: str
    chapter_ids: list[str] | None = None
    chapter: str | None = None
    chapter_names: list[str] | None = None
    conversation_history: list[ConversationTurn] | None = None


def _fallback_answer_from_docs(query: str):
    """
    If local Llama GGUF model is missing, answer from retrieved document chunks
    directly so PDF Q&A can still work.
    """
    global vectorstore
    if vectorstore is None:
        return "The answer is not found in the document."

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    docs = retriever.invoke(query)
    if not docs:
        return "The answer is not found in the document."

    best_text = ""
    best_score = -1
    best_page = 10**9
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        score = keyword_match_score(query, text)
        pg = document_page(d)
        if score > best_score or (score == best_score and pg < best_page):
            best_score = score
            best_page = pg
            best_text = text

    if not best_text:
        return "The answer is not found in the document."

    parts = re.split(r"(?<=[.?!])\s+", best_text)
    snippet = " ".join(parts[:2]).strip()
    if not snippet:
        snippet = best_text[:500].strip()
    return snippet


async def tutor_qa_answer_for_query(query: str) -> tuple[str, str | None]:
    """
    Same logic as ``POST /chat`` for producing the answer string.

    Returns ``(answer, retrieval_warning)``. ``retrieval_warning`` is set when
    Mistral is unavailable (missing API key) and the retrieval-only fallback was used.
    """
    global vectorstore
    if vectorstore is None:
        vectorstore = load_vector_store()

    try:
        qa_chain = get_qa_chain(vectorstore)
        text = (qa_chain.run(query) or "").strip()
        return text, None
    except FileNotFoundError as e:
        text = (_fallback_answer_from_docs(query) or "").strip()
        return text, str(e)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        docs = process_pdf(file_path)
        vectorstore = create_vector_store(docs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e}")
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

    return {
        "message": "PDF processed successfully",
        "chunks": len(docs)
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    response, warning = await tutor_qa_answer_for_query(req.query)
    if warning is not None:
        return {
            "answer": response,
            "mode": "retrieval_fallback",
            "warning": warning,
        }
    return {"answer": response}


@app.post("/auth/chat")
async def chapter_chat(
    req: ChapterChatRequest,
    _current_user: Annotated[User, Depends(get_current_user)],
):
    """Chapter-aware chat: retrieve from the subject's ChromaDB collection,
    optionally filtered to the selected chapter IDs."""
    collection = f"{req.board}_{req.class_level}_{req.subject_name}".replace(" ", "_")
    logger.debug(
        "[CHAT] collection=%r chapter_ids=%s query=%r",
        collection, req.chapter_ids, req.query[:80],
    )
    history = (
        [{"role": t.role, "content": t.content} for t in req.conversation_history]
        if req.conversation_history
        else None
    )
    answer, related_images = await chapter_aware_qa(
        req.query,
        collection_name=collection,
        chapter_ids=req.chapter_ids,
        class_level=req.class_level,
        board=req.board,
        subject_name=req.subject_name,
        chapter=req.chapter or "",
        chapter_names=req.chapter_names,
        conversation_history=history,
        student_name=_current_user.full_name,
    )
    return {"answer": answer, "related_images": related_images}


@app.post("/auth/chat/stream")
async def chapter_chat_stream(
    req: ChapterChatRequest,
    _current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Streaming chapter chat (NDJSON).

    Each line is a JSON object:
      {"type":"token","content":"..."}
      {"type":"related_images","images":[...]}  (may appear mid-answer)
      {"type":"done"}
    """
    collection = f"{req.board}_{req.class_level}_{req.subject_name}".replace(" ", "_")
    logger.debug(
        "[STREAM] collection=%r chapter_ids=%s query=%r",
        collection, req.chapter_ids, req.query[:80],
    )

    async def ndjson_generator():
        pending: list[bytes] = []

        async def emit_imgs(imgs: list[dict]) -> None:
            pending.append(
                (json.dumps({"type": "related_images", "images": imgs}, ensure_ascii=False) + "\n").encode()
            )

        history = (
            [{"role": t.role, "content": t.content} for t in req.conversation_history]
            if req.conversation_history
            else None
        )
        async for token in chapter_aware_qa_stream(
            req.query,
            collection_name=collection,
            chapter_ids=req.chapter_ids,
            class_level=req.class_level,
            board=req.board,
            subject_name=req.subject_name,
            chapter=req.chapter or "",
            chapter_names=req.chapter_names,
            emit_related_images=emit_imgs,
            conversation_history=history,
            student_name=_current_user.full_name,
        ):
            while pending:
                yield pending.pop(0)
            yield (json.dumps({"type": "token", "content": token}, ensure_ascii=False) + "\n").encode()

        while pending:
            yield pending.pop(0)
        yield (json.dumps({"type": "done"}, ensure_ascii=False) + "\n").encode()

    return StreamingResponse(
        ndjson_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


# ---------------------------------------------------------------------------
# Debug: pedagogy figure-rank inspector
# ---------------------------------------------------------------------------

class FigureRankDebugRequest(BaseModel):
    query: str
    chapter_ids: list[str]
    chapter_names: list[str] | None = None
    chapter: str | None = None
    # Optional: supply board/class_level/subject_name to enable RAG doc retrieval
    board: str | None = None
    class_level: str | None = None
    subject_name: str | None = None
    top_n: int = 12


@app.post("/auth/debug/figure-rank", dependencies=[Depends(get_current_user)])
async def debug_figure_rank(req: FigureRankDebugRequest):
    """
    Developer endpoint: returns per-figure pedagogy score breakdowns.

    Useful for inspecting why a specific query returned unexpected images.
    Accepts optional board/class_level/subject_name to enable RAG retrieval
    (adds section-overlap scores); otherwise figures are ranked on caption
    BGE + concept overlap + page proximity only.

    Response schema:
      {
        "query": "...",
        "intent": { "topic_phrases": [...], "preferred_types": [...], ... },
        "figures": [
          { "file_name": ..., "caption": ..., "image_type": ..., "page": ...,
            "scores": { "caption": 42.1, "section": 30.0, ... },
            "final": 55.3, "url": ... },
          ...
        ]
      }
    """
    from app.services.image_service.textbook_image_retrieval import debug_rank_figures
    from app.services.image_service.image_intent_extractor import extract_image_intent
    from app.services.vector_service import retrieve_from_collection

    docs: list = []
    if req.board and req.class_level and req.subject_name:
        collection = f"{req.board}_{req.class_level}_{req.subject_name}".replace(" ", "_")
        try:
            docs = retrieve_from_collection(
                req.query,
                collection_name=collection,
                chapter_ids=req.chapter_ids,
                k=5,
            ) or []
        except Exception as exc:
            logger.warning("debug_figure_rank: RAG retrieval failed: %s", exc)

    intent = extract_image_intent(req.query, docs)
    figures = debug_rank_figures(
        req.chapter_ids,
        req.chapter_names or [],
        req.chapter or "",
        req.query,
        docs,
        top_n=req.top_n,
    )

    return {
        "query": req.query,
        "rag_docs_used": len(docs),
        "intent": {
            "core_concept": intent.core_concept,
            "required_terms": intent.required_terms,
            "concept_tokens": sorted(intent.concept_tokens),
            "preferred_types": intent.preferred_types,
            "excluded_types": intent.excluded_types,
            "rag_section_tokens": sorted(intent.rag_section_tokens),
            "query_type": intent.query_type,
        },
        "figures": figures,
    }
