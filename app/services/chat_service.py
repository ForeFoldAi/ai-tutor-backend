"""
AI Tutor chat service.

Key improvements over the original:
- Fully async: uses httpx.AsyncClient instead of requests (no event-loop blocking)
- Streaming: stream_chapter_qa() yields tokens for StreamingResponse
- Grade-aware prompt: adapts language complexity to class level (1-10)
- Answer-type detection: one-word / definition / stepwise / paragraph / exam-format
- Redis cache: repeated questions answered instantly at zero LLM cost
- Fallback: keyword-chunk answer when Mistral key is missing
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, AsyncIterator

import httpx
from langchain_core.prompts import PromptTemplate

from app.config import (
    CONTEXT_CHAR_BUDGET,
    IMAGE_RETRIEVAL_TIMEOUT_SEC,
    MISTRAL_API_KEY,
    MISTRAL_MAX_TOKENS,
    MISTRAL_MODEL,
    MISTRAL_TEMPERATURE,
    RETRIEVAL_K,
    TOP_RELATED_IMAGES,
)
from app.services.vector_service import InMemoryDocVectorStore

logger = logging.getLogger(__name__)


def _log_image_stage(stage: str, images: list[dict]) -> None:
    """Structured per-stage image trace for debugging pipeline leaks."""
    try:
        summary = [
            f"{x.get('file_name') or x.get('image_url', '?')}:"
            f"{(x.get('caption') or '')[:60]}"
            for x in (images or [])
        ]
    except Exception:
        summary = ["<unserializable>"]
    logger.debug("[IMAGE-STAGE] stage=%s count=%d images=%s", stage, len(images or []), summary)


# ── Answer-type detection ────────────────────────────────────────────────────

_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|hii|helo|heya|good (morning|afternoon|evening|night)|"
    r"how are you|how r u|what('s| is) up|sup|yo|namaste|hiya|howdy|"
    r"nice to meet|greet|thanks|thank you|bye|goodbye|see you|ok|okay|"
    r"cool|awesome|great|good|fine|i('m| am) (good|fine|ok|bored|tired|happy|sad)|"
    r"can you help|are you (there|ready|a bot|ai|real))\b",
    re.I,
)
_ONE_WORD_PATTERNS = re.compile(
    r"\b(who (is|was|invented|discovered|founded)|"
    r"when (was|did|is)|"
    r"which (country|city|element|planet|organ)|"
    r"name (the|a|an|one)\b|"
    r"capital of|"
    r"symbol (of|for)|"
    r"formula (of|for))\b",
    re.I,
)
_EXAM_PATTERNS = re.compile(
    r"\b(\d\s*marks?|long answer|write (in )?detail|elaborate|"
    r"essay|comprehensive|full answer|describe (in )?detail)\b",
    re.I,
)
_STEPWISE_PATTERNS = re.compile(
    r"\b(steps?|procedure|how (to|does|do)|process of|method|"
    r"working of|mechanism|stages? of)\b",
    re.I,
)
_SIMPLE_PATTERNS = re.compile(
    r"\b(simple|easy|in simple words|for kids|class [1-3]\b|"
    r"explain simply|layman)\b",
    re.I,
)
_DEFINITION_STARTS = re.compile(
    r"^(define|what is|what are|meaning of|definition of)\b",
    re.I,
)
_EXPLAIN_PATTERNS = re.compile(
    r"\b(explain|describe|why|how does|what happens|elaborate on|"
    r"discuss|talk about)\b",
    re.I,
)
_BULLET_PATTERNS = re.compile(
    r"\b(bullet points?|list (the|all|some)|points? (on|about)|"
    r"give points|write points)\b",
    re.I,
)


def detect_answer_type(query: str) -> str:
    q = query.strip()
    # Greetings and casual chat get a warm-but-brief tutor reply
    if _GREETING_PATTERNS.match(q) and len(q.split()) <= 10:
        return "greeting"
    if _EXAM_PATTERNS.search(q):
        return "exam-format"
    if _BULLET_PATTERNS.search(q):
        return "bullet-points"
    if _STEPWISE_PATTERNS.search(q):
        return "stepwise"
    if _SIMPLE_PATTERNS.search(q):
        return "simplified"
    if _ONE_WORD_PATTERNS.search(q):
        return "one-word"
    if _DEFINITION_STARTS.match(q):
        return "definition"
    if _EXPLAIN_PATTERNS.search(q):
        return "paragraph"
    return "short-answer"


_ANSWER_INSTRUCTIONS: dict[str, str] = {
    "one-word": (
        "Give ONE word or a very short phrase (2-3 words max) only. "
        "No sentence. No explanation. Just the answer."
    ),
    "definition": (
        "Give exactly ONE clear, simple sentence that defines the term. "
        "Keep it natural and easy to understand."
    ),
    "stepwise": (
        "Explain step by step. Number each step clearly (Step 1, Step 2, ...). "
        "Keep each step short, simple, and easy to follow."
    ),
    "paragraph": (
        "Write a clear, friendly paragraph of 3-4 sentences. "
        "Use simple language. Add a relatable example if it helps."
    ),
    "exam-format": (
        "Write a well-structured answer suitable for an exam:\n"
        "- Start with a one-line introduction.\n"
        "- Give 4-6 numbered key points, each as a complete sentence.\n"
        "- End with a one-line conclusion.\n"
        "Keep the language clear and academic but not robotic."
    ),
    "simplified": (
        "Use the simplest everyday words possible. Very short sentences. "
        "No technical terms. Explain like you are talking to a young child. "
        "Add a fun or relatable example."
    ),
    "bullet-points": (
        "Answer using clear bullet points. "
        "Each bullet should be one short, complete idea."
    ),
    "short-answer": (
        "Answer in 2-3 short, clear sentences. Be direct and friendly."
    ),
    "greeting": (
        "The student is greeting or making small talk. Reply WARMLY and VERY BRIEFLY "
        "(1-2 sentences max) in a friendly teacher tone. "
        "Then gently guide them back to the current chapter with ONE short invite. "
        "Example: 'Hi! Great to see you. Ready to explore {subject} today? Ask me anything about {chapter}!' "
        "Never become a social chatbot. Always stay as a helpful tutor."
    ),
}

# ── Grade-aware prompt template ──────────────────────────────────────────────

_GRADE_LABELS: dict[str, str] = {
    "CLASS_1": "Class 1 (age 6-7)",
    "CLASS_2": "Class 2 (age 7-8)",
    "CLASS_3": "Class 3 (age 8-9)",
    "CLASS_4": "Class 4 (age 9-10)",
    "CLASS_5": "Class 5 (age 10-11)",
    "CLASS_6": "Class 6 (age 11-12)",
    "CLASS_7": "Class 7 (age 12-13)",
    "CLASS_8": "Class 8 (age 13-14)",
    "CLASS_9": "Class 9 (age 14-15)",
    "CLASS_10": "Class 10 (age 15-16)",
}

_GRADE_COMPLEXITY: dict[str, str] = {
    "CLASS_1": (
        "Speak like a kind, patient teacher talking to a 6-year-old. "
        "Use only the simplest words. One short idea per sentence. "
        "No jargon at all. Use fun comparisons from daily life."
    ),
    "CLASS_2": (
        "Speak like a kind teacher talking to a 7-year-old. "
        "Use only simple everyday words. One idea per sentence. "
        "No technical terms. Make it feel like a friendly story."
    ),
    "CLASS_3": (
        "Use simple, friendly language for an 8-year-old student. "
        "Short sentences. Everyday words only. "
        "Give a simple real-life example when helpful."
    ),
    "CLASS_4": (
        "Use clear, simple language for a 9-year-old. "
        "Short sentences. Introduce a basic term only if truly needed, "
        "and explain it right away in simple words."
    ),
    "CLASS_5": (
        "Use clear, friendly language for a 10-year-old. "
        "You may use basic subject terms but explain them simply. "
        "Add a helpful example to make the idea stick."
    ),
    "CLASS_6": (
        "Use clear, natural language for an 11-year-old. "
        "Subject-specific terms are fine but always explain them briefly. "
        "Keep sentences short and easy to read."
    ),
    "CLASS_7": (
        "Use clear, confident language for a 12-year-old. "
        "Use correct subject terms with a short explanation. "
        "Answers can be slightly more detailed but still easy to follow."
    ),
    "CLASS_8": (
        "Use standard, friendly academic language for a 13-year-old. "
        "Well-structured answers. Use correct terminology. "
        "Keep the tone warm and encouraging, not textbook-robotic."
    ),
    "CLASS_9": (
        "Use standard academic language for a 14-15-year-old. "
        "Use proper terminology. Answers can be detailed and well-organised. "
        "Maintain a helpful, clear, exam-ready tone."
    ),
    "CLASS_10": (
        "Use precise academic language for a 15-16-year-old board exam student. "
        "Use correct technical terms. Give thorough, well-structured answers. "
        "Tone should be clear, confident, and exam-ready."
    ),
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are a friendly AI Tutor for school students from Class 1 to Class 10.
Your goal is to help students learn easily, clearly, and confidently — like a kind, patient teacher.

STUDENT DETAILS:
- Class: {grade_label}
- Board: {board}
- Subject: {subject}
- Chapter: {chapter}

LANGUAGE RULE:
{complexity_rule}

ANSWER FORMAT RULE:
{answer_instruction}

CHAPTER CONTEXT (always use this first to answer):
{context}

STUDENT'S QUESTION:
{question}

YOUR RULES:
1. Answer using the chapter context above first. That is always your first source.
2. If the chapter context does not have enough information, give a safe, accurate general explanation and say: "This is a general explanation as it is not covered in detail in this chapter."
3. Never make up textbook-specific facts — no invented dates, names, formulas, or page numbers.
4. Never mix in content from other chapters or subjects.
5. Use a warm, encouraging, teacher-like tone. Never sound robotic or cold.
6. Start naturally when suitable — for example:
   - "Great question!"
   - "Let me explain simply."
   - "Nice doubt!"
   - "Here's an easy answer."
   - "Let's understand this step by step."
7. Add a relatable example when it genuinely helps understanding.
8. Do NOT give unsolicited study tips.
9. When suitable, end your answer with ONE helpful next-step offer, such as:
   - "Want a short answer too?"
   - "Want me to explain with an example?"
   - "Want this in even simpler words?"
   - "Want a 5-marks style answer?"
   (Only offer this when it adds real value. Skip it for one-word or very short answers.)
10. Keep your answer voice-friendly — smooth, natural sentences that sound good when read aloud.
11. GREETING / CASUAL CHAT RULE: If the student greets you or makes small talk (hi, hello, how are you, thanks, bye, etc.):
    - Reply warmly and briefly (1-2 sentences max).
    - Stay teacher-like — never become a social chatbot.
    - Gently guide them back to the subject with a short invite.
    - Example: "Hi! Happy to help. Ready to learn {subject} today? Ask me anything!"

YOUR ANSWER ({answer_type}):"""

_LEGACY_PROMPT_TEMPLATE = """\
You are a friendly AI Tutor helping school students learn clearly and confidently.

Use the provided document context to answer the student's question.
Each context block shows the page, source file, and section when available — use these to stay accurate.

If the answer is not in the document, say:
"This doesn't seem to be covered in the uploaded document. Here is a general explanation:"
Then give a brief, accurate, student-friendly answer.

If the question refers to a specific chapter (for example "Chapter 1"), only use context clearly from that chapter. Do not mix in other chapters.

Rules:
- Use simple, clear, friendly language.
- Match the answer length to the question (one word if asked, full paragraph if needed).
- Never invent facts not present in the context.
- Keep the tone warm and encouraging.

Context:
{context}

Student's Question:
{question}

Your Answer:""".strip()

PROMPT = PromptTemplate(
    template=_LEGACY_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def _build_prompt(
    query: str,
    context: str,
    *,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
) -> str:
    answer_type = detect_answer_type(query)
    grade_label = _GRADE_LABELS.get(class_level, class_level or "School student")
    complexity = _GRADE_COMPLEXITY.get(class_level, "Use clear, age-appropriate language.")
    raw_instruction = _ANSWER_INSTRUCTIONS.get(answer_type, _ANSWER_INSTRUCTIONS["short-answer"])
    # Fill subject/chapter placeholders in the greeting instruction
    instruction = raw_instruction.format(
        subject=subject_name or "your subject",
        chapter=chapter or "the current chapter",
    ) if answer_type == "greeting" else raw_instruction

    return _SYSTEM_PROMPT_TEMPLATE.format(
        grade_label=grade_label,
        board=board or "General",
        subject=subject_name or "General",
        chapter=chapter or "Current chapter",
        complexity_rule=complexity,
        answer_instruction=instruction,
        context=context,
        question=query,
        answer_type=answer_type,
    )


# ── Context assembly ─────────────────────────────────────────────────────────

def _format_chunk_for_context(doc) -> str:
    meta = getattr(doc, "metadata", None) or {}
    page_raw = meta.get("page")
    human_page = None
    if page_raw is not None:
        try:
            human_page = int(page_raw) + 1
        except (TypeError, ValueError):
            pass
    src = meta.get("source", "")
    short_src = os.path.basename(str(src)) if src else ""
    sec = (meta.get("section_hint") or "").strip()
    tag = f"[page {human_page}]" if human_page is not None else "[page ?]"
    if short_src:
        tag += f" [source: {short_src}]"
    if sec:
        tag += f" [section: {sec}]"
    body = (getattr(doc, "page_content", "") or "").strip()
    return f"{tag}\n{body}"


def _join_context_within_budget(docs: list) -> str:
    parts: list[str] = []
    used = 0
    for d in docs:
        block = _format_chunk_for_context(d)
        add = len(block) + (2 if parts else 0)
        if parts and used + add > CONTEXT_CHAR_BUDGET:
            break
        if not parts and len(block) > CONTEXT_CHAR_BUDGET:
            parts.append(block[:CONTEXT_CHAR_BUDGET])
            break
        parts.append(block)
        used += add
    return "\n\n".join(parts)


# ── Mistral API calls (async) ────────────────────────────────────────────────

def _ensure_mistral_config() -> None:
    if not MISTRAL_API_KEY:
        raise FileNotFoundError(
            "Missing MISTRAL_API_KEY env var. Set it to enable AI answers."
        )


async def _call_mistral_async(prompt: str) -> str:
    """Non-blocking Mistral chat completion via httpx.AsyncClient."""
    _ensure_mistral_config()
    logger.debug("Mistral request model=%s max_tokens=%d", MISTRAL_MODEL, MISTRAL_MAX_TOKENS)

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MISTRAL_MAX_TOKENS,
                "temperature": MISTRAL_TEMPERATURE,
            },
        )
        resp.raise_for_status()

    payload: dict[str, Any] = resp.json()
    text = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    logger.debug("Mistral response length=%d chars", len(text))
    return text or "The answer is not found in the document."


async def _stream_mistral_async(prompt: str) -> AsyncIterator[str]:
    """Yield tokens from Mistral SSE stream for use with StreamingResponse."""
    _ensure_mistral_config()
    logger.debug("Mistral streaming request model=%s", MISTRAL_MODEL)

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MISTRAL_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MISTRAL_MAX_TOKENS,
                "temperature": MISTRAL_TEMPERATURE,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue


# ── Fallback (no API key) ────────────────────────────────────────────────────

def _best_chunk_fallback(query: str, docs: list) -> str:
    from app.services.query_match import document_page, keyword_match_score

    best_text, best_score, best_page = "", -1, 10**9
    for d in docs:
        text = (d.page_content or "").strip()
        if not text:
            continue
        score = keyword_match_score(query, text)
        pg = document_page(d)
        if score > best_score or (score == best_score and pg < best_page):
            best_score, best_page, best_text = score, pg, text

    if not best_text:
        return "The answer is not found in the document."

    parts = re.split(r"(?<=[.?!])\s+", best_text)
    snippet = " ".join(parts[:3]).strip()
    return snippet or best_text[:600].strip()


# ── Public API ───────────────────────────────────────────────────────────────

async def chapter_aware_qa(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None = None,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    chapter_names: list[str] | None = None,
    conversation_history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks from ChromaDB and answer via Mistral (async).

    Returns ``(answer_text, related_images)``. Checks Redis cache first.
    """
    from app.core.cache import (
        deserialize_tutor_cache,
        get_cached_answer,
        set_cached_answer,
    )
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.image_service.textbook_image_retrieval import related_images_for_query
    from app.services.vector_service import retrieve_from_collection

    conv = resolve_conversation_context(
        query, conversation_history=conversation_history, chapter=chapter
    )
    retrieval_query = conv.retrieval_query or query
    img_allowed = should_retrieve_images(conv, chapter_ids=chapter_ids)

    cached = await get_cached_answer(collection_name, chapter_ids, query)
    if cached:
        answer, _ = deserialize_tutor_cache(cached)
        # Images are never stored in cache — re-rank fresh per query.
        imgs: list[dict] = []
        if chapter_ids and img_allowed:
            docs_for_imgs = retrieve_from_collection(
                retrieval_query,
                collection_name=collection_name,
                chapter_ids=chapter_ids,
                k=RETRIEVAL_K,
            )
            if docs_for_imgs:
                try:
                    imgs = await asyncio.wait_for(
                        asyncio.to_thread(
                            related_images_for_query,
                            collection_name,
                            chapter_ids,
                            chapter_names or [],
                            chapter,
                            query,
                            docs_for_imgs,
                            answer_text=answer,
                            top_n=TOP_RELATED_IMAGES,
                            conversation_history=conversation_history,
                        ),
                        timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
                    )
                except Exception as exc:
                    logger.warning("Image retrieval on cache hit failed: %s", exc)
        return answer, imgs

    docs = retrieve_from_collection(
        retrieval_query,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        k=RETRIEVAL_K,
    )
    if not docs:
        return "The answer is not found in the document.", []

    context = _join_context_within_budget(docs)
    prompt = _build_prompt(
        query,
        context,
        class_level=class_level,
        board=board,
        subject_name=subject_name,
        chapter=chapter,
    )

    try:
        answer = await _call_mistral_async(prompt)
    except FileNotFoundError:
        answer = _best_chunk_fallback(query, docs)
    except Exception as exc:
        logger.error("Mistral call failed: %s: %s", type(exc).__name__, exc)
        answer = _best_chunk_fallback(query, docs)

    related: list[dict] = []
    if chapter_ids and img_allowed:
        try:
            related = await asyncio.wait_for(
                asyncio.to_thread(
                    related_images_for_query,
                    collection_name,
                    chapter_ids,
                    chapter_names or [],
                    chapter,
                    query,
                    docs,
                    answer_text=answer,
                    top_n=TOP_RELATED_IMAGES,
                    conversation_history=conversation_history,
                ),
                timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Image retrieval timed out after %.0fs for query=%r",
                IMAGE_RETRIEVAL_TIMEOUT_SEC,
                query[:80],
            )
        except Exception as exc:
            logger.warning("Image retrieval failed: %s", exc)

    await set_cached_answer(
        collection_name,
        chapter_ids,
        query,
        answer,
        related_images=related,
    )
    return answer, related


async def chapter_aware_qa_stream(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None = None,
    class_level: str = "",
    board: str = "",
    subject_name: str = "",
    chapter: str = "",
    chapter_names: list[str] | None = None,
    emit_related_images: Callable[[list[dict]], Awaitable[None]] | None = None,
    conversation_history: list[dict] | None = None,
) -> AsyncIterator[str]:
    """
    Streaming version of chapter_aware_qa.

    Optionally invokes *emit_related_images* when ranked images are ready
    (usually during the first tokens, without blocking retrieval).
    """
    from app.core.cache import (
        deserialize_tutor_cache,
        get_cached_answer,
        set_cached_answer,
    )
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.image_service.textbook_image_retrieval import early_related_images_for_query, related_images_for_query
    from app.services.vector_service import retrieve_from_collection

    conv = resolve_conversation_context(
        query, conversation_history=conversation_history, chapter=chapter
    )
    retrieval_query = conv.retrieval_query or query
    img_allowed = should_retrieve_images(conv, chapter_ids=chapter_ids)

    cached = await get_cached_answer(collection_name, chapter_ids, query)
    if cached:
        answer, _ = deserialize_tutor_cache(cached)
        # Images are never stored in cache — always retrieve fresh so they match
        # this specific query rather than a previous one with a similar answer.
        imgs: list[dict] = []
        if chapter_ids and img_allowed:
            docs_for_imgs = retrieve_from_collection(
                retrieval_query,
                collection_name=collection_name,
                chapter_ids=chapter_ids,
                k=RETRIEVAL_K,
            )
            if docs_for_imgs:
                try:
                    imgs = await asyncio.wait_for(
                        asyncio.to_thread(
                            related_images_for_query,
                            collection_name,
                            chapter_ids,
                            chapter_names or [],
                            chapter,
                            query,
                            docs_for_imgs,
                            answer_text=answer,
                            top_n=TOP_RELATED_IMAGES,
                            conversation_history=conversation_history,
                        ),
                        timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
                    )
                except Exception as exc:
                    logger.warning("Image retrieval on cache hit failed: %s", exc)
        _log_image_stage("cache_hit_emit", imgs)
        if emit_related_images:
            await emit_related_images(imgs)
        yield answer
        return

    docs = retrieve_from_collection(
        retrieval_query,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        k=RETRIEVAL_K,
    )
    if not docs:
        if emit_related_images:
            await emit_related_images([])
        yield "The answer is not found in the document."
        return

    if emit_related_images and not img_allowed:
        await emit_related_images([])

    context = _join_context_within_budget(docs)
    prompt = _build_prompt(
        query,
        context,
        class_level=class_level,
        board=board,
        subject_name=subject_name,
        chapter=chapter,
    )

    last_imgs: list[dict] = []
    images_emitted = False

    async def _retrieve_images_for_answer(answer_text: str) -> list[dict]:
        if not chapter_ids or not img_allowed:
            return []
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    related_images_for_query,
                    collection_name,
                    chapter_ids,
                    chapter_names or [],
                    chapter,
                    query,
                    docs,
                    answer_text=answer_text,
                    top_n=TOP_RELATED_IMAGES,
                    conversation_history=conversation_history,
                ),
                timeout=IMAGE_RETRIEVAL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Image retrieval timed out after %.0fs (stream end)",
                IMAGE_RETRIEVAL_TIMEOUT_SEC,
            )
            return []
        except Exception as exc:
            logger.warning("Image retrieval failed (stream end): %s", exc)
            return []

    # Bootstrap image search from retrieved textbook context while the LLM streams.
    from app.services.image_service.textbook_image_retrieval import _context_excerpt_from_docs

    bootstrap_ctx = _context_excerpt_from_docs(docs)[:2000]
    img_task: asyncio.Task[list[dict]] | None = None
    early_img_task: asyncio.Task[list[dict]] | None = None
    if chapter_ids and img_allowed:
        early_img_task = asyncio.create_task(
            asyncio.to_thread(
                early_related_images_for_query,
                chapter_ids,
                docs,
                query,
                top_n=TOP_RELATED_IMAGES,
                conversation_history=conversation_history,
                chapter_single=chapter,
            )
        )
        img_task = asyncio.create_task(_retrieve_images_for_answer(bootstrap_ctx))

    async def _maybe_emit_bootstrap_images(answer_so_far: str = "") -> None:
        nonlocal images_emitted, last_imgs
        if images_emitted or not emit_related_images or not img_allowed:
            return
        from app.config import EARLY_IMAGE_MIN_CHARS

        if len(answer_so_far) < EARLY_IMAGE_MIN_CHARS:
            return
        if early_img_task is not None and early_img_task.done():
            try:
                last_imgs = early_img_task.result()
            except Exception:
                last_imgs = []
            if last_imgs:
                _log_image_stage("early_emit", last_imgs)
                await emit_related_images(last_imgs)
                images_emitted = True
                return
        if img_task is None or not img_task.done():
            return
        try:
            last_imgs = img_task.result()
        except Exception:
            last_imgs = []
        if last_imgs:
            _log_image_stage("bootstrap_emit", last_imgs)
            await emit_related_images(last_imgs)
            images_emitted = True

    try:
        full_answer: list[str] = []
        async for token in _stream_mistral_async(prompt):
            full_answer.append(token)
            await _maybe_emit_bootstrap_images("".join(full_answer))
            yield token
        answer_text = "".join(full_answer)
        final_imgs = await _retrieve_images_for_answer(answer_text)
        last_imgs = final_imgs
        _log_image_stage("final_emit", final_imgs)
        if emit_related_images:
            await emit_related_images(final_imgs)
        if answer_text:
            await set_cached_answer(
                collection_name,
                chapter_ids,
                query,
                answer_text,
                related_images=last_imgs,
            )
    except FileNotFoundError:
        fb = _best_chunk_fallback(query, docs)
        last_imgs = await _retrieve_images_for_answer(fb)
        if emit_related_images:
            await emit_related_images(last_imgs)
        yield fb
    except Exception as exc:
        logger.error("Mistral stream failed: %s: %s", type(exc).__name__, exc)
        fb = _best_chunk_fallback(query, docs)
        last_imgs = await _retrieve_images_for_answer(fb)
        if emit_related_images:
            await emit_related_images(last_imgs)
        yield fb


# ── Legacy compatibility (used by /upload + /chat in main.py) ────────────────

def get_qa_chain(vectorstore):
    """
    Backward-compatible wrapper for the old main.py ``/chat`` endpoint.
    Returns a sync-compatible object whose ``.run()`` blocks the caller
    (acceptable only for the legacy /chat route).
    """

    class _QARunnable:
        def run(self, query: str) -> str:
            import asyncio

            retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
            docs = retriever.invoke(query)

            if isinstance(vectorstore, InMemoryDocVectorStore):
                def _page(d):
                    m = getattr(d, "metadata", None) or {}
                    try:
                        return int(m.get("page", 0) or 0)
                    except Exception:
                        return 0
                docs = sorted(docs, key=_page)

            context = _join_context_within_budget(docs)
            prompt = PROMPT.format(context=context, question=query)

            # Run the async Mistral call in a new event loop (legacy sync path)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(asyncio.run, _call_mistral_async(prompt))
                        return future.result(timeout=90)
                return loop.run_until_complete(_call_mistral_async(prompt))
            except Exception:
                return _best_chunk_fallback(query, docs)

    return _QARunnable()
