"""
Real-time AI Voice Tutor via WebSocket.

Concurrent pipeline:
  LLM tokens → responsive chunking → asyncio.Queue → TTS worker → MP3 WebSocket frames
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from app.services.edge_tts_service import stream_edge_tts
from app.services.tts_sanitize import sanitize_chunk_for_tts
from app.services.voice_chunking import (
    VoicePipelineTiming,
    _IDLE_FLUSH_SEC,
    extract_responsive_chunks,
)

logger = logging.getLogger(__name__)

ws_router = APIRouter()

# Queue sentinel: end TTS worker for this turn
_TTS_STOP = None


@dataclass
class _ProducerState:
    buf: str = ""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_token_at: float = field(default_factory=time.monotonic)


async def _clear_queue(queue: asyncio.Queue[str | None]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break


async def tts_worker(
    websocket: WebSocket,
    sentence_queue: asyncio.Queue[str | None],
    stop_event: asyncio.Event,
    timing: VoicePipelineTiming,
    on_speaking: Callable[[], Awaitable[None]],
) -> None:
    """
    Consumer: process sentences sequentially — one Edge-TTS stream at a time.
    """
    while True:
        if stop_event.is_set():
            await _clear_queue(sentence_queue)

        try:
            sentence = await asyncio.wait_for(sentence_queue.get(), timeout=0.25)
        except asyncio.TimeoutError:
            if stop_event.is_set():
                await _clear_queue(sentence_queue)
                continue
            continue

        if sentence is _TTS_STOP:
            sentence_queue.task_done()
            break

        if stop_event.is_set() or not sentence:
            sentence_queue.task_done()
            continue

        await on_speaking()
        await stream_edge_tts(sentence, websocket, stop_event, timing=timing)
        sentence_queue.task_done()


async def _idle_flush_loop(
    state: _ProducerState,
    sentence_queue: asyncio.Queue[str | None],
    stop_event: asyncio.Event,
    timing: VoicePipelineTiming,
) -> None:
    """Flush partial buffer after token inactivity (500ms)."""
    while not stop_event.is_set():
        await asyncio.sleep(0.05)
        async with state.lock:
            chunk = state.buf.strip()
            idle = time.monotonic() - state.last_token_at
            if not chunk or idle < _IDLE_FLUSH_SEC:
                continue
            state.buf = ""

        await _enqueue_sanitized(sentence_queue, timing, chunk)


async def _enqueue_sanitized(
    sentence_queue: asyncio.Queue[str | None],
    timing: VoicePipelineTiming,
    raw_chunk: str,
) -> None:
    cleaned = sanitize_chunk_for_tts(raw_chunk)
    if not cleaned:
        return
    timing.mark_sentence_queued(cleaned)
    await sentence_queue.put(cleaned)


async def _enqueue_from_buffer(
    state: _ProducerState,
    sentence_queue: asyncio.Queue[str | None],
    timing: VoicePipelineTiming,
) -> None:
    async with state.lock:
        chunks, state.buf = extract_responsive_chunks(state.buf)
    for chunk in chunks:
        await _enqueue_sanitized(sentence_queue, timing, chunk)


class _Session:
    __slots__ = ("board", "class_level", "subject_name", "chapter_ids", "chapter", "chapter_names", "history")

    def __init__(self) -> None:
        self.board = ""
        self.class_level = ""
        self.subject_name = ""
        self.chapter_ids: list[str] | None = None
        self.chapter = ""
        self.chapter_names: list[str] = []
        self.history: list[dict] = []

    def configure(self, msg: dict) -> None:
        self.board = msg.get("board", self.board)
        self.class_level = msg.get("class_level", self.class_level)
        self.subject_name = msg.get("subject_name", self.subject_name)
        self.chapter_ids = msg.get("chapter_ids", self.chapter_ids)
        self.chapter = msg.get("chapter", self.chapter)
        raw_names = msg.get("chapter_names")
        if isinstance(raw_names, list):
            self.chapter_names = [str(x) for x in raw_names if str(x).strip()]
        elif isinstance(raw_names, str) and raw_names.strip():
            self.chapter_names = [raw_names.strip()]

    def remember(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > 20:
            self.history = self.history[-20:]

    @property
    def has_context(self) -> bool:
        return bool(self.board and self.subject_name)

    @property
    def collection(self) -> str:
        return f"{self.board}_{self.class_level}_{self.subject_name}".replace(" ", "_")


async def _send(ws: WebSocket, payload: dict) -> None:
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json(payload)
    except Exception:
        pass


async def _run_llm_producer(
    ws: WebSocket,
    token_iter: AsyncIterator[str],
    sentence_queue: asyncio.Queue[str | None],
    stop: asyncio.Event,
    timing: VoicePipelineTiming,
    full_tokens: list[str],
) -> None:
    """Producer: consume LLM tokens without blocking on TTS."""
    state = _ProducerState()
    idle_task = asyncio.create_task(
        _idle_flush_loop(state, sentence_queue, stop, timing),
        name="voice-idle-flush",
    )
    first_token = True

    try:
        async for token in token_iter:
            if stop.is_set():
                return

            if first_token:
                timing.mark_llm_first_token()
                first_token = False

            full_tokens.append(token)
            await _send(ws, {"type": "ai_text_token", "token": token})

            async with state.lock:
                state.buf += token
                state.last_token_at = time.monotonic()

            await _enqueue_from_buffer(state, sentence_queue, timing)

        async with state.lock:
            remainder = state.buf.strip()
            state.buf = ""

        if remainder and not stop.is_set():
            await _enqueue_sanitized(sentence_queue, timing, remainder)

    finally:
        idle_task.cancel()
        try:
            await idle_task
        except asyncio.CancelledError:
            pass


async def _general_answer_stream(question: str, class_level: str) -> AsyncIterator[str]:
    import httpx
    from app.config import MISTRAL_API_KEY, MISTRAL_MAX_TOKENS, MISTRAL_MODEL, MISTRAL_TEMPERATURE
    from app.services.chat_service import _GRADE_COMPLEXITY, _GRADE_LABELS

    if not MISTRAL_API_KEY:
        yield "Sorry, I need a Mistral API key to answer general questions."
        return

    grade = _GRADE_LABELS.get(class_level, "school student")
    complexity = _GRADE_COMPLEXITY.get(class_level, "Use clear simple language.")

    prompt = (
        f"You are a friendly AI Tutor for a {grade}.\n"
        f"Language rule: {complexity}\n"
        f"Voice rule: Use short spoken sentences. No bullet symbols, no markdown, no formatting.\n"
        f"Be warm and encouraging like a kind teacher.\n\n"
        f"Student asks: {question}\n\nAnswer:"
    )

    async with httpx.AsyncClient(timeout=90.0) as client:
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
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue


async def _pipeline_cached_answer(
    ws: WebSocket,
    answer: str,
    stop: asyncio.Event,
    timing: VoicePipelineTiming,
) -> None:
    speaking = False

    async def on_speaking() -> None:
        nonlocal speaking
        if not speaking:
            await _send(ws, {"type": "speaking"})
            speaking = True

    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
    tts_task = asyncio.create_task(
        tts_worker(ws, sentence_queue, stop, timing, on_speaking),
        name="voice-tts-worker",
    )

    for word in answer.split():
        if stop.is_set():
            break
        await _send(ws, {"type": "ai_text_token", "token": word + " "})

    for part in re.split(r"(?<=[.?!])\s+", answer):
        s = part.strip()
        if s:
            await _enqueue_sanitized(sentence_queue, timing, s)

    await sentence_queue.put(_TTS_STOP)
    await tts_task


async def _stream_answer(
    ws: WebSocket,
    question: str,
    session: _Session,
    stop: asyncio.Event,
) -> None:
    from app.core.cache import deserialize_tutor_cache, get_cached_answer
    from app.services.chat_service import chapter_aware_qa_stream

    turn_id = uuid.uuid4().hex[:8]
    timing = VoicePipelineTiming(turn_id=turn_id)
    logger.info("[voice %s] Turn started", turn_id)

    if session.has_context:
        cached = await get_cached_answer(session.collection, session.chapter_ids, question)
        if cached and not stop.is_set():
            await _send(ws, {"type": "thinking"})
            answer, imgs = deserialize_tutor_cache(cached)
            await _send(ws, {"type": "related_images", "images": imgs})
            await _pipeline_cached_answer(ws, answer, stop, timing)
            if not stop.is_set():
                await _send(ws, {"type": "done"})
            session.remember("user", question)
            session.remember("assistant", answer)
            logger.info("[voice %s] Timing %s", turn_id, timing.summary())
            return

    if stop.is_set():
        return

    await _send(ws, {"type": "thinking"})

    full_tokens: list[str] = []
    speaking = False

    async def on_speaking() -> None:
        nonlocal speaking
        if not speaking:
            await _send(ws, {"type": "speaking"})
            speaking = True

    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
    tts_task = asyncio.create_task(
        tts_worker(ws, sentence_queue, stop, timing, on_speaking),
        name="voice-tts-worker",
    )

    async def _emit_imgs(imgs: list[dict]) -> None:
        if stop.is_set():
            return
        await _send(ws, {"type": "related_images", "images": imgs})

    try:
        if session.has_context:
            token_iter = chapter_aware_qa_stream(
                question,
                collection_name=session.collection,
                chapter_ids=session.chapter_ids,
                class_level=session.class_level,
                board=session.board,
                subject_name=session.subject_name,
                chapter=session.chapter,
                chapter_names=session.chapter_names,
                emit_related_images=_emit_imgs,
                conversation_history=session.history,
            )
        else:
            token_iter = _general_answer_stream(question, session.class_level)

        await _run_llm_producer(ws, token_iter, sentence_queue, stop, timing, full_tokens)

    except asyncio.CancelledError:
        await _clear_queue(sentence_queue)
        raise
    except Exception as exc:
        logger.error("Voice generation error: %s", exc, exc_info=True)
        if not stop.is_set():
            await _send(ws, {"type": "error", "message": "Answer generation failed."})
        return
    finally:
        await sentence_queue.put(_TTS_STOP)
        try:
            await asyncio.wait_for(tts_task, timeout=120.0)
        except asyncio.TimeoutError:
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass

    if stop.is_set():
        logger.info("[voice %s] Interrupted — %s", turn_id, timing.summary())
        return

    await _send(ws, {"type": "done"})
    logger.info("[voice %s] Timing %s", turn_id, timing.summary())

    full_answer = "".join(full_tokens)
    session.remember("user", question)
    session.remember("assistant", full_answer)


@ws_router.websocket("/ws/voice")
async def voice_ws(
    websocket: WebSocket,
    token: str = Query(default=""),
):
    await websocket.accept()

    if token:
        try:
            from app.core.security import decode_token
            decode_token(token)
        except Exception:
            pass

    logger.info("Voice WS connected")

    session = _Session()
    stop = asyncio.Event()
    gen_task: asyncio.Task | None = None

    async def _cancel_gen() -> None:
        nonlocal gen_task
        if gen_task and not gen_task.done():
            stop.set()
            try:
                await asyncio.wait_for(asyncio.shield(gen_task), timeout=1.5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                gen_task.cancel()
                try:
                    await gen_task
                except (asyncio.CancelledError, Exception):
                    pass
        stop.clear()
        gen_task = None

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=120)
            except asyncio.TimeoutError:
                await _send(websocket, {"type": "ping"})
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type", "")

            if mtype == "session_start":
                session.configure(msg)
                await _send(websocket, {"type": "listening"})

            elif mtype == "question":
                text = (msg.get("text") or "").strip()
                if not text or len(text) < 2:
                    continue
                await _cancel_gen()
                stop.clear()
                gen_task = asyncio.create_task(
                    _stream_answer(websocket, text, session, stop),
                    name="voice-gen",
                )

            elif mtype == "interrupt":
                await _cancel_gen()
                await _send(websocket, {"type": "interrupt_ack"})
                await _send(websocket, {"type": "listening"})

            elif mtype == "stop":
                await _cancel_gen()
                break

            elif mtype == "ping":
                await _send(websocket, {"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Voice WS disconnected")
    except Exception as exc:
        logger.error("Voice WS fatal error: %s", exc, exc_info=True)
        try:
            await _send(websocket, {"type": "error", "message": "Server error."})
        except Exception:
            pass
    finally:
        await _cancel_gen()
        logger.info("Voice WS session ended")
