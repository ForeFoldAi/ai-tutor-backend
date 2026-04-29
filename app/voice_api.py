"""
Voice assistant REST API integrated into `ai-tutor-backend`.

This is based on `voice/rest_api.py`, but adapted to:
- expose routes via a FastAPI `APIRouter` (so it lives inside the backend)
- use Mistral (instead of Groq) for the language-model part

Frontend contract (see `ai-tutor-frontend/src/pages/ai-voice.tsx`):
- `POST /voice-stream` streams a framed binary stream:
  Frame format: [type: 1 byte][length: 4 bytes LE][data: N bytes]
  type 1 -> UTF-8 text token
  type 2 -> int16 LE PCM chunk @ 24,000 Hz mono
  type 3 -> done
"""

from __future__ import annotations

import io
import json
import os
import re
import struct
import threading
from typing import Any, Iterator

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    import soundfile as sf  # type: ignore
except Exception:  # allow backend to start without voice deps
    sf = None

try:
    from fastrtc import get_tts_model  # type: ignore
except Exception:  # allow backend to start without voice deps
    get_tts_model = None

try:
    from loguru import logger  # type: ignore
except Exception:
    import logging

    logger = logging.getLogger("voice_api")

load_dotenv()

# Backend base URL used to proxy the existing /upload and /chat endpoints.
# Since voice is integrated into this backend, default to the current local port.
AI_TUTOR_API_URL = os.environ.get("AI_TUTOR_API_URL", "http://127.0.0.1:8004").rstrip("/")

# Mistral settings for the voice assistant.
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_MAX_TOKENS = int(os.environ.get("MISTRAL_MAX_TOKENS", "250"))
MISTRAL_TEMPERATURE = float(os.environ.get("MISTRAL_VOICE_TEMPERATURE", "0.4"))

router = APIRouter(prefix="")

_tts_model: Any = None


def _require_tts_model():
    """
    Lazily load TTS so the backend can start even if voice dependencies
    aren't installed yet.
    """
    global _tts_model
    if _tts_model is not None:
        return _tts_model

    # If imports failed during server startup, retry when the endpoint is hit.
    global get_tts_model
    if get_tts_model is None:
        try:
            from fastrtc import get_tts_model as _get_tts_model  # type: ignore

            get_tts_model = _get_tts_model
        except Exception as e:
            raise RuntimeError(
                "Voice dependencies are missing. Install `fastrtc` (and friends) "
                "so /voice-stream can synthesize audio."
            ) from e

    _tts_model = get_tts_model()
    return _tts_model

SYSTEM_PROMPT = """You are Jarvis, a smart voice assistant.

Available tools — use ONLY when needed:
1. web_search(query)    — search the web
2. read_pdf(path)       — read a PDF file
3. open_app(name)       — open an application
4. move_mouse(x,y)      — move the mouse cursor

If a tool is needed, reply ONLY in this exact format:
TOOL: tool_name(arguments)

Otherwise answer clearly and concisely. Do NOT use emojis or special characters.
"""

# Simple per-conversation memory store.
_memory_lock = threading.Lock()
_memory_by_conversation: dict[str, list[dict[str, str]]] = {}

# Avoid overlapping requests impacting shared tools/memory.
_lock_for_llm = threading.Lock()


class JarvisChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"


class ChatVoiceRequest(BaseModel):
    message: str


def _get_memory(conversation_id: str) -> list[dict[str, str]]:
    with _memory_lock:
        return _memory_by_conversation.setdefault(conversation_id, [])


def _append_memory(conversation_id: str, role: str, content: str) -> None:
    with _memory_lock:
        mem = _memory_by_conversation.setdefault(conversation_id, [])
        mem.append({"role": role, "content": content})
        # Keep it bounded.
        if len(mem) > 12:
            del mem[:-12]


def web_search(query: str) -> str:
    try:
        from duckduckgo_search import DDGS  # type: ignore

        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=3)]
        return " ".join(results)
    except Exception as e:
        return f"Web search failed: {e}"


def read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(path.strip())
        return "".join(p.extract_text() for p in reader.pages)[:2000]
    except Exception as e:
        return f"Could not read PDF: {e}"


def open_app(name: str) -> str:
    # Note: may be restricted depending on your environment.
    import subprocess

    try:
        subprocess.Popen(name.strip())
        return f"Opening {name}"
    except Exception as e:
        return f"Could not open application: {e}"


def move_mouse(x: str, y: str) -> str:
    # Note: pyautogui may require permissions on macOS.
    import pyautogui

    try:
        pyautogui.moveTo(int(x), int(y))
        return "Mouse moved"
    except Exception as e:
        return f"Mouse move failed: {e}"


def execute_tool(tool_call: str) -> str:
    try:
        if tool_call.startswith("web_search"):
            query = tool_call.split("(", 1)[1].rsplit(")", 1)[0]
            return web_search(query)
        if tool_call.startswith("read_pdf"):
            path = tool_call.split("(", 1)[1].rsplit(")", 1)[0]
            return read_pdf(path)
        if tool_call.startswith("open_app"):
            name = tool_call.split("(", 1)[1].rsplit(")", 1)[0]
            return open_app(name)
        if tool_call.startswith("move_mouse"):
            args = tool_call.split("(", 1)[1].rsplit(")", 1)[0]
            x, y = args.split(",")
            return move_mouse(x.strip(), y.strip())
    except Exception as e:
        return f"Tool execution failed: {e}"
    return "Unknown tool."


def ask_llm_text(user_input: str, conversation_id: str) -> str:
    # Prefer PDF-grounded backend if available.
    try:
        rag_resp = requests.post(
            f"{AI_TUTOR_API_URL}/chat",
            json={"query": user_input},
            timeout=90,
        )
        if rag_resp.ok:
            rag_json = rag_resp.json()
            rag_answer = (rag_json.get("answer") or "").strip()
            if rag_answer:
                _append_memory(conversation_id, "user", user_input)
                _append_memory(conversation_id, "assistant", rag_answer)
                return rag_answer
    except Exception as e:
        logger.debug(f"RAG backend unavailable, falling back to Mistral: {e}")

    if not MISTRAL_API_KEY:
        raise RuntimeError("Missing MISTRAL_API_KEY env var (required for the voice assistant).")

    mem = _get_memory(conversation_id)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(mem[-6:])
    messages.append({"role": "user", "content": user_input})

    # Avoid overlapping requests impacting shared tools/memory.
    with _lock_for_llm:
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MISTRAL_MODEL,
                "messages": messages,
                "max_tokens": MISTRAL_MAX_TOKENS,
                "temperature": MISTRAL_TEMPERATURE,
            },
            timeout=90,
        )
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        text = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        # Tool call convention used by SYSTEM_PROMPT.
        if text.startswith("TOOL:"):
            tool_call = text.replace("TOOL:", "").strip()
            logger.debug(f"Tool call: {tool_call}")
            text = execute_tool(tool_call)

    _append_memory(conversation_id, "user", user_input)
    _append_memory(conversation_id, "assistant", text)
    return text


def _stream_llm_tokens(user_input: str, conversation_id: str) -> Iterator[str]:
    """
    Streams Mistral tokens via SSE and yields them as strings.

    Handles TOOL: prefix by buffering the initial prefix, silently draining
    the rest of the stream, then yielding the tool result as a single token.
    """
    # First choice: PDF-aware backend answer (if running). We stream it word-by-word
    # to keep the same frontend contract.
    try:
        rag_resp = requests.post(
            f"{AI_TUTOR_API_URL}/chat",
            json={"query": user_input},
            timeout=90,
        )
        if rag_resp.ok:
            rag_json = rag_resp.json()
            rag_answer = (rag_json.get("answer") or "").strip()
            if rag_answer:
                parts = rag_answer.split(" ")
                for i, p in enumerate(parts):
                    if i < len(parts) - 1:
                        yield p + " "
                    else:
                        yield p
                return
    except Exception as e:
        logger.debug(f"RAG streaming source unavailable, falling back to Mistral: {e}")

    if not MISTRAL_API_KEY:
        yield "Sorry, neither AI_TUTOR_API_URL nor MISTRAL_API_KEY is available."
        return

    mem = _get_memory(conversation_id)
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(mem[-6:])
    messages.append({"role": "user", "content": user_input})

    try:
        with _lock_for_llm:
            resp = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"},
                json={
                    "model": MISTRAL_MODEL,
                    "messages": messages,
                    "max_tokens": MISTRAL_MAX_TOKENS,
                    "temperature": MISTRAL_TEMPERATURE,
                    "stream": True,
                },
                stream=True,
                timeout=90,
            )
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Mistral streaming request failed: {e}")
        yield "Sorry, I'm having trouble connecting to the language model."
        return

    prefix_buf = ""
    prefix_done = False

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue

        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue

        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break

        try:
            payload = json.loads(data)
            delta = (
                payload.get("choices", [{}])[0]
                .get("delta", {})
                .get("content", "")
                or payload.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception:
            continue

        if not delta:
            continue

        if prefix_done:
            yield delta
            continue

        prefix_buf += delta

        # Once we have enough chars to detect TOOL:, decide which mode.
        if "\n" in prefix_buf or len(prefix_buf) >= 60:
            prefix_done = True
            if prefix_buf.lstrip().startswith("TOOL:"):
                # Drain the rest of the stream silently.
                for rl2 in resp.iter_lines():
                    if not rl2:
                        continue
                    l2 = rl2.decode("utf-8") if isinstance(rl2, bytes) else rl2
                    if not l2.startswith("data:"):
                        continue
                    d2 = l2[len("data:") :].strip()
                    if d2 == "[DONE]":
                        break
                    try:
                        prefix_buf += (
                            json.loads(d2)
                            .get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                    except Exception:
                        continue

                tool_call = prefix_buf.replace("TOOL:", "").strip()
                logger.debug(f"Tool call: {tool_call}")
                yield execute_tool(tool_call)
                return

            # Normal text: flush the buffered prefix once, then continue.
            yield prefix_buf

    # Short response that never left the prefix buffer.
    if not prefix_done and prefix_buf:
        if prefix_buf.lstrip().startswith("TOOL:"):
            yield execute_tool(prefix_buf.replace("TOOL:", "").strip())
        else:
            yield prefix_buf


def _frame(ftype: int, data: bytes) -> bytes:
    """Pack a binary frame: [type: 1 byte][length: 4 bytes LE][data]."""
    return struct.pack("<BI", ftype, len(data)) + data


# Frame type constants
_FRAME_TEXT = 1  # UTF-8 text token
_FRAME_AUDIO = 2  # int16 LE PCM chunk at 24,000 Hz mono
_FRAME_DONE = 3  # stream finished (empty data)


def _voice_stream_generator(user_input: str, conversation_id: str) -> Iterator[bytes]:
    """
    Unified LLM + TTS streaming generator.

    Yields binary frames that carry BOTH text tokens and audio PCM chunks so
    the frontend can display text and play audio at the same time.
    """
    sentence_buf = ""
    full_reply_parts: list[str] = []

    def tts_frames(sentence: str) -> Iterator[bytes]:
        tts_model = _require_tts_model()
        for sr, chunk in tts_model.stream_tts_sync(sentence):
            if chunk is None:
                continue
            pcm = (
                np.asarray(chunk, dtype=np.float32) * 32767
            ).clip(-32768, 32767).astype(np.int16)
            yield _frame(_FRAME_AUDIO, pcm.tobytes())

    for token in _stream_llm_tokens(user_input, conversation_id):
        full_reply_parts.append(token)
        sentence_buf += token

        # Emit the text token immediately so the UI can show it.
        yield _frame(_FRAME_TEXT, token.encode("utf-8"))

        # Flush every complete sentence to TTS right away.
        segments = re.split(r"(?<=[.?!])\s+", sentence_buf)
        for complete in segments[:-1]:
            s = complete.strip()
            if s:
                logger.debug(f"TTS ▶ {s}")
                yield from tts_frames(s)
        sentence_buf = segments[-1]

        # Low-latency fallback: if a sentence runs long without terminal punctuation,
        # flush on a soft punctuation or at a whitespace boundary after ~90 chars.
        if len(sentence_buf) >= 90:
            soft_split = max(
                sentence_buf.rfind(", "),
                sentence_buf.rfind("; "),
                sentence_buf.rfind(": "),
            )
            if soft_split < 20:
                soft_split = sentence_buf.rfind(" ")
            if soft_split > 20:
                early = sentence_buf[:soft_split].strip()
                sentence_buf = sentence_buf[soft_split:].strip()
                if early:
                    logger.debug(f"TTS ▶ {early} (early flush)")
                    yield from tts_frames(early)

    # Flush any trailing text that didn't end with punctuation.
    if sentence_buf.strip():
        logger.debug(f"TTS ▶ {sentence_buf.strip()} (flush)")
        yield from tts_frames(sentence_buf.strip())

    # Persist conversation memory.
    full_reply = "".join(full_reply_parts)
    _append_memory(conversation_id, "user", user_input)
    _append_memory(conversation_id, "assistant", full_reply)

    yield _frame(_FRAME_DONE, b"")


def synthesize_wav_bytes(text: str) -> bytes:
    # Kokoro yields float32 samples + samplerate, we just concatenate.
    audio_samples: list[tuple[int, Any]] = []
    sample_rate: int | None = None
    global sf
    if sf is None:
        try:
            import soundfile as _sf  # type: ignore

            sf = _sf
        except ModuleNotFoundError as e:
            raise RuntimeError("soundfile is not installed; cannot synthesize WAV audio.") from e
    tts_model = _require_tts_model()

    for sr, chunk in tts_model.stream_tts_sync(text):
        sample_rate = sample_rate or sr
        audio_samples.append((sr, chunk))

    if sample_rate is None:
        sample_rate = 24000

    concatenated = np.concatenate([c for _sr, c in audio_samples]).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, concatenated, sample_rate, format="WAV")
    return buf.getvalue()


def _wav_header(sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """WAV header with placeholder data size for streaming."""
    data_size = 0xFFFFFFFF  # unknown size
    riff_size = 0xFFFFFFFF
    return (
        struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
        + struct.pack(
            "<4sIHHIIHH",
            b"fmt ",
            16,
            1,  # PCM
            num_channels,
            sample_rate,
            sample_rate * num_channels * bits_per_sample // 8,
            num_channels * bits_per_sample // 8,
            bits_per_sample,
        )
        + struct.pack("<4sI", b"data", data_size)
    )


def _stream_tts_wav(text: str) -> Iterator[bytes]:
    """
    Generator that yields a WAV header followed by int16 PCM chunks as Kokoro
    produces them.
    """
    sample_rate = 24000
    header_sent = False
    tts_model = _require_tts_model()

    for sr, chunk in tts_model.stream_tts_sync(text):
        if chunk is None:
            continue
        if not header_sent:
            sample_rate = sr or sample_rate
            yield _wav_header(sample_rate)
            header_sent = True
        pcm = (
            np.asarray(chunk, dtype=np.float32) * 32767
        ).clip(-32768, 32767).astype(np.int16)
        yield pcm.tobytes()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and let the existing backend `/upload` build the vector store.
    """
    try:
        filename = file.filename or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            resp = requests.post(
                f"{AI_TUTOR_API_URL}/upload",
                files={
                    "file": (
                        filename,
                        content,
                        file.content_type or "application/pdf",
                    )
                },
                timeout=300,
            )
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to reach AI tutor backend upload endpoint: {e}",
            )

        if not resp.ok:
            upstream = ""
            try:
                payload = resp.json()
                if isinstance(payload, dict):
                    upstream = str(payload.get("detail") or payload.get("message") or "").strip()
            except Exception:
                upstream = ""
            upstream = upstream or (resp.text or "").strip()
            if not upstream:
                upstream = "PDF upload failed on backend."
            if len(upstream) > 400:
                upstream = upstream[:400] + "..."
            raise HTTPException(status_code=resp.status_code, detail=upstream)

        try:
            return resp.json()
        except Exception:
            return {"message": "PDF uploaded successfully."}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected /upload-pdf failure")
        raise HTTPException(status_code=500, detail=f"Unexpected /upload-pdf failure: {e}")


@router.post("/jarvis-chat")
def jarvis_chat(req: JarvisChatRequest) -> dict[str, str]:
    text = ask_llm_text(req.message, req.conversation_id)
    return {"text": text, "conversation_id": req.conversation_id}


@router.post("/chat-voice", response_class=None)
def chat_voice(req: ChatVoiceRequest):
    """
    Non-streaming TTS endpoint (kept for backwards compatibility).
    """
    wav_bytes = synthesize_wav_bytes(req.message)
    from fastapi.responses import Response

    return Response(content=wav_bytes, media_type="audio/wav")


@router.post("/chat-voice-stream")
def chat_voice_stream(req: ChatVoiceRequest):
    """
    Streaming TTS endpoint.
    """
    return StreamingResponse(
        _stream_tts_wav(req.message),
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@router.post("/voice-stream")
def voice_stream(req: JarvisChatRequest):
    """
    Combined LLM + TTS streaming endpoint.
    """
    return StreamingResponse(
        _voice_stream_generator(req.message, req.conversation_id),
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )

