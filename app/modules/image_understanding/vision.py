"""Vision provider abstraction (OpenAI-compatible multimodal chat)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any, Protocol

import httpx

from app.config import IMAGE_ANALYSIS_TIMEOUT, IMAGE_VISION_PROVIDER, VISION_MAX_RETRIES, llm_model_for
from app.modules.image_understanding.errors import VisionProviderError
from app.modules.image_understanding.prompts import VISION_SYSTEM_PROMPT, vision_user_text
from app.modules.image_understanding.schemas import ImageUnderstandingResult, sanitize_vision_payload

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)
_RETRY_WAIT_SEC = 1.0
_MIN_ATTEMPT_SEC = 5.0
# Room for ~1500 chars of ocr_text in Indic scripts (≈1 token/char) plus the rest of the JSON.
_MAX_TOKENS = 3000
MALFORMED_OUTPUT = "model_output_malformed"


class VisionProvider(Protocol):
    async def analyze_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        *,
        student_message: str = "",
        class_level: str = "",
        subject_name: str = "",
        board: str = "",
    ) -> ImageUnderstandingResult: ...


def _parse_json_content(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    m = _JSON_FENCE.search(raw)
    if m:
        raw = m.group(1).strip()
    # strict=False: vision models put literal newlines inside string values (e.g. ocr_text)
    try:
        obj = json.loads(raw, strict=False)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1], strict=False)
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


def _describe(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_{exc.response.status_code}"
    return type(exc).__name__


class OpenAICompatibleVisionProvider:
    """Mistral Pixtral (or any OpenAI-compatible vision model set by LLM_VISION_MODEL)."""

    attempts: int = 0

    async def analyze_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        *,
        student_message: str = "",
        class_level: str = "",
        subject_name: str = "",
        board: str = "",
    ) -> ImageUnderstandingResult:
        from app.services import llm_client

        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{b64}"
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": vision_user_text(
                    student_message=student_message,
                    class_level=class_level,
                    subject_name=subject_name,
                    board=board,
                ),
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        attempts = 1 + VISION_MAX_RETRIES
        # One wall-clock budget for all attempts: httpx's timeout is per connect/read, not per call.
        deadline = time.monotonic() + IMAGE_ANALYSIS_TIMEOUT
        text = ""
        for attempt in range(1, attempts + 1):
            self.attempts = attempt
            remaining = deadline - time.monotonic()
            last = attempt == attempts or remaining < _MIN_ATTEMPT_SEC * 2
            try:
                text = await asyncio.wait_for(
                    llm_client.complete(
                        messages,
                        feature="vision",
                        max_tokens=_MAX_TOKENS,
                        temperature=0.1,
                        empty_fallback="",
                        timeout=remaining,
                    ),
                    timeout=remaining,
                )
            except Exception as exc:
                if not last and _is_transient(exc) and deadline - time.monotonic() > _MIN_ATTEMPT_SEC + _RETRY_WAIT_SEC:
                    logger.warning("vision transient failure (%s), retry %d/%d", _describe(exc), attempt, attempts - 1)
                    await asyncio.sleep(_RETRY_WAIT_SEC)
                    continue
                logger.exception("vision provider call failed")
                raise VisionProviderError() from None

            parsed = _parse_json_content(text)
            if parsed is not None:
                return sanitize_vision_payload(parsed)
            logger.warning("vision returned non-JSON (len=%d) attempt %d/%d", len(text or ""), attempt, attempts)
            if last:
                break

        return ImageUnderstandingResult(
            content_summary="I could not fully parse the image analysis.",
            confidence=0.2,
            unclear_regions=[MALFORMED_OUTPUT],
        )


def get_vision_provider() -> VisionProvider:
    # ponytail: one provider; swap via IMAGE_VISION_PROVIDER when a second exists
    _ = IMAGE_VISION_PROVIDER
    return OpenAICompatibleVisionProvider()


def vision_model_name() -> str:
    return llm_model_for("vision")
