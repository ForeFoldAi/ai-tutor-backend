from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import llm_api_key_for
from app.services import llm_client
from app.services.lesson_planner.schemas import ARTIFACT_SCHEMAS

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n([\s\S]*?)```", re.IGNORECASE)


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    match = _JSON_FENCE_RE.search(text)
    raw = match.group(1).strip() if match else text
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(raw[start : end + 1])
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def generate_artifact_markdown(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 8192,
    log_label: str = "artifact",
) -> str:
    """Generate teacher-ready markdown (not JSON)."""
    if not llm_api_key_for("lesson"):
        logger.warning("LLM API key missing — cannot generate %s markdown", log_label)
        return ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = llm_client.complete_sync(
        messages,
        feature="lesson",
        max_tokens=max_tokens,
        timeout=180.0,
    )

    text = (content or "").strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    return text


def generate_lesson_plan_markdown(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 8192,
) -> str:
    """Generate teacher-ready markdown lesson plan (not JSON)."""
    return generate_artifact_markdown(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        log_label="lesson plan",
    )


def generate_artifact_json(
    *,
    artifact_type: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    if not llm_api_key_for("lesson"):
        logger.warning("LLM API key missing — returning empty artifact for %s", artifact_type)
        return {}

    schema_cls = ARTIFACT_SCHEMAS.get(artifact_type)
    schema_hint = schema_cls.model_json_schema() if schema_cls else {}

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n\n"
                f"Return ONLY valid JSON matching this schema:\n{json.dumps(schema_hint, indent=2)}"
            ),
        },
    ]

    content = llm_client.complete_sync(
        messages,
        feature="lesson",
        max_tokens=max_tokens,
        timeout=120.0,
    )

    data = _extract_json(content)
    if schema_cls:
        try:
            return schema_cls.model_validate(data).model_dump()
        except Exception as exc:
            logger.warning("Artifact validation failed for %s: %s", artifact_type, exc)
            return data
    return data
