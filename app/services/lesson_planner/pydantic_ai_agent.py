from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Type

from pydantic import BaseModel

from app.config import llm_api_key_for, llm_base_url_for, llm_model_for
from app.services.lesson_planner.llm import generate_artifact_json
from app.services.lesson_planner.observability.tracing import trace_span

logger = logging.getLogger(__name__)

_USE_PYDANTIC_AI = os.environ.get("LESSON_PLANNER_USE_PYDANTIC_AI", "true").lower() in {"1", "true", "yes"}
_MISTRAL_HOST = "mistral.ai"


def _pydantic_ai_available() -> bool:
    # pydantic-ai mistral: provider only when still on Mistral defaults
    if not llm_api_key_for("lesson") or not _USE_PYDANTIC_AI:
        return False
    if _MISTRAL_HOST not in llm_base_url_for("lesson"):
        return False
    try:
        import pydantic_ai  # noqa: F401
        return True
    except ImportError:
        return False


async def _run_pydantic_agent(
    *,
    system_prompt: str,
    user_prompt: str,
    schema_cls: Type[BaseModel],
) -> dict[str, Any]:
    from pydantic_ai import Agent

    model = llm_model_for("lesson")
    model_name = model if model.startswith("mistral-") else f"mistral-{model}"
    agent = Agent(
        f"mistral:{model_name}",
        output_type=schema_cls,
        system_prompt=system_prompt,
    )
    result = await agent.run(user_prompt)
    if hasattr(result, "output"):
        out = result.output
        return out.model_dump() if isinstance(out, BaseModel) else dict(out)
    if hasattr(result, "data"):
        data = result.data
        return data.model_dump() if isinstance(data, BaseModel) else dict(data)
    return {}


def generate_structured_artifact(
    *,
    artifact_type: str,
    system_prompt: str,
    user_prompt: str,
    schema_cls: Type[BaseModel] | None,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """PydanticAI structured generation with httpx JSON fallback."""
    with trace_span("lesson_planner.generate_artifact", attributes={"artifact_type": artifact_type}):
        if schema_cls and _pydantic_ai_available():
            try:
                return asyncio.run(
                    _run_pydantic_agent(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        schema_cls=schema_cls,
                    )
                )
            except Exception as exc:
                logger.warning("PydanticAI failed for %s, falling back to httpx: %s", artifact_type, exc)

        return generate_artifact_json(
            artifact_type=artifact_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
