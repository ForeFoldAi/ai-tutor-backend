"""LLM final pick: chapter image candidates → best-fit for question + answer.

Fail-closed: timeout / parse / low confidence → [].
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM = """\
You pick textbook figures that ILLUSTRATE the tutor's answer to the student's question.
Only choose from the numbered candidates. Prefer none over a weak match.
Return ONLY JSON: {"ids":["0","2"],"confidence":0.0}
ids = candidate index strings (0-based). confidence 0-1 for the whole set.
If nothing clearly fits, return {"ids":[],"confidence":0.0}.
"""


def _parse_select_json(raw: str) -> tuple[list[str], float] | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    ids = data.get("ids")
    if ids is None:
        ids = []
    if not isinstance(ids, list):
        return None
    try:
        confidence = float(data.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    out_ids = [str(i).strip() for i in ids if str(i).strip() != ""]
    return out_ids, max(0.0, min(1.0, confidence))


def _candidate_lines(candidates: list[dict]) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates):
        fig = c.get("figure_number") or ""
        cap = (c.get("caption") or c.get("title") or "")[:180]
        kind = c.get("content_kind") or ""
        page = c.get("page")
        lines.append(
            f'{i}: figure={fig!r} page={page} kind={kind!r} caption={cap!r}'
        )
    return "\n".join(lines)


def apply_select_result(
    candidates: list[dict],
    ids: list[str],
    *,
    max_images: int,
) -> list[dict]:
    """Map index ids → candidate payloads; drop invalid / over-max."""
    picked: list[dict] = []
    seen: set[int] = set()
    for raw in ids:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(candidates) or idx in seen:
            continue
        seen.add(idx)
        picked.append(candidates[idx])
        if len(picked) >= max_images:
            break
    return picked


async def select_images_for_qa(
    question: str,
    answer: str,
    candidates: list[dict],
    *,
    max_images: int | None = None,
    min_confidence: float | None = None,
    timeout_sec: float | None = None,
    candidate_cap: int | None = None,
) -> list[dict]:
    """Return best-fit images for this Q+A, or [] on any failure."""
    from app.config import (
        LLM_IMAGE_SELECT_CANDIDATES,
        LLM_IMAGE_SELECT_MAX,
        LLM_IMAGE_SELECT_MIN_CONFIDENCE,
        LLM_IMAGE_SELECT_TIMEOUT_SEC,
    )
    from app.services.llm_client import complete

    if max_images is None:
        max_images = LLM_IMAGE_SELECT_MAX
    if min_confidence is None:
        min_confidence = LLM_IMAGE_SELECT_MIN_CONFIDENCE
    if timeout_sec is None:
        timeout_sec = LLM_IMAGE_SELECT_TIMEOUT_SEC
    if candidate_cap is None:
        candidate_cap = LLM_IMAGE_SELECT_CANDIDATES

    pool = list(candidates or [])[: max(0, int(candidate_cap))]
    if not pool:
        logger.info(
            "llm_image_select fail=empty_pool candidates=0 selected=0"
        )
        return []

    q = (question or "").strip()[:800]
    a = (answer or "").strip()[:1500]
    if not q and not a:
        logger.info("llm_image_select fail=empty_qa candidates=%d selected=0", len(pool))
        return []

    user = (
        f"Student question:\n{q or '(none)'}\n\n"
        f"Tutor answer:\n{a or '(none)'}\n\n"
        f"Candidates:\n{_candidate_lines(pool)}\n"
    )
    t0 = time.monotonic()
    fail = ""
    try:
        import asyncio

        raw = await asyncio.wait_for(
            complete(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user},
                ],
                feature="image_select",
                max_tokens=80,
            ),
            timeout=float(timeout_sec),
        )
    except Exception as exc:
        fail = "timeout" if "Timeout" in type(exc).__name__ or "timeout" in str(exc).lower() else "llm_error"
        logger.info(
            "llm_image_select fail=%s candidates=%d selected=0 latency_ms=%d err=%s",
            fail,
            len(pool),
            int((time.monotonic() - t0) * 1000),
            type(exc).__name__,
        )
        return []

    parsed = _parse_select_json(raw)
    if parsed is None:
        logger.info(
            "llm_image_select fail=parse candidates=%d selected=0 latency_ms=%d",
            len(pool),
            int((time.monotonic() - t0) * 1000),
        )
        return []

    ids, confidence = parsed
    if confidence < float(min_confidence):
        logger.info(
            "llm_image_select fail=low_conf candidates=%d selected=0 confidence=%.2f latency_ms=%d",
            len(pool),
            confidence,
            int((time.monotonic() - t0) * 1000),
        )
        return []

    selected = apply_select_result(pool, ids, max_images=int(max_images))
    figs = [s.get("figure_number") or s.get("file_name") for s in selected]
    logger.info(
        "llm_image_select fail=%s candidates=%d selected=%d confidence=%.2f figs=%s latency_ms=%d",
        "empty" if not selected else "ok",
        len(pool),
        len(selected),
        confidence,
        figs,
        int((time.monotonic() - t0) * 1000),
    )
    return selected
