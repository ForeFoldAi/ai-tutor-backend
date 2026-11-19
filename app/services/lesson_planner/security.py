from __future__ import annotations

import re
from typing import Any

# ponytail: naive pattern strip — upgrade to allowlist + LLM guardrail for production hardening
_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"<\s*/?\s*script", re.I),
)


def sanitize_user_text(text: str, *, max_length: int = 5000) -> str:
    cleaned = (text or "").strip()[:max_length]
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()


def sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    for key in ("learning_objectives", "chapter_name", "title", "subject", "grade"):
        if key in out and isinstance(out[key], str):
            out[key] = sanitize_user_text(out[key], max_length=255 if key != "learning_objectives" else 5000)
    if isinstance(out.get("topics"), list):
        out["topics"] = [
            sanitize_user_text(str(t), max_length=120)
            for t in out["topics"]
            if str(t).strip()
        ][:8]
    if isinstance(out.get("sections"), list):
        out["sections"] = [
            sanitize_user_text(str(s), max_length=32)
            for s in out["sections"]
            if str(s).strip()
        ][:12]
    if "ppt_template" in out:
        from app.services.lesson_planner.export.pptx_themes import DEFAULT_THEME_ID, THEMES

        key = sanitize_user_text(str(out.get("ppt_template") or ""), max_length=64).lower().replace("-", "_")
        out["ppt_template"] = key if key in THEMES else DEFAULT_THEME_ID
    if "ppt_slide_count" in out:
        try:
            n = int(out.get("ppt_slide_count") or 12)
        except (TypeError, ValueError):
            n = 12
        if n <= 10:
            out["ppt_slide_count"] = 8
        elif n <= 14:
            out["ppt_slide_count"] = 12
        else:
            out["ppt_slide_count"] = 16
    return out
