"""Learning Intelligence / monitoring adapters."""

from __future__ import annotations

from typing import Any


def sample_tutor_guidance(*, student_user_id: int = 1, query: str = "What is weather?") -> dict[str, Any]:
    """
    Attempt a real LIA guidance call; return structured diagnostic on failure.

    Production path: learning_intelligence orchestrator.
    """
    import os

    if os.environ.get("EVAL_SKIP_LIA_RUNTIME", "").lower() in ("1", "true", "yes"):
        return {"ok": False, "error": "EVAL_SKIP_LIA_RUNTIME", "guidance": None}
    try:
        from app.modules.learning_intelligence.schemas import TutorGuidanceRequest
        from app.services.learning_intelligence.orchestration.orchestrator import (
            get_tutor_guidance,
        )
        from app.core.database import SessionLocal  # type: ignore

        db = SessionLocal()
        try:
            try:
                db.execute(__import__("sqlalchemy").text("SELECT 1"))
            except Exception as db_exc:  # noqa: BLE001
                return {"ok": False, "error": f"db_unreachable: {db_exc}", "guidance": None}
            req = TutorGuidanceRequest(student_user_id=student_user_id, query=query)
            out = get_tutor_guidance(db, req)
            if hasattr(out, "model_dump"):
                return {"ok": True, "guidance": out.model_dump()}
            if isinstance(out, dict):
                return {"ok": True, "guidance": out}
            return {"ok": True, "guidance": {"raw": str(out)}}
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "guidance": None}


def prompt_sections_from_guidance(guidance: dict[str, Any] | None) -> list[str]:
    """Build prompt-augmentation evidence from a guidance dict or object."""
    if not guidance:
        return []

    # Prefer already-rendered prompt_instructions from TutorGuidanceObject
    pi = guidance.get("prompt_instructions") if isinstance(guidance, dict) else None
    if isinstance(pi, str) and pi.strip():
        return [pi.strip()]

    try:
        from app.modules.learning_intelligence.schemas import TutorGuidanceObject
        from app.services.learning_intelligence.prompt_builder import (
            guidance_to_prompt_sections,
        )

        obj = guidance if not isinstance(guidance, dict) else TutorGuidanceObject.model_validate(guidance)
        learner, adaptive, full = guidance_to_prompt_sections(obj, profile={})
        sections = [s for s in (learner, adaptive, full) if (s or "").strip()]
        if sections:
            return sections
    except Exception:
        pass

    # Structural fallback from guidance fields
    keys = (
        "preferred_explanation",
        "teaching_pace",
        "difficulty",
        "next_best_action",
        "hint_level",
        "learning_style",
    )
    return [f"{k}={guidance[k]}" for k in keys if isinstance(guidance, dict) and guidance.get(k) not in (None, "")]
