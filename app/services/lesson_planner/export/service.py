from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.core.database import SessionLocal
from app.modules.teacher.lesson_planner.constants import ExportStatus
from app.modules.teacher.lesson_planner.models import LessonArtifact, LessonExport, LessonPlan
from app.modules.teacher.lesson_planner.service import build_plan_snapshot
from app.services.lesson_planner.export.templates import render_artifact_lines

logger = logging.getLogger(__name__)

EXPORT_DIR = Path(os.environ.get("LESSON_PLANNER_EXPORT_DIR", "exports/lesson_planner"))


def run_export(export_id: str, artifact_types: list[str] | None = None) -> dict:
    db = SessionLocal()
    try:
        export = db.get(LessonExport, int(export_id))
        if not export:
            return {"ok": False, "error": "export not found"}

        plan = db.get(LessonPlan, export.lesson_plan_id)
        if not plan:
            export.status = ExportStatus.FAILED
            export.error_message = "Lesson plan not found"
            db.commit()
            return {"ok": False, "error": "plan not found"}

        artifacts = list(
            db.scalars(
                select(LessonArtifact).where(LessonArtifact.lesson_plan_id == plan.id)
            ).all()
        )
        plan.artifacts = artifacts
        snapshot = build_plan_snapshot(plan)
        artifacts_map = dict(snapshot.get("artifacts") or {})
        if artifact_types:
            artifacts_map = {k: v for k, v in artifacts_map.items() if k in artifact_types}
        if not artifacts_map:
            raise ValueError("No artifacts available to export")

        snapshot = {**snapshot, "artifacts": artifacts_map}

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        fmt = export.export_format.value
        suffix = artifact_types[0] if artifact_types and len(artifact_types) == 1 else "bundle"
        out_path = EXPORT_DIR / f"{plan.id}_{suffix}_{export.id}.{fmt}"

        if fmt == "docx":
            _export_docx(out_path, plan, snapshot)
        elif fmt == "pdf":
            _export_pdf(out_path, plan, snapshot)
        elif fmt == "pptx":
            _export_pptx(out_path, plan, snapshot)
        else:
            raise ValueError(f"Unsupported export format: {fmt}")

        export.file_path = str(out_path)
        export.status = ExportStatus.COMPLETED
        export.updated_at = datetime.now(UTC)
        db.commit()
        return {"ok": True, "file_path": str(out_path)}
    except Exception as exc:
        logger.exception("Export failed: %s", export_id)
        db.rollback()
        export = db.get(LessonExport, int(export_id))
        if export:
            export.status = ExportStatus.FAILED
            export.error_message = str(exc)
            db.commit()
        return {"ok": False, "error": str(exc)}
    finally:
        db.close()


def _export_docx(path: Path, plan: LessonPlan, snapshot: dict) -> None:
    from app.services.lesson_planner.export.markdown_docx import export_docx_document
    from app.services.lesson_planner.export.markdown_pdf import ARTIFACT_TITLES

    sections = _artifact_sections(snapshot)
    artifacts = snapshot.get("artifacts") or {}
    single_key = next(iter(artifacts.keys()), None) if len(artifacts) == 1 else None
    title = ARTIFACT_TITLES.get(single_key, plan.title) if single_key else plan.title
    export_docx_document(
        path,
        title=title,
        subtitle=f"Subject: {plan.subject} | Grade: {plan.grade} | Chapter: {plan.chapter_name}",
        sections=sections,
    )


def _artifact_sections(snapshot: dict) -> list[tuple[str, dict]]:
    sections: list[tuple[str, dict]] = []
    for key, content in (snapshot.get("artifacts") or {}).items():
        if content:
            sections.append((key, content))
    return sections


def _export_pdf(path: Path, plan: LessonPlan, snapshot: dict) -> None:
    from app.services.lesson_planner.export.markdown_pdf import ARTIFACT_TITLES, export_pdf_document

    artifacts = snapshot.get("artifacts") or {}
    single_key = next(iter(artifacts.keys()), None) if len(artifacts) == 1 else None
    title = ARTIFACT_TITLES.get(single_key, plan.title) if single_key else plan.title
    export_pdf_document(
        path,
        title=title,
        subtitle=f"Subject: {plan.subject} | Grade: {plan.grade} | Chapter: {plan.chapter_name}",
        sections=_artifact_sections(snapshot),
    )


def _export_pptx(path: Path, plan: LessonPlan, snapshot: dict) -> None:
    from app.services.lesson_planner.export.deck_schema import deck_from_ppt_artifact
    from app.services.lesson_planner.export.figure_assets import attach_figures_to_slides
    from app.services.lesson_planner.export.pptx_builder import build_classroom_pptx

    ppt = (snapshot.get("artifacts") or {}).get("ppt_outline") or {}
    if isinstance(ppt, dict):
        ppt = dict(ppt)
        meta_plan = plan.plan_metadata if isinstance(plan.plan_metadata, dict) else {}
        if not ppt.get("slide_count_target") and meta_plan.get("ppt_slide_count"):
            ppt["slide_count_target"] = meta_plan.get("ppt_slide_count")
        if not ppt.get("chapter_name"):
            ppt["chapter_name"] = plan.chapter_name
        if not ppt.get("subject"):
            ppt["subject"] = plan.subject
    slides = deck_from_ppt_artifact(
        ppt if isinstance(ppt, dict) else {},
        chapter=plan.chapter_name,
        subject=plan.subject,
    )
    meta_plan = plan.plan_metadata if isinstance(plan.plan_metadata, dict) else {}
    figures = []
    if isinstance(ppt, dict):
        figures = list(ppt.get("figures") or [])
    if not figures:
        figures = list(meta_plan.get("figures") or [])
    slides = attach_figures_to_slides(slides, figures)
    theme_id = (
        (ppt.get("template_id") if isinstance(ppt, dict) else None)
        or meta_plan.get("ppt_template")
        or "clean_academic"
    )
    grade = str(plan.grade or "").replace("CLASS_", "Grade ")
    meta = {
        "chapter": plan.chapter_name or plan.title or "Lesson",
        "subject": plan.subject or "",
        "grade": grade,
        "eyebrow": f"{plan.subject or 'Lesson'} · Classroom Deck",
    }

    if not slides:
        synthetic = [
            {
                "title": meta["chapter"],
                "layout": "title",
                "bullets": [],
                "side_heading": "",
                "callout": "",
                "speaker_notes": "",
            }
        ]
        for key, content in (snapshot.get("artifacts") or {}).items():
            if key == "ppt_outline" or not content:
                continue
            lines = render_artifact_lines(key, content)[:5]
            synthetic.append(
                {
                    "title": key.replace("_", " ").title(),
                    "layout": "bullets",
                    "side_heading": "From lesson pack",
                    "bullets": lines,
                    "right_bullets": [],
                    "callout": "",
                    "speaker_notes": "",
                }
            )
        slides = synthetic

    build_classroom_pptx(path, slides=slides, theme_id=str(theme_id), meta=meta)
