"""
Student-facing figure captions from the vision LLM (any board, class or subject).

One figure per call (batching several images in one request let descriptions drift onto
the wrong image). The model sees the figure plus what the textbook says about it: printed
caption, surrounding text, chapter/section, class. It returns JSON:

    {"is_teaching_figure": true,
     "kind": "diagram",
     "title": "Basic proportionality theorem",
     "caption": "Triangle ABC with line DE drawn parallel to side BC, cutting AB at D and AC at E.",
     "labels_seen": ["A", "B", "C", "D", "E"]}

Validation rejects empty/boilerplate output and captions that name diagram points or
measurements the model did not report seeing and that the textbook text does not mention
(the usual hallucination). Failures return None so callers keep the printed caption.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.modules.catalog.models import TextbookImage, TextbookUpload

logger = logging.getLogger(__name__)

KINDS = (
    "diagram", "graph", "map", "photo", "illustration", "cartoon", "chart", "flowchart",
    "table", "painting", "qr_code", "decoration", "text",
)
NON_TEACHING_KINDS = frozenset({"qr_code", "decoration", "text"})

SYSTEM_PROMPT = """You write captions for figures in school textbooks. Students read your caption next to the figure.

Look at the image and use what the textbook says about it. Reply with ONLY a JSON object:
{"is_teaching_figure": true|false,
 "kind": one of %(kinds)s,
 "title": "3-8 words naming what the figure shows",
 "caption": "1-2 sentences, at most 40 words",
 "labels_seen": ["every letter, number, value or word label you can read in the image"]}

Caption rules:
- Say what the figure shows and what the student should notice, in simple words for %(audience)s.
- Never contradict the printed caption. Keep its names, places and dates.
- Diagrams, graphs and geometry: name the shape or setup and the labelled points, angles and values you can read. Do not invent any label or number.
- Photos, paintings and story illustrations: describe the scene and connect it to the chapter.
- Do not start with "This image", "The figure shows a picture" or similar filler. Do not mention the textbook, page or figure number.

is_teaching_figure is false for QR codes, logos, mascots or cartoon characters that only decorate the page, page ornaments, icons, and boxes that contain only text."""

_FILLER_RE = re.compile(r"(?i)^(this|the)\s+(image|picture|figure|illustration)\s+(shows|depicts|is)\s+(an?\s+)?(image|picture)?\s*(of)?\s*")
_BOILERPLATE_RE = re.compile(r"(?i)textbook figure|as an ai|i cannot|i can't|unable to (see|view)")
# Diagram point names: "A", "B'", "ABC", "PQ" (each capital is one point) and
# measurements ("5 cm", "30°") a caption may cite.
_POINT_RUN_RE = re.compile(r"(?<![A-Za-z])([A-Z]{1,4})(?:['′₁₂₃₄]|\d)?(?![a-z])")
_MEASURE_RE = re.compile(r"(?i)\b(\d+(?:\.\d+)?)\s*(cm|mm|m|km|°|degrees?|kg|g|s|v|a|Ω|ohm)\b")


@dataclass
class FigureContext:
    subject: str = ""
    class_level: str = ""
    chapter: str = ""
    section: str = ""
    figure_number: str = ""
    printed_caption: str = ""
    text_before: str = ""
    text_after: str = ""


@dataclass
class FigureCaption:
    is_teaching_figure: bool
    kind: str
    title: str
    caption: str
    labels_seen: list[str] = field(default_factory=list)


def audience(class_level: str) -> str:
    m = re.search(r"(\d+)", class_level or "")
    return f"a Class {m.group(1)} student" if m else "a school student"


def image_data_url(image_bytes: bytes, max_px: int) -> str | None:
    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as im:
            rgb = im.convert("RGB")
            rgb.thumbnail((max_px, max_px))
            buf = BytesIO()
            rgb.save(buf, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _clip(text: str, n: int) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def build_messages(image_url: str, ctx: FigureContext) -> list[dict[str, Any]]:
    system = SYSTEM_PROMPT % {"kinds": ", ".join(KINDS), "audience": audience(ctx.class_level)}
    facts = [
        f"Subject: {ctx.subject}" if ctx.subject else "",
        f"Class: {ctx.class_level}" if ctx.class_level else "",
        f"Chapter: {ctx.chapter}" if ctx.chapter else "",
        f"Section: {ctx.section}" if ctx.section else "",
        f"Figure number: {ctx.figure_number}" if ctx.figure_number else "",
        f"Printed caption: {_clip(ctx.printed_caption, 400)}" if ctx.printed_caption else "Printed caption: (none)",
        f"Text before the figure: {_clip(ctx.text_before, 600)}" if ctx.text_before else "",
        f"Text after the figure: {_clip(ctx.text_after, 600)}" if ctx.text_after else "",
    ]
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "\n".join(f for f in facts if f)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]


def parse_response(raw: str) -> FigureCaption | None:
    text = re.sub(r"(?s)<think>.*?</think>", "", raw or "").strip()
    start = text.find("{")
    if start < 0:
        return None
    try:
        # raw_decode stops at the end of the first JSON object, so trailing extra braces or
        # prose after it do not break parsing.
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    caption = re.sub(r"\s+", " ", str(data.get("caption") or "")).strip()
    caption = _FILLER_RE.sub("", caption).strip()
    if caption:
        caption = caption[0].upper() + caption[1:]
    kind = str(data.get("kind") or "").strip().lower().replace(" ", "_")
    labels = data.get("labels_seen") or []
    if not isinstance(labels, list):
        labels = []
    teaching = data.get("is_teaching_figure")
    return FigureCaption(
        is_teaching_figure=teaching if isinstance(teaching, bool) else True,
        kind=kind if kind in KINDS else "illustration",
        title=re.sub(r"\s+", " ", str(data.get("title") or "")).strip().strip(".")[:120],
        caption=caption,
        labels_seen=[str(x).strip() for x in labels if str(x).strip()][:60],
    )


def validation_problems(result: FigureCaption, ctx: FigureContext) -> list[str]:
    """Reasons to reject a caption. Empty list = accept."""
    problems: list[str] = []
    if not result.is_teaching_figure:
        return problems  # caption is irrelevant; caller hides the figure
    cap = result.caption
    if len(cap) < 15:
        problems.append("caption too short")
    if len(cap) > 320:
        problems.append("caption too long")
    if _BOILERPLATE_RE.search(cap):
        problems.append("boilerplate")
    known = " ".join([" ".join(result.labels_seen), ctx.printed_caption, ctx.text_before, ctx.text_after])
    known_points = _points(known)
    # Point names only matter in diagram-like figures; elsewhere capitals start names/sentences.
    if result.kind in ("diagram", "graph", "chart", "flowchart"):
        cap_points = _points(cap) - {"I"}
        if re.match(r"^A\s", cap):
            cap_points.discard("A")  # sentence-initial article
        stray = sorted(cap_points - known_points)
        if stray:
            problems.append(f"names unseen points {stray}")
    known_numbers = set(re.findall(r"\d+(?:\.\d+)?", known))
    for n, u in _MEASURE_RE.findall(cap):
        if n not in known_numbers:
            problems.append(f"invents measurement {n} {u}")
            break
    return problems


# Per-minute provider quotas (Groq/Mistral) return 429; wait them out instead of failing.
_RATE_LIMIT_WAITS_SEC = (5.0, 15.0, 30.0, 60.0)


def _rate_limit_wait(exc: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying *exc*, or None when it should not be retried."""
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if status not in (429, 503) or attempt >= len(_RATE_LIMIT_WAITS_SEC):
        return None
    try:
        header = float(resp.headers.get("retry-after") or 0)
    except (TypeError, ValueError):
        header = 0.0
    return min(120.0, max(header, _RATE_LIMIT_WAITS_SEC[attempt]))


def _points(text: str) -> set[str]:
    return {ch for run in _POINT_RUN_RE.findall(text or "") for ch in run}


def caption_figure(image_bytes: bytes, ctx: FigureContext) -> FigureCaption | None:
    """Ask the vision model for a caption. None when unavailable, unparseable or rejected."""
    from app.config import FIGURE_CAPTION_VLM_MAX_PX, FIGURE_CAPTION_VLM_TIMEOUT_SEC
    from app.services.llm_client import complete_sync

    url = image_data_url(image_bytes, FIGURE_CAPTION_VLM_MAX_PX)
    if not url:
        return None
    messages = build_messages(url, ctx)
    raw = ""
    for attempt in range(len(_RATE_LIMIT_WAITS_SEC) + 1):
        try:
            raw = complete_sync(
                messages,
                feature="vision",
                max_tokens=400,
                timeout=FIGURE_CAPTION_VLM_TIMEOUT_SEC,
            )
            break
        except Exception as exc:
            wait = _rate_limit_wait(exc, attempt)
            if wait is None:
                logger.warning("figure_caption_vlm call failed (%s): %s", ctx.figure_number or "-", exc)
                raise
            logger.info("figure_caption_vlm rate limited; retry in %.0fs", wait)
            time.sleep(wait)
    result = parse_response(raw)
    if result is None:
        logger.info("figure_caption_vlm unparsed: %r", (raw or "")[:200])
        return None
    problems = validation_problems(result, ctx)
    if problems:
        logger.info("figure_caption_vlm rejected (%s): %s | %r", ctx.figure_number or "-", problems, result.caption[:160])
        return None
    return result


def context_for_row(row: "TextbookImage", upload: "TextbookUpload | None") -> FigureContext:
    def _enum(v: Any) -> str:
        return str(getattr(v, "value", v) or "")

    printed = (row.caption or "") if (row.caption_source or "") not in ("vision", "vlm") else ""
    return FigureContext(
        subject=str(getattr(upload, "subject_name", "") or row.subject or ""),
        class_level=_enum(getattr(upload, "class_level", "")) or str(row.grade_level or ""),
        chapter=str(row.chapter_title or getattr(upload, "chapter", "") or ""),
        section=str(row.section_title or ""),
        figure_number=str(row.figure_number or ""),
        printed_caption=printed,
        text_before=row.nearby_text_before_figure or "",
        text_after=row.nearby_text_after_figure or "",
    )


def apply_to_row(row: "TextbookImage", result: FigureCaption) -> None:
    """Store a validated caption. Non-teaching *unnumbered* figures are hidden from students."""
    if not result.is_teaching_figure or result.kind in NON_TEACHING_KINDS:
        tags = [t for t in (row.educational_tags or "").split("|")
                if t and not t.startswith("junk:") and t != "needs_recrop"]
        if (row.figure_number or "").strip():
            # A numbered figure the model sees as text/decoration is a bad crop, not junk:
            # keep it visible with its printed caption and flag it for re-extraction.
            row.educational_tags = "|".join(["needs_recrop", *tags])[:256]
            return
        row.is_decorative = True
        row.educational_role = "decorative"
        row.educational_tags = "|".join(["junk:vlm_not_teaching", *tags])[:256]
        return
    row.generated_caption = result.caption
    row.caption_source = "vlm"
    if result.title:
        row.title = result.title[:255]
    row.visual_summary = f"type: {result.kind}; subject: {result.title or result.caption}"[:255]
    labels = ", ".join(result.labels_seen[:20])
    row.figure_context = "\n".join(
        p for p in (row.figure_context or "", f"Figure shows: {result.caption}",
                    f"Labels: {labels}" if labels else "") if p
    )[:8000]
    row.context_bge_indexed = False
    row.multimodal_indexed = False


def caption_upload_figures(
    upload_id: int,
    *,
    force: bool = False,
    limit: int | None = None,
    on_result: Any = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Caption every visible figure of an upload. Commits after each figure (resumable).

    Skips figures already captioned by the VLM unless *force*. *on_result(row, result)* is
    called for each figure (result None when the model gave nothing usable). With
    *dry_run* nothing is written.
    """
    from sqlalchemy import select

    from app.config import FIGURE_CAPTION_VLM_ENABLED
    from app.core.database import SessionLocal
    from app.modules.catalog.models import TextbookImage, TextbookUpload
    from app.services.image_service.storage_backend import get_storage_backend
    from app.services.image_service.textbook_image_extraction import image_storage_key

    stats = {"captioned": 0, "hidden": 0, "needs_recrop": 0, "rejected": 0, "failed": 0, "skipped": 0}
    if not FIGURE_CAPTION_VLM_ENABLED:
        return stats
    backend = get_storage_backend()
    with SessionLocal() as db:
        upload = db.get(TextbookUpload, upload_id)
        if upload is None:
            return stats
        q = (
            select(TextbookImage)
            .where(
                TextbookImage.textbook_upload_id == upload_id,
                TextbookImage.content_kind == "figure",
                TextbookImage.is_decorative.is_(False),
            )
            .order_by(TextbookImage.page_index, TextbookImage.sequence)
        )
        rows = list(db.scalars(q))
        done = 0
        for row in rows:
            if limit is not None and done >= limit:
                break
            if not force and row.caption_source == "vlm":
                stats["skipped"] += 1
                continue
            data = backend.get_bytes(image_storage_key(upload, row.file_name))
            if not data:
                stats["failed"] += 1
                continue
            done += 1
            try:
                result = caption_figure(data, context_for_row(row, upload))
            except Exception:
                stats["failed"] += 1
                if on_result:
                    on_result(row, None)
                continue
            if on_result:
                on_result(row, result)
            if result is None:
                stats["rejected"] += 1
                continue
            if dry_run:
                db.expunge(row)  # never flush the in-memory change
            apply_to_row(row, result)
            if row.is_decorative:
                stats["hidden"] += 1
            elif row.caption_source == "vlm":
                stats["captioned"] += 1
            else:
                stats["needs_recrop"] += 1
            if not dry_run:
                db.commit()
    logger.info("figure_caption_vlm upload=%s %s", upload_id, stats)
    return stats
