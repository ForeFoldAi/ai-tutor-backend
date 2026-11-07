"""LLM final pick: chapter image candidates → best-fit for question + answer.

Fail-closed: timeout / parse / low confidence → [].
``select_images_for_qa_status`` also reports whether the LLM actually decided
(so callers can trust a deliberate "none fits" but fail open on errors).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM = """\
You pick textbook figures that ILLUSTRATE the tutor's answer to the student's question.
Only choose from the numbered candidates. Prefer none over a weak match.
A figure fits if what it SHOWS (caption / visual / context) is the main subject of the answer —
the same empire, person, place, object, process or concept (e.g. a map of the empire being
discussed, a photo of the monument named, a diagram of the process explained). It need not
show every detail. It does NOT fit if it only shares the chapter/page or shows a different subject.
If the student asks for a kind of visual (map, diagram, graph, photo, timeline), choose only
candidates that are that kind — e.g. a timeline or a painting is not a map.
"seen" (when present) is what the image was observed to show by looking at its pixels — trust it
over caption/title/context, which are sometimes paired with the wrong image. A candidate whose
"seen" is a text box, keyword list, exercise or a different subject does not fit.
"visual" (when present) is an auto-generated description; trust it over "kind".
Return ONLY JSON: {"ids":["0","2"],"confidence":0.0}
ids = candidate index strings (0-based), best first. confidence 0-1 for the whole set.
If nothing clearly fits, return {"ids":[],"confidence":0.0}.
"""

_CONTEXT_NOISE_RE = re.compile(r"^\s*(?:chapter\b[^\n]*|CHAPTER)\s*$", re.I | re.M)
# A "figure" the vision model sees as text is a mis-extracted page fragment; captions on such
# crops often name the topic, so the text-only pick would keep them.
_SEEN_TEXT_ONLY_RE = re.compile(
    r"\b(?:text\s*box|keywords?|word\s*list|list\s+of\s+(?:words|terms)|glossary|"
    r"worksheet|exercises?|questions?|table|text\s+only|paragraph)\b",
    re.I,
)


def _seen_as_text_only(c: dict) -> bool:
    if (c.get("content_kind") or "figure") != "figure":
        return False
    seen = c.get("_seen") or ""
    m = re.search(r"type\s*:\s*([^;,]+(?:\+[^;,]+)?)", seen, re.I)
    kind = m.group(1) if m else seen[:60]
    # "diagram + text box" still shows a picture.
    if re.search(r"\b(diagram|photo|map|illustration|painting|drawing|sculpture)\b", kind, re.I):
        return False
    return bool(_SEEN_TEXT_ONLY_RE.search(kind))


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


def _clip(text: str | None, n: int) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()[:n]


def _candidate_lines(candidates: list[dict]) -> str:
    lines: list[str] = []
    for i, c in enumerate(candidates):
        fig = c.get("figure_number") or ""
        cap = _clip(c.get("caption") or c.get("title"), 180)
        kind = c.get("content_kind") or ""
        page = c.get("page")
        line = f'{i}: figure={fig!r} page={page} kind={kind!r}'
        seen = _clip(c.get("_seen"), 220)
        if seen:
            line += f" seen={seen!r}"
        line += f" caption={cap!r}"
        title = _clip(c.get("title"), 120)
        if title and title.lower() not in cap.lower():
            line += f" title={title!r}"
        visual = _clip(c.get("visual_description"), 160)
        if visual:
            line += f" visual={visual!r}"
        ctx = _clip(_CONTEXT_NOISE_RE.sub("", c.get("figure_context") or ""), 160)
        if ctx:
            line += f" context={ctx!r}"
        lines.append(line)
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
    selected, _decided = await select_images_for_qa_status(
        question,
        answer,
        candidates,
        max_images=max_images,
        min_confidence=min_confidence,
        timeout_sec=timeout_sec,
        candidate_cap=candidate_cap,
    )
    return selected


async def select_images_for_qa_status(
    question: str,
    answer: str,
    candidates: list[dict],
    *,
    max_images: int | None = None,
    min_confidence: float | None = None,
    timeout_sec: float | None = None,
    candidate_cap: int | None = None,
) -> tuple[list[dict], bool]:
    """Return (selected, decided). decided=False means the LLM never gave a usable verdict
    (no candidates, timeout, error, unparseable reply)."""
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
        return [], False

    q = (question or "").strip()[:800]
    a = (answer or "").strip()[:1500]
    if not q and not a:
        logger.info("llm_image_select fail=empty_qa candidates=%d selected=0", len(pool))
        return [], False

    t0 = time.monotonic()
    mode = "text"
    seen = await _vision_descriptions(pool)
    if seen:
        mode = "vision+text"
        pool = [dict(c, _seen=seen[i]) if i in seen else c for i, c in enumerate(pool)]
        kept = [c for c in pool if not _seen_as_text_only(c)]
        if len(kept) < len(pool):
            logger.info(
                "llm_image_select dropped_text_only=%s",
                [c.get("figure_number") or c.get("file_name") for c in pool if _seen_as_text_only(c)],
            )
        if not kept:
            return [], True
        pool = kept

    user = (
        f"Student question:\n{q or '(none)'}\n\n"
        f"Tutor answer:\n{a or '(none)'}\n\n"
        f"Candidates:\n{_candidate_lines(pool)}\n"
    )
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
        return [], False

    parsed = _parse_select_json(raw)
    if parsed is None:
        logger.info(
            "llm_image_select fail=parse candidates=%d selected=0 latency_ms=%d raw=%r",
            len(pool),
            int((time.monotonic() - t0) * 1000),
            (raw or "")[:200],
        )
        return [], False

    ids, confidence = parsed
    if confidence < float(min_confidence):
        logger.info(
            "llm_image_select fail=low_conf mode=%s candidates=%d selected=0 confidence=%.2f latency_ms=%d",
            mode,
            len(pool),
            confidence,
            int((time.monotonic() - t0) * 1000),
        )
        return [], True

    selected = [
        {k: v for k, v in s.items() if k != "_seen"}
        for s in apply_select_result(pool, ids, max_images=int(max_images))
    ]
    figs = [s.get("figure_number") or s.get("file_name") for s in selected]
    logger.info(
        "llm_image_select fail=%s mode=%s candidates=%d selected=%d confidence=%.2f figs=%s latency_ms=%d",
        "empty" if not selected else "ok",
        mode,
        len(pool),
        len(selected),
        confidence,
        figs,
        int((time.monotonic() - t0) * 1000),
    )
    return selected, True


_DESCRIBE_SYSTEM = """\
You look at textbook images and say what each one actually shows.
For every image give at most 20 words: its type (map, photo, painting, diagram, chart,
timeline, table, text box, etc.) and its subject. Describe only what is visible.
Return ONLY JSON mapping image number to description, e.g. {"0":"...","1":"..."}.
"""
_THUMB_PX = 448


def _vision_pool_size() -> int:
    from app.config import LLM_IMAGE_SELECT_VISION_MAX

    return max(1, LLM_IMAGE_SELECT_VISION_MAX)


def _candidate_thumbs(candidates: list[dict]) -> dict[int, str]:
    """index → JPEG data URL for candidates whose stored asset can be read."""
    import base64
    from io import BytesIO

    from PIL import Image

    from app.core.database import SessionLocal
    from app.modules.catalog.models import TextbookUpload
    from app.services.image_service.textbook_image_extraction import image_disk_path

    out: dict[int, str] = {}
    uploads: dict[int, TextbookUpload | None] = {}
    with SessionLocal() as db:
        for i, c in enumerate(candidates):
            uid, fn = c.get("textbook_upload_id"), c.get("file_name")
            if not uid or not fn:
                continue
            try:
                uid = int(uid)
                if uid not in uploads:
                    uploads[uid] = db.get(TextbookUpload, uid)
                path = image_disk_path(uid, fn, upload=uploads[uid])
                with Image.open(path) as im:
                    rgb = im.convert("RGB")
                    rgb.thumbnail((_THUMB_PX, _THUMB_PX))
                    buf = BytesIO()
                    rgb.save(buf, format="JPEG", quality=80)
                out[i] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as exc:
                logger.debug("llm_image_select thumb failed %s/%s: %s", uid, fn, exc)
    return out


def _flatten_description(v: Any) -> str:
    """Vision models return strings or {"type": ..., "subject": ...} / lists — flatten to text."""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (list, tuple)):
        return ", ".join(p for p in (_flatten_description(x) for x in v) if p)
    if isinstance(v, dict):
        return "; ".join(
            f"{k}: {t}" for k, t in ((k, _flatten_description(x)) for k, x in v.items()) if t
        )
    return str(v).strip() if v is not None else ""


_DESC_LINE_RE = re.compile(r'^\s*"?(?:image\s*)?(\d+)"?\s*:\s*(.+?),?\s*$', re.I)


def _parse_description_lines(text: str, n: int) -> dict[int, str]:
    """Fallback for malformed JSON such as ``"0": "{"Map": "India in 1600"}",``."""
    out: dict[int, str] = {}
    for line in text.splitlines():
        m = _DESC_LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        desc = re.sub(r'[{}"]+', " ", m.group(2))
        desc = re.sub(r"\s+", " ", desc).strip(" ,")
        if 0 <= idx < n and desc:
            out[idx] = desc
    return out


def _parse_descriptions(raw: str, n: int) -> dict[int, str]:
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return _parse_description_lines(text, n)
    try:
        data = json.loads(text[start : end + 1], strict=False)
    except (json.JSONDecodeError, ValueError, TypeError):
        return _parse_description_lines(text, n)
    out: dict[int, str] = {}
    for k, v in (data.items() if isinstance(data, dict) else ()):
        m = re.search(r"\d+", str(k))
        if not m:
            continue
        idx, desc = int(m.group()), _flatten_description(v)
        if 0 <= idx < n and desc:
            out[idx] = desc
    return out


async def _vision_descriptions(pool: list[dict]) -> dict[int, str]:
    """index → what the image shows. Uses the stored ``visual_summary`` when present and only
    sends the remaining thumbnails to the vision model (results are saved for next time).
    {} when disabled and nothing is stored — selection then runs text-only."""
    import asyncio

    from app.config import LLM_IMAGE_SELECT_VISION

    vpool = pool[: _vision_pool_size()]
    seen = {i: c["visual_summary"] for i, c in enumerate(vpool) if c.get("visual_summary")}
    missing = [i for i in range(len(vpool)) if i not in seen]
    new: dict[int, str] = {}
    if missing and LLM_IMAGE_SELECT_VISION:
        fresh = await describe_with_vision([vpool[i] for i in missing])
        new = {missing[j]: d for j, d in fresh.items()}
        seen.update(new)
        if new:
            asyncio.get_running_loop().run_in_executor(
                None, save_visual_summaries, [(vpool[i], d) for i, d in new.items()]
            )
    logger.info(
        "llm_image_select seen stored=%d fresh=%d missing=%d",
        len(vpool) - len(missing),
        len(new),
        len(missing),
    )
    return seen


def save_visual_summaries(items: list[tuple[dict, str]]) -> None:
    """Persist descriptions and re-derive image_type / educational_role from them."""
    from sqlalchemy import select

    from app.core.database import SessionLocal
    from app.modules.catalog.models import TextbookImage
    from app.services.image_service.textbook_image_extraction import (
        classify_educational_role,
        classify_image_type,
    )

    try:
        with SessionLocal() as db:
            for c, desc in items:
                uid, fn = c.get("textbook_upload_id"), c.get("file_name")
                if not (uid and fn):
                    continue
                for row in db.scalars(
                    select(TextbookImage).where(
                        TextbookImage.textbook_upload_id == int(uid), TextbookImage.file_name == fn
                    )
                ):
                    row.visual_summary = desc[:255]
                    if (row.content_kind or "figure") == "figure":
                        row.image_type = classify_image_type(
                            row.caption or "", row.page_text_snippet or "", row.visual_summary
                        )
                        if row.educational_role not in ("decorative", "activity"):
                            row.educational_role = classify_educational_role(
                                row.image_type, row.caption or row.visual_summary
                            )
            db.commit()
    except Exception as exc:
        logger.warning("llm_image_select save visual_summary failed: %s", exc)


async def describe_with_vision(vpool: list[dict], *, timeout_sec: float | None = None) -> dict[int, str]:
    """Vision model describes candidate thumbnails (captions withheld so it can't echo them).
    {} when nothing is readable or the call fails."""
    import asyncio

    from app.config import LLM_IMAGE_SELECT_VISION_TIMEOUT_SEC
    from app.services.llm_client import complete

    if not vpool:
        return {}
    budget = timeout_sec or LLM_IMAGE_SELECT_VISION_TIMEOUT_SEC
    t0 = time.monotonic()
    try:
        thumbs = await asyncio.wait_for(asyncio.to_thread(_candidate_thumbs, vpool), timeout=budget)
        if not thumbs:
            return {}
        content: list[dict[str, Any]] = []
        for i in sorted(thumbs):
            content.append({"type": "text", "text": f"Image {i}:"})
            content.append({"type": "image_url", "image_url": {"url": thumbs[i]}})
        remaining = budget - (time.monotonic() - t0)
        if remaining <= 1:
            return {}
        raw = await asyncio.wait_for(
            complete(
                [
                    {"role": "system", "content": _DESCRIBE_SYSTEM},
                    {"role": "user", "content": content},
                ],
                feature="vision",
                max_tokens=80 * len(thumbs),
                temperature=0.0,
                timeout=remaining,
            ),
            timeout=remaining,
        )
    except Exception as exc:
        logger.info(
            "llm_image_select vision_fail err=%s latency_ms=%d — text-only selection",
            type(exc).__name__,
            int((time.monotonic() - t0) * 1000),
        )
        return {}
    seen = {i: d[:255] for i, d in _parse_descriptions(raw, len(vpool)).items()}
    if not seen:
        logger.info("llm_image_select vision_unparsed raw=%r", (raw or "")[:300])
    logger.info(
        "llm_image_select vision_seen=%d/%d latency_ms=%d",
        len(seen),
        len(vpool),
        int((time.monotonic() - t0) * 1000),
    )
    return seen


def backfill_visual_summaries(upload_id: int, *, batch: int = 8, overwrite: bool = False) -> int:
    """Describe every figure of an upload with the vision model and store ``visual_summary``.
    Runs once per extraction so chat requests never wait on the vision model."""
    import asyncio

    from sqlalchemy import select

    from app.core.database import SessionLocal
    from app.modules.catalog.models import TextbookImage

    with SessionLocal() as db:
        q = select(TextbookImage.file_name).where(
            TextbookImage.textbook_upload_id == upload_id,
            TextbookImage.content_kind == "figure",
        )
        if not overwrite:
            q = q.where(TextbookImage.visual_summary.is_(None))
        names = list(db.scalars(q.order_by(TextbookImage.page_index, TextbookImage.sequence)))
    cands = [{"textbook_upload_id": upload_id, "file_name": n} for n in names]

    async def _run() -> int:
        done = 0
        for k in range(0, len(cands), batch):
            chunk = cands[k : k + batch]
            got = await describe_with_vision(chunk, timeout_sec=45.0)
            if got:
                save_visual_summaries([(chunk[i], d) for i, d in got.items()])
                done += len(got)
        return done

    n = asyncio.run(_run()) if cands else 0
    logger.info("visual_summary backfill upload=%s described=%d/%d", upload_id, n, len(cands))
    return n
