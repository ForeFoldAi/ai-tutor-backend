"""Shared image understanding orchestration for chapter chat, Ask AI, and voice."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.config import (
    IMAGE_MAX_IMAGES_PER_REQUEST,
    IMAGE_TEMP_DIR,
    IMAGE_TEMP_TTL_SEC,
    TESSERACT_MAX_LOW_CONF_RATIO,
    TESSERACT_MIN_CONFIDENCE,
    TESSERACT_MIN_WORDS,
)
from app.modules.image_understanding.cache import get_cached_analysis, set_cached_analysis
from app.modules.image_understanding.errors import (
    ImageNotFoundError,
    InvalidImageError,
    VisionProviderError,
)
from app.modules.image_understanding.intent import detect_intent
from app.modules.image_understanding.ocr.schemas import OCRResult
from app.modules.image_understanding.ocr.service import run_ocr
from app.modules.image_understanding.preprocessing import preprocess_image
from app.modules.image_understanding.prompts import build_retrieval_query, build_tutor_prompt_block
from app.modules.image_understanding.quality import assess_quality
from app.modules.image_understanding.routing import Task, decide
from app.modules.image_understanding.schemas import (
    DetectedQuestion,
    ImageUnderstandingResult,
    UnderstandingBundle,
)
from app.modules.image_understanding.validators import ValidatedImage, validate_image_bytes
from app.modules.image_understanding.vision import MALFORMED_OUTPUT, get_vision_provider, vision_model_name

logger = logging.getLogger(__name__)

_REDIS_BYTES_PREFIX = "tutor_img:bytes:"
_REDIS_META_PREFIX = "tutor_img:meta:"


@dataclass
class StoredImageMeta:
    image_id: str
    user_id: int
    path: str  # local path when disk fallback used; empty when Redis-only
    mime_type: str
    content_hash: str
    width: int
    height: int
    expires_at: float


def _meta_path(image_id: str) -> Path:
    return Path(IMAGE_TEMP_DIR) / f"{image_id}.meta.json"


def _ensure_temp_dir() -> Path:
    root = Path(IMAGE_TEMP_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sync_redis():
    """Shared Redis for multi-worker ephemeral uploads. None if unavailable."""
    try:
        import redis

        from app.core.config import get_settings

        r = redis.from_url(
            get_settings().redis_url,
            socket_connect_timeout=1.0,
            socket_timeout=2.0,
        )
        r.ping()
        return r
    except Exception:
        return None


_SWEEP_EVERY_SEC = 60.0
_last_sweep = 0.0


def _sweep_expired_temp(root: Path) -> None:
    """Delete temp uploads older than the TTL (abandoned uploads are otherwise never loaded, so never expire)."""
    global _last_sweep
    now = time.time()
    if now - _last_sweep < _SWEEP_EVERY_SEC:
        return
    _last_sweep = now
    for f in root.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < now - IMAGE_TEMP_TTL_SEC:
                f.unlink(missing_ok=True)
        except OSError:
            pass


def _valid_image_id(image_id: str) -> bool:
    return bool(image_id) and all(c in "0123456789abcdef" for c in image_id.lower())


def store_validated_image(validated: ValidatedImage, *, user_id: int) -> StoredImageMeta:
    processed, mime, width, height = preprocess_image(validated.data, validated.mime_type)
    content_hash = hashlib.sha256(processed).hexdigest()
    image_id = uuid.uuid4().hex
    expires_at = time.time() + IMAGE_TEMP_TTL_SEC
    ext = "jpg" if mime == "image/jpeg" else ("png" if mime == "image/png" else "webp")

    meta = StoredImageMeta(
        image_id=image_id,
        user_id=int(user_id),
        path="",
        mime_type=mime,
        content_hash=content_hash,
        width=width,
        height=height,
        expires_at=expires_at,
    )
    meta_payload = json.dumps(
        {
            "image_id": meta.image_id,
            "user_id": meta.user_id,
            "path": "",
            "mime_type": meta.mime_type,
            "content_hash": meta.content_hash,
            "width": meta.width,
            "height": meta.height,
            "expires_at": meta.expires_at,
        }
    )

    # Redis first — required for multi-instance production (upload ≠ chat worker).
    r = _sync_redis()
    stored_remote = False
    if r is not None:
        try:
            pipe = r.pipeline()
            pipe.setex(f"{_REDIS_BYTES_PREFIX}{image_id}", IMAGE_TEMP_TTL_SEC, processed)
            pipe.setex(f"{_REDIS_META_PREFIX}{image_id}", IMAGE_TEMP_TTL_SEC, meta_payload)
            pipe.execute()
            stored_remote = True
            r.close()
        except Exception:
            logger.exception("Redis store failed for tutor image id=%s", image_id[:8])
            try:
                r.close()
            except Exception:
                pass

    # Local disk fallback only when Redis failed: Redis keys expire by themselves, disk files need the sweep.
    if not stored_remote:
        root = _ensure_temp_dir()
        _sweep_expired_temp(root)
        path = root / f"{image_id}.{ext}"
        path.write_bytes(processed)
        meta.path = str(path)
        disk_meta = json.loads(meta_payload)
        disk_meta["path"] = meta.path
        _meta_path(image_id).write_text(json.dumps(disk_meta), encoding="utf-8")
        logger.warning(
            "Tutor image %s stored on local disk only — multi-worker production needs Redis "
            "(REDIS_URL). Chat on another instance will return 'image no longer available'.",
            image_id[:8],
        )
    return meta


def _meta_from_dict(data: dict) -> StoredImageMeta:
    return StoredImageMeta(
        image_id=str(data["image_id"]),
        user_id=int(data["user_id"]),
        path=str(data.get("path") or ""),
        mime_type=str(data["mime_type"]),
        content_hash=str(data["content_hash"]),
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
        expires_at=float(data["expires_at"]),
    )


def _load_from_redis(image_id: str) -> tuple[bytes, StoredImageMeta] | None:
    r = _sync_redis()
    if r is None:
        return None
    try:
        raw_meta = r.get(f"{_REDIS_META_PREFIX}{image_id}")
        raw_bytes = r.get(f"{_REDIS_BYTES_PREFIX}{image_id}")
        r.close()
    except Exception:
        logger.exception("Redis load failed for tutor image id=%s", image_id[:8])
        try:
            r.close()
        except Exception:
            pass
        return None
    if not raw_meta or not raw_bytes:
        return None
    try:
        if isinstance(raw_meta, bytes):
            raw_meta = raw_meta.decode("utf-8")
        meta = _meta_from_dict(json.loads(raw_meta))
    except Exception as exc:
        raise ImageNotFoundError() from exc
    if meta.expires_at < time.time():
        delete_image(image_id)
        raise ImageNotFoundError()
    if isinstance(raw_bytes, str):
        raw_bytes = raw_bytes.encode("latin-1")
    return raw_bytes, meta


def _load_from_disk(image_id: str) -> tuple[bytes, StoredImageMeta]:
    mp = _meta_path(image_id)
    if not mp.is_file():
        raise ImageNotFoundError()
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ImageNotFoundError() from exc
    meta = _meta_from_dict(data)
    if meta.expires_at < time.time():
        delete_image(image_id)
        raise ImageNotFoundError()
    root = _ensure_temp_dir().resolve()
    try:
        resolved = Path(meta.path).resolve()
        if not str(resolved).startswith(str(root)) or not resolved.is_file():
            raise ImageNotFoundError()
    except ImageNotFoundError:
        raise
    except Exception as exc:
        raise ImageNotFoundError() from exc
    return resolved.read_bytes(), meta


def load_image_for_user(image_id: str, *, user_id: int) -> tuple[bytes, StoredImageMeta]:
    if not _valid_image_id(image_id):
        raise ImageNotFoundError()
    loaded = _load_from_redis(image_id)
    if loaded is None:
        loaded = _load_from_disk(image_id)
    data, meta = loaded
    if meta.user_id != int(user_id):
        raise ImageNotFoundError()
    return data, meta


def delete_image(image_id: str) -> None:
    if not _valid_image_id(image_id):
        return
    r = _sync_redis()
    if r is not None:
        try:
            r.delete(f"{_REDIS_BYTES_PREFIX}{image_id}", f"{_REDIS_META_PREFIX}{image_id}")
            r.close()
        except Exception:
            try:
                r.close()
            except Exception:
                pass
    try:
        mp = _meta_path(image_id)
        if mp.is_file():
            try:
                data = json.loads(mp.read_text(encoding="utf-8"))
                p = data.get("path")
                if p and os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
            mp.unlink(missing_ok=True)
    except Exception:
        logger.debug("temp image cleanup failed for id=%s", image_id[:8] if image_id else "?")


def _math_block_from_result(result, class_level: str) -> str:
    if not result.mathematical_content.detected:
        return ""
    exprs = [e.strip() for e in result.mathematical_content.expressions if e and e.strip()]
    qtexts = [q.text.strip() for q in result.questions if q.text and q.text.strip()]
    candidates = exprs + qtexts
    if not candidates:
        return ""
    try:
        from app.services.math_engine import try_solve

        for cand in candidates[:3]:
            eng = try_solve(cand, class_level=class_level or "")
            if eng and eng.solved:
                return eng.to_prompt_block()
    except Exception:
        logger.exception("math_engine on image expression failed")
    return ""


_OCR_TEXT_CAP = 1500
_QUESTION_LINE = re.compile(r"\?\s*$|^\s*(Q\.?\s*\d+[.):]?|\d{1,2}[.)]|\([a-z0-9]{1,3}\))\s", re.I)
_LANG_NAMES = {"eng": "English", "hin": "Hindi", "tel": "Telugu", "tam": "Tamil", "kan": "Kannada"}
_LANG_SCRIPTS = {"eng": "LATIN", "hin": "DEVANAGARI", "tel": "TELUGU", "tam": "TAMIL", "kan": "KANNADA"}
UNCLEAR_REASON = "Unable to reliably understand the uploaded image."


def _ocr_usable(ocr: OCRResult | None) -> bool:
    return bool(ocr and not ocr.error and ocr.has_text and ocr.word_count >= TESSERACT_MIN_WORDS)


def _ocr_language(ocr: OCRResult) -> str:
    """Configured language, or — for multi-language configs like eng+hin — the dominant script in the text."""
    langs = [p for p in (ocr.language or "eng").split("+") if p]
    if len(langs) > 1:
        scripts = Counter(unicodedata.name(c, "?").split()[0] for c in ocr.text if c.isalpha())
        if scripts:
            script = scripts.most_common(1)[0][0]
            for code in langs:
                if _LANG_SCRIPTS.get(code) == script:
                    return _LANG_NAMES.get(code, code)
    return _LANG_NAMES.get(langs[0], langs[0])


def _result_from_ocr(ocr: OCRResult, *, subject_name: str, degraded: bool = False) -> ImageUnderstandingResult:
    lines = [ln.strip() for ln in ocr.text.splitlines() if ln.strip()]
    questions = [DetectedQuestion(text=ln[:300]) for ln in lines if _QUESTION_LINE.search(ln)][:8]
    return ImageUnderstandingResult(
        image_type="textbook_question" if questions else "textbook_page",
        language=_ocr_language(ocr),
        detected_subject=subject_name,
        ocr_text=ocr.text[:_OCR_TEXT_CAP],
        content_summary=" ".join(lines)[:200],
        question_detected=bool(questions),
        questions=questions,
        confidence=min(ocr.average_confidence, 0.44) if degraded else ocr.average_confidence,
        unclear_regions=(
            ["only printed text was extracted; diagrams, handwriting or math layout may be missed"]
            if degraded
            else []
        ),
        source="tesseract",
    )


def _unclear_result(reason: str) -> ImageUnderstandingResult:
    return ImageUnderstandingResult(
        status="unclear",
        confidence=0.0,
        requires_clearer_image=True,
        reason=reason,
        source="none",
    )


async def _analyze(
    data: bytes,
    meta: StoredImageMeta,
    query: str,
    *,
    model: str,
    class_level: str,
    subject_name: str,
    board: str,
) -> tuple[ImageUnderstandingResult, str]:
    """Quality → one OCR pass → route → at most one vision call (+ bounded retries). Never loops."""
    t0 = time.perf_counter()
    log: dict = {"image": meta.image_id[:8], "ocr_attempted": False, "vision_attempted": False}

    quality = await asyncio.to_thread(assess_quality, data)
    ocr: OCRResult | None = None
    if quality.is_usable:
        ocr, ocr_hit = await run_ocr(data, meta.content_hash)
        if ocr is not None:
            log.update(
                ocr_attempted=True,
                ocr_cache_hit=ocr_hit,
                ocr_ok=not ocr.error,
                ocr_error=ocr.error,
                ocr_ms=round(ocr.processing_time_ms),
                ocr_confidence=round(ocr.average_confidence, 3),
                ocr_low_conf_ratio=round(ocr.low_confidence_ratio, 3),
                ocr_coverage=round(ocr.text_coverage, 3),
                ocr_words=ocr.word_count,
            )
    decision = decide(ocr, quality, query)
    log.update(route=decision.task.value, route_reason=decision.reason, quality=quality.reason or "ok")

    failure = ""
    if decision.task is Task.UNCLEAR:
        failure = f"quality:{quality.reason}"
        result = _unclear_result(
            "The image looks blank or too small to read." if quality.mostly_blank or quality.too_small else UNCLEAR_REASON
        )
    elif not decision.use_vision and ocr is not None:
        result = _result_from_ocr(ocr, subject_name=subject_name)
    else:
        log["vision_attempted"] = True
        vision = get_cached_analysis(meta.content_hash, model, query)
        log["vision_cache_hit"] = vision is not None
        if vision is None:
            provider = get_vision_provider()
            v0 = time.perf_counter()
            try:
                vision = await provider.analyze_image(
                    data,
                    meta.mime_type,
                    student_message=query,
                    class_level=class_level,
                    subject_name=subject_name,
                    board=board,
                )
            except VisionProviderError:
                vision, failure = None, "vision_error"
            log.update(vision_ms=round((time.perf_counter() - v0) * 1000), vision_attempts=getattr(provider, "attempts", 1))
            if vision is not None and MALFORMED_OUTPUT in vision.unclear_regions:
                vision, failure = None, "vision_malformed"
            if vision is not None:
                set_cached_analysis(meta.content_hash, model, vision, query)

        if vision is not None:
            hybrid = (
                _ocr_usable(ocr)
                and ocr.average_confidence >= TESSERACT_MIN_CONFIDENCE
                and ocr.low_confidence_ratio <= TESSERACT_MAX_LOW_CONF_RATIO
            )
            update: dict = {"source": "hybrid" if hybrid else "vision"}
            if hybrid and not vision.ocr_text.strip():
                update["ocr_text"] = ocr.text[:_OCR_TEXT_CAP]
            result = vision.model_copy(update=update)
        elif _ocr_usable(ocr):
            result = _result_from_ocr(ocr, subject_name=subject_name, degraded=True)
        else:
            result = _unclear_result(UNCLEAR_REASON)

    result.image_quality = quality
    result.ocr_confidence = ocr.average_confidence if ocr is not None and not ocr.error else None
    log.update(
        source=result.source,
        status=result.status,
        confidence=round(result.confidence, 3),
        failure=failure or None,
        total_ms=round((time.perf_counter() - t0) * 1000),
    )
    logger.info("image_understanding %s", json.dumps(log, default=str))
    return result, decision.task.value


async def understand(
    image_ids: list[str] | None,
    query: str,
    *,
    user_id: int,
    class_level: str = "",
    subject_name: str = "",
    board: str = "",
    delete_after: bool = True,
) -> UnderstandingBundle | None:
    """
    Analyze student-uploaded images. Returns None when image_ids empty.
    Raises ImageUnderstandingError subclasses for user-facing failures.
    """
    ids = [i for i in (image_ids or []) if i]
    if not ids:
        return None
    if len(ids) > IMAGE_MAX_IMAGES_PER_REQUEST:
        raise InvalidImageError(
            f"Please upload at most {IMAGE_MAX_IMAGES_PER_REQUEST} image(s) per message."
        )

    # v1: first image only (schema allows list)
    image_id = ids[0]
    # Blocking Redis/disk/Pillow work stays off the event loop (voice websockets share it).
    data, meta = await asyncio.to_thread(load_image_for_user, image_id, user_id=user_id)
    model = vision_model_name()
    result, task = await _analyze(
        data,
        meta,
        query or "",
        model=model,
        class_level=class_level,
        subject_name=subject_name,
        board=board,
    )

    intent = detect_intent(query or "", result)
    low = result.status == "unclear" or result.confidence < 0.45 or bool(result.unclear_regions)
    tutor_block = build_tutor_prompt_block(result, intent=intent, low_confidence=low)
    math_block = _math_block_from_result(result, class_level)
    retrieval = build_retrieval_query(result, query or "")

    if delete_after:
        for iid in ids:
            await asyncio.to_thread(delete_image, iid)

    return UnderstandingBundle(
        result=result,
        intent=intent,
        retrieval_query=retrieval,
        tutor_prompt_block=tutor_block,
        math_prompt_block=math_block,
        low_confidence=low,
        provider_model=model,
        routing_task=task,
    )


async def save_upload(
    raw: bytes,
    *,
    user_id: int,
    filename: str | None = None,
    declared_mime: str | None = None,
) -> StoredImageMeta:
    def _save() -> StoredImageMeta:
        validated = validate_image_bytes(raw, filename=filename, declared_mime=declared_mime)
        return store_validated_image(validated, user_id=user_id)

    # Decode + EXIF strip + re-encode is ~1 s for a phone photo; keep it off the event loop.
    return await asyncio.to_thread(_save)
