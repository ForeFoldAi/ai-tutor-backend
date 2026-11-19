"""
Vision-based captions for textbook figures when PDF text pairing is weak.

Uses BLIP (Salesforce/blip-image-captioning-base) via Transformers to describe
figure pixels, then wraps the result with subject/chapter context for BGE retrieval.
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import TYPE_CHECKING, Any

from app.config import HF_TOKEN, VISION_CAPTION_ENABLED, VISION_CAPTION_MODEL

if TYPE_CHECKING:
    from app.modules.catalog.models import TextbookImage, TextbookUpload

logger = logging.getLogger(__name__)

_blip_model: Any = None
_blip_processor: Any = None
_loaded_model_name: str | None = None

_WEAK_CAPTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)^illustration\s+(showing|of)\s+chapter"),
    re.compile(r"(?i)^chapter:\s*chapter"),
    re.compile(r"(?i)^illustration\s*[—\-]"),
    re.compile(r"(?i)ever-evolving world of science"),
)


def is_weak_text_caption(caption: str | None) -> bool:
    """True when caption is missing, generic, OCR-broken, or chapter boilerplate."""
    from app.services.image_service.figure_context_gates import (
        is_corrupt_ml_caption,
        is_minimal_figure_caption,
    )

    cap = (caption or "").strip()
    if not cap:
        return True
    if is_minimal_figure_caption(cap):
        return True
    if is_corrupt_ml_caption(cap):
        return True
    for pat in _WEAK_CAPTION_PATTERNS:
        if pat.search(cap):
            return True
    # Short captions with many broken tokens (mid-word fragments)
    words = re.findall(r"\b[a-z]{2,}\b", cap.lower())
    if len(words) < 4 and len(cap) > 40:
        return True
    return False


def vision_caption_available() -> bool:
    if not VISION_CAPTION_ENABLED:
        return False
    try:
        return _load_blip() is not None
    except Exception:
        return False


def _load_blip() -> tuple[Any, Any] | None:
    global _blip_model, _blip_processor, _loaded_model_name
    if _blip_model is not None and _blip_processor is not None:
        return _blip_model, _blip_processor
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        name = VISION_CAPTION_MODEL
        logger.info("Loading vision caption model: %s", name)
        kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
        _blip_processor = BlipProcessor.from_pretrained(name, **kwargs)
        _blip_model = BlipForConditionalGeneration.from_pretrained(name, **kwargs)
        _blip_model.eval()
        _loaded_model_name = name
        torch.set_num_threads(min(4, torch.get_num_threads()))
        return _blip_model, _blip_processor
    except Exception as exc:
        logger.warning("Vision caption model unavailable (%s): %s", VISION_CAPTION_MODEL, exc)
        _blip_model = None
        _blip_processor = None
        return None


def _raw_blip_caption(image_bytes: bytes) -> str | None:
    pair = _load_blip()
    if pair is None:
        return None
    model, processor = pair
    try:
        import torch
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as im:
            rgb = im.convert("RGB")
        inputs = processor(images=rgb, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48)
        text = processor.decode(out[0], skip_special_tokens=True).strip()
        return text or None
    except Exception as exc:
        logger.warning("BLIP caption failed: %s", exc)
        return None


def format_educational_vision_caption(
    raw_caption: str,
    *,
    subject: str = "",
    chapter_title: str = "",
    section_title: str = "",
) -> str:
    """Wrap a vision caption with curriculum context for retrieval."""
    raw = re.sub(r"\s+", " ", (raw_caption or "").strip())
    if not raw:
        return ""
    suffix_parts: list[str] = []
    if subject:
        suffix_parts.append(f"{subject.strip()} textbook figure")
    if chapter_title:
        suffix_parts.append(chapter_title.strip())
    if section_title and section_title.lower() not in raw.lower():
        suffix_parts.append(section_title.strip())
    if suffix_parts:
        return f"{raw}. {', '.join(suffix_parts)}."
    return raw


def caption_image_bytes(
    image_bytes: bytes,
    *,
    subject: str = "",
    chapter_title: str = "",
    section_title: str = "",
) -> str | None:
    """Generate an educational caption from figure pixels."""
    if not VISION_CAPTION_ENABLED:
        return None
    raw = _raw_blip_caption(image_bytes)
    if not raw:
        return None
    return format_educational_vision_caption(
        raw,
        subject=subject,
        chapter_title=chapter_title,
        section_title=section_title,
    )


def apply_vision_caption_to_row(
    row: "TextbookImage",
    *,
    image_bytes: bytes,
    upload: "TextbookUpload",
    nearby_before: str = "",
    nearby_after: str = "",
) -> bool:
    """
    Replace weak captions with a vision-generated caption and refresh metadata.

    Returns True when a vision caption was applied.
    """
    from app.services.image_service.textbook_image_extraction import (
        build_figure_context,
        classify_educational_role,
        classify_image_type,
        compute_educational_salience,
        normalize_caption,
    )

    effective = (row.caption or row.generated_caption or "").strip()
    if not is_weak_text_caption(effective):
        return False

    subject = str(getattr(upload, "subject_name", "") or row.subject or "")
    chapter = str(row.chapter_title or getattr(upload, "content_label", "") or "")
    section = str(row.section_title or "")

    vision_cap = caption_image_bytes(
        image_bytes,
        subject=subject,
        chapter_title=chapter,
        section_title=section,
    )
    if not vision_cap:
        return False

    row.generated_caption = vision_cap
    row.caption_source = "vision"
    row.caption_normalized = normalize_caption(vision_cap)

    snippet = (row.page_text_snippet or "")[:600]
    row.image_type = classify_image_type(vision_cap, snippet)
    if row.image_type == "unknown":
        # Textbook figures described by vision are pedagogical diagrams by default.
        row.image_type = "diagram"
    row.educational_role = classify_educational_role(row.image_type, vision_cap)
    row.educational_salience = max(
        float(row.educational_salience or 0),
        compute_educational_salience(vision_cap),
    )
    row.has_caption = True
    row.is_decorative = False

    row.figure_context = build_figure_context(
        caption=vision_cap,
        nearby_before=nearby_before or (row.nearby_text_before_figure or ""),
        nearby_after=nearby_after or (row.nearby_text_after_figure or ""),
        section_title=row.section_title,
        subsection_title=row.subsection_title,
        chapter_title=row.chapter_title,
        page_snippet=row.page_text_snippet,
    )

    from app.services.image_service.caption_generator import (
        extract_semantic_keywords,
        generate_educational_tags,
        generate_educational_title,
    )

    row.semantic_keywords = extract_semantic_keywords(
        vision_cap,
        vision_cap,
        nearby_before or (row.nearby_text_before_figure or ""),
        nearby_after or (row.nearby_text_after_figure or ""),
        row.section_title,
        row.chapter_title,
        row.image_type,
    )
    row.educational_tags = generate_educational_tags(
        vision_cap,
        vision_cap,
        nearby_before or (row.nearby_text_before_figure or ""),
        nearby_after or (row.nearby_text_after_figure or ""),
        row.image_type,
        row.educational_role,
        row.section_title,
    )
    edu = generate_educational_title(
        figure_number=row.figure_number,
        image_type=row.image_type,
        caption=vision_cap,
        section_title=row.section_title,
        subsection_title=row.subsection_title,
        chapter_title=row.chapter_title,
        nearby_before=nearby_before or (row.nearby_text_before_figure or ""),
        nearby_after=nearby_after or (row.nearby_text_after_figure or ""),
    )
    if edu.get("short_title"):
        row.title = edu["short_title"]
    if edu.get("description"):
        row.educational_description = edu["description"]
    if edu.get("concept_tags"):
        row.concept_tags = "|".join(edu["concept_tags"])[:512]

    row.context_bge_indexed = False
    row.multimodal_indexed = False
    return True
