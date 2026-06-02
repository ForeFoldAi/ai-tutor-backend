"""
Figure-context gates for symbolic filtering (minimal-caption topic anchors).

When a caption is only a figure label (e.g. "Fig. 2.2"), concept overlap must be
judged from figure_context (nearby text + section), never from the shared page
snippet that lets unrelated page-mates through.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.modules.catalog.models import TextbookImage
    from app.services.image_service.image_intent_extractor import ImageIntent

def is_minimal_figure_caption(caption: str | None) -> bool:
    """True when the caption is a bare figure label with no descriptive title."""
    cap = (caption or "").strip()
    if not cap:
        return True
    from app.services.image_service.textbook_image_extraction import _FIG_ONLY_CAPTION_RE

    if _FIG_ONLY_CAPTION_RE.match(cap):
        return True
    from app.services.image_service.textbook_image_extraction import normalize_caption

    norm = normalize_caption(cap)
    if not norm:
        return True
    if re.fullmatch(r"[\d.]+", norm.replace(" ", "")):
        return True
    words = [w for w in re.findall(r"\b[a-z][a-z]+\b", norm) if len(w) >= 4]
    return len(words) == 0


def figure_descriptive_text_for_gates(
    im: "TextbookImage",
    caption_normalized: str,
) -> str:
    """
    Authoritative topic text for hard gates.

    Minimal captions → nearby before/after + section (no shared page snippet).
    Descriptive captions → caption + section only (existing symbolic rule).
    """
    sec = (getattr(im, "section_title", None) or "").strip()
    sub = (getattr(im, "subsection_title", None) or "").strip()
    parts: list[str] = []

    if is_minimal_figure_caption(im.caption):
        before = (getattr(im, "nearby_text_before_figure", None) or "").strip()
        after = (getattr(im, "nearby_text_after_figure", None) or "").strip()
        if before:
            parts.append(before)
        if caption_normalized and not re.fullmatch(r"[\d.]+", caption_normalized.replace(" ", "")):
            parts.append(caption_normalized)
        elif (im.caption or "").strip():
            parts.append((im.caption or "").strip())
        if after:
            parts.append(after)
    else:
        if caption_normalized:
            parts.append(caption_normalized)
        elif (im.caption or "").strip():
            parts.append((im.caption or "").strip())

    if sec:
        parts.append(sec)
    if sub and sub != sec:
        parts.append(sub)
    ch = (getattr(im, "chapter_title", None) or "").strip()
    if ch:
        parts.append(ch)
    return " ".join(parts).lower()


def context_supports_topic(intent: "ImageIntent", gate_text: str) -> bool:
    """Whether figure_context/nearby text aligns with the query topic."""
    if not gate_text.strip():
        return False

    core = (intent.core_concept or "").lower().strip()
    if core and core in gate_text:
        return True

    from app.services.image_service.symbolic_image_filters import tokenize, level3_overlap_allowed

    concept_tokens = intent.concept_tokens
    overlap = concept_tokens & tokenize(gate_text)
    if not overlap:
        return False
    if len(core.split()) < 2:
        return True
    return level3_overlap_allowed(intent, overlap)
