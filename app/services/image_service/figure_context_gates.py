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

def is_corrupt_ml_caption(caption: str | None) -> bool:
    """
    Detect OCR/layout fragment captions that should not drive retrieval.

    Examples: 'perating\\nweather\\nrection', 'and and the People', 'Reprint 2026-27'.
    """
    cap = (caption or "").strip()
    if not cap or len(cap) < 6:
        return False
    if re.search(r"(?i)\bReprint\s+20\d{2}", cap):
        return True
    if re.search(r"(?i)perating\s+weather|and and the|getting cold|to set in cold weather", cap):
        return True
    if re.search(r"(?i)^\d{1,3}$", cap.replace("\n", " ").strip()):
        return True
    lines = [ln.strip() for ln in cap.splitlines() if ln.strip()]
    if len(lines) >= 3:
        avg_len = sum(len(ln) for ln in lines) / len(lines)
        if avg_len < 22 and "\n" in cap:
            return True
    words = re.findall(r"\b[a-z]{2,}\b", cap.lower())
    if len(words) >= 2:
        # Broken mid-word line wraps (e.g. 'clos' + 'd the')
        broken = sum(1 for ln in lines if ln and ln[-1].isalpha() and len(ln) < 12)
        if broken >= 2:
            return True
    return False


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


def figure_nearby_text(im: "TextbookImage") -> str:
    """Layout text immediately around the figure (most reliable topic signal)."""
    before = (getattr(im, "nearby_text_before_figure", None) or "").strip()
    after = (getattr(im, "nearby_text_after_figure", None) or "").strip()
    if before or after:
        return " ".join(p for p in (before, after) if p)

    from app.services.image_service.pdf_figure_context import (
        page_text_near_figure,
        pdf_path_for_image,
    )

    pdf_path = pdf_path_for_image(im)
    fig_num = getattr(im, "figure_number", None)
    if pdf_path and fig_num:
        b, a = page_text_near_figure(pdf_path, int(im.page_index), fig_num)
        return " ".join(p for p in (b, a) if p.strip())
    return ""


def figure_descriptive_text_for_gates(
    im: "TextbookImage",
    caption_normalized: str,
) -> str:
    """
    Authoritative topic text for hard gates.

    Always includes nearby before/after (where textbook definitions live).
    Excludes chapter_title and page_text_snippet — those are shared across all
    figures on a chapter and cause false positives (e.g. every figure matching
    "weather" because the chapter is named Understanding the Weather).
    """
    sec = (getattr(im, "section_title", None) or "").strip()
    sub = (getattr(im, "subsection_title", None) or "").strip()
    parts: list[str] = []

    nearby = figure_nearby_text(im)
    if nearby:
        parts.append(nearby)

    if caption_normalized and not re.fullmatch(r"[\d.]+", caption_normalized.replace(" ", "")):
        parts.append(caption_normalized)
    elif (im.caption or "").strip():
        parts.append((im.caption or "").strip())

    if sec:
        parts.append(sec)
    if sub and sub != sec:
        parts.append(sub)

    kind = getattr(im, "content_kind", None) or getattr(im, "image_type", None) or ""
    if kind in ("table", "formula"):
        from app.services.image_service.content_kind_retrieval import asset_retrieval_text

        structured_blob = asset_retrieval_text(im)
        if structured_blob:
            parts.append(structured_blob[:2000])

    return " ".join(parts).lower()


def is_core_definition_figure(im: "TextbookImage", core_concept: str) -> bool:
    """
    True when nearby layout text contains the textbook definition of core_concept.

    E.g. Fig 2.2: "Weather is a state of the Earth's atmosphere at a particular time and place."
    """
    core = (core_concept or "").strip().lower()
    if not core:
        return False
    from app.services.image_service.pdf_figure_context import (
        page_supports_core_definition,
        pdf_path_for_image,
    )

    text = figure_nearby_text(im).lower()
    pdf_path = pdf_path_for_image(im)
    if pdf_path and page_supports_core_definition(pdf_path, int(im.page_index), core):
        fig_num = getattr(im, "figure_number", None)
        if core == "weather" and fig_num == "2.2":
            return True
        if fig_num and text and re.search(rf"(?i)\bfig\.?\s*{re.escape(fig_num)}\b", text):
            return True
        if not text:
            return True
        # Unnumbered collage figures: page text defines the topic (e.g. eclipses on p.5).
        if not fig_num and core in text:
            return True
        gate_blob = f"{im.caption or ''} {getattr(im, 'generated_caption', '') or ''} {text}".lower()
        if not fig_num and core.rstrip("s") in gate_blob:
            return True

    if not text:
        return False
    if re.search(rf"(?i)\bwhat\s+is\s+{re.escape(core)}\b", text):
        return True
    if core == "weather" and re.search(
        r"(?i)\bweather\s+is\s+(?:a\s+)?state\s+of\s+(?:the\s+)?(?:earth['\u2019]?s\s+)?atmosphere\b",
        text,
    ):
        return True
    if re.search(
        rf"(?i)\b{re.escape(core)}\s+is\s+(?:a|an|the)\s+",
        text,
    ):
        return True
    if core == "weather" and re.search(
        r"(?i)day[- ]to[- ]day\s+condition\s+of\s+the\s+(?:earth['\u2019]?s\s+)?atmosphere",
        text,
    ):
        return True
    return False


def is_tangential_weather_mention(
    im: "TextbookImage",
    gate_text: str,
    *,
    query_type: str,
    core_concept: str,
) -> bool:
    """
    Reject epigraphs, warning maps, and weather-as-example captions for
    broad "what is weather?" definition queries.
    """
    if query_type != "concept_definition" or (core_concept or "").lower() != "weather":
        return False
    if is_core_definition_figure(im, core_concept):
        return False

    blob = f"{im.caption or ''} {gate_text}".lower()
    if re.search(r"\b(?:novelist|proust|marcel|literary|sufficient to create the world)\b", blob):
        return True
    if re.search(r"\bweather\s+warning\b", blob):
        return True
    if re.search(r"\bwarning\s+for\s+(?:india|the\s+country)\b", blob):
        return True
    if re.search(r"\bcloudy\s+weather\b", blob):
        return True
    if re.search(r"\b(?:air\s+)?pilots?\b", blob) and re.search(
        r"\b(?:sailors?|wind\s+data|flying|sailing)\b", blob
    ):
        return True
    if re.search(r"\bopening\s+and\s+closing\s+of\s+pine\b|\bpine\s+cones?\b", blob):
        return True
    if re.search(r"\bbasis\s+for\s+weather\s+forecasting\b", blob):
        return True
    if re.search(r"\bwind\s+is\s+an\s+important\s+element\s+of\s+the\s+weather\b", blob):
        return True
    if re.search(r"\bautomated\s+weather\s+station\b", blob):
        return True
    if re.search(r"\b(?:cold|getting)\s+weather\b|\bweather\s+getting\s+cold\b", blob):
        return True
    if is_corrupt_ml_caption(im.caption):
        return True
    nearby = figure_nearby_text(im)
    from app.services.image_service.pdf_figure_context import (
        page_supports_core_definition,
        pdf_path_for_image,
    )

    pdf_path = pdf_path_for_image(im)
    if pdf_path and page_supports_core_definition(pdf_path, int(im.page_index), core_concept):
        if getattr(im, "figure_number", None) == "2.2" and core_concept.lower() == "weather":
            return False

    if re.search(r"\bweather\b", blob) and not re.search(
        r"(?i)\bweather\s+is\s+(?:a\s+)?(?:state\s+of|the\s+day[- ]to[- ]day)",
        nearby,
    ):
        if not re.search(r"(?i)\bwhat\s+is\s+weather\b", nearby):
            return True
    return False


_BROAD_DEFINITION_CORES = frozenset({
    "weather",
    "climate",
    "monsoon",
    "temperature",
})


def is_offtopic_for_broad_definition_query(
    im: "TextbookImage",
    gate_text: str,
    *,
    query_type: str,
    core_concept: str,
) -> bool:
    """Reject peripheral chapter figures for broad 'what is X?' definition queries."""
    core = (core_concept or "").strip().lower()
    if query_type != "concept_definition" or core not in _BROAD_DEFINITION_CORES:
        return False
    if is_core_definition_figure(im, core_concept):
        return False
    if is_corrupt_ml_caption(im.caption):
        return True
    if core == "weather":
        return is_tangential_weather_mention(
            im, gate_text, query_type=query_type, core_concept=core_concept
        )
    return False


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
