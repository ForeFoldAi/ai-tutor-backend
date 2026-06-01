"""
Topic-centric figure retrieval with mandatory / supporting stages.

Pipeline:
  1. Symbolic hard filters (no page rescue)
  2. Mandatory figure stage — bypasses ranking entirely
  3. Supporting figure stage — weighted BGE context-first scoring
  4. Adaptive assembly (1–3 images)

Mandatory figures NEVER receive a score bonus; they are selected first.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.modules.catalog.models import TextbookImage
from app.services.figure_context_bge import embed_query, figure_context_bge_score
from app.services.image_intent_extractor import ImageIntent
from app.services.symbolic_image_filters import (
    SymbolicMatchInfo,
    count_required_term_matches,
    extract_caption_entities,
    tokenize,
    topic_purity_score,
)
from app.services.textbook_image_extraction import build_figure_context, normalize_caption

logger = logging.getLogger(__name__)

_FIG_REF_RE = re.compile(r"(?i)\b(?:fig\.?|figure)\s*([0-9]+(?:\.[0-9]+)*)")

# Preferred teaching types (subject-agnostic categories)
_PREFERRED_TYPES: frozenset[str] = frozenset({
    "diagram", "process", "instrument", "map", "chart", "weather_station",
})
_LOW_PRIORITY_TYPES: frozenset[str] = frozenset({
    "wildlife", "landscape", "decorative", "unknown",
})

_ROLE_SCORE: dict[str, float] = {
    "primary_concept": 100.0,
    "supporting_example": 75.0,
    "activity": 70.0,
    "sidebar_example": 30.0,
    "decorative": 5.0,
    "unknown": 50.0,
}

_RANK_WEIGHTS = {
    "context": 0.40,
    "caption": 0.15,
    "section": 0.15,
    "type": 0.10,
    "role": 0.10,
    "page": 0.05,
    "clip": 0.05,
}


@dataclass
class FigureRankRecord:
    """Explainable record for mandatory or supporting selection."""

    image_id: str
    file_name: str
    mandatory: bool = False
    topic_anchor: bool = False
    figure_context_score: float = 0.0
    caption_score: float = 0.0
    section_score: float = 0.0
    type_score: float = 0.0
    role_score: float = 0.0
    page_score: float = 0.0
    clip_score: float = 0.0
    final_score: float = 0.0
    selected_reason: str | None = None
    rejected_reason: str | None = None
    hard_rejected: bool = False
    concept_specificity: float = 0.0
    topic_purity: float = 0.0


@dataclass
class RetrievalStages:
    mandatory_images: list[tuple[TextbookImage, FigureRankRecord]] = field(default_factory=list)
    supporting_images: list[tuple[TextbookImage, FigureRankRecord]] = field(default_factory=list)
    rejected: list[FigureRankRecord] = field(default_factory=list)


def get_figure_context(im: TextbookImage) -> str:
    ctx = getattr(im, "figure_context", None)
    if ctx and str(ctx).strip():
        return str(ctx).strip()[:4000]
    return build_figure_context(
        caption=im.caption,
        nearby_before=getattr(im, "nearby_text_before_figure", None) or "",
        nearby_after=getattr(im, "nearby_text_after_figure", None) or "",
        section_title=getattr(im, "section_title", None),
        subsection_title=getattr(im, "subsection_title", None),
        chapter_title=getattr(im, "chapter_title", None),
        page_snippet=im.page_text_snippet,
    )


def get_has_caption(im: TextbookImage) -> bool:
    from app.services.figure_context_gates import is_minimal_figure_caption

    if is_minimal_figure_caption(im.caption):
        return False
    if getattr(im, "has_caption", None) is not None:
        return bool(im.has_caption)
    return len((im.caption or "").strip()) >= 8


def _query_text(intent: ImageIntent, rag_docs: list[Any] | None) -> str:
    parts = [intent.intent_text]
    if rag_docs:
        for d in rag_docs[:4]:
            t = (getattr(d, "page_content", None) or "").strip()
            if t:
                parts.append(t[:700])
    return "\n".join(parts)[:3200]


def _bge_cosine_pair(query_text: str, doc_text: str, qvec: np.ndarray | None) -> float:
    if not doc_text.strip():
        return 0.0
    if qvec is not None:
        from app.services.figure_context_bge import embed_texts, cosine_100

        dvecs = embed_texts([doc_text[:2000]])
        if dvecs[0] is None:
            return 0.0
        return cosine_100(qvec, dvecs[0])
    from app.services.figure_context_bge import embed_texts, cosine_100

    vecs = embed_texts([query_text, doc_text[:2000]])
    if vecs[0] is None or vecs[1] is None:
        return 0.0
    return cosine_100(vecs[0], vecs[1])


def caption_bge_score(
    intent: ImageIntent,
    im: TextbookImage,
    rag_docs: list[Any] | None,
    qvec: np.ndarray | None,
    *,
    context_fallback: float,
) -> float:
    if not get_has_caption(im):
        return context_fallback
    cap = getattr(im, "caption_normalized", None) or normalize_caption(im.caption or "")
    return _bge_cosine_pair(_query_text(intent, rag_docs), cap, qvec)


def section_match_score(intent: ImageIntent, im: TextbookImage) -> float:
    """Match retrieved section/subsection/topic against image metadata."""
    topic_tokens = set(intent.concept_tokens)
    topic_tokens |= intent.rag_section_tokens
    for t in (intent.core_concept or "").lower().split():
        if len(t) >= 3:
            topic_tokens.add(t)

    meta_parts = [
        getattr(im, "section_title", None) or "",
        getattr(im, "subsection_title", None) or "",
        getattr(im, "chapter_title", None) or "",
        get_figure_context(im)[:500],
    ]
    img_tokens = tokenize(" ".join(meta_parts))
    if not img_tokens or not topic_tokens:
        return 0.0

    common = img_tokens & topic_tokens
    if intent.rag_section_tokens:
        sec_common = img_tokens & intent.rag_section_tokens
        if sec_common:
            return min(100.0, 60.0 + len(sec_common) * 15.0)
    ratio = len(common) / max(1, min(len(topic_tokens), 12))
    return min(100.0, ratio * 120.0)


def image_type_score(im: TextbookImage, intent: ImageIntent) -> float:
    img_type = getattr(im, "image_type", None) or "unknown"
    if intent.preferred_types and img_type in intent.preferred_types:
        idx = intent.preferred_types.index(img_type)
        return max(75.0, 100.0 - idx * 8.0)
    if img_type in _PREFERRED_TYPES:
        return 80.0
    if img_type in _LOW_PRIORITY_TYPES:
        return 25.0
    if img_type in (intent.excluded_types or []):
        return 0.0
    return 50.0


def educational_role_score(im: TextbookImage) -> float:
    role = getattr(im, "educational_role", None) or "unknown"
    return _ROLE_SCORE.get(role, 50.0)


def caption_conflicts_with_topic(
    intent: ImageIntent,
    im: TextbookImage,
    concept_specificity: float,
) -> tuple[bool, str]:
    if not get_has_caption(im):
        return False, ""
    cap_norm = normalize_caption(im.caption or "")
    full_text = f"{cap_norm} {im.page_text_snippet or ''}".lower()
    _, passes = count_required_term_matches(intent, full_text)
    if len((intent.core_concept or "").split()) >= 2 and not passes and concept_specificity < 35.0:
        return True, "caption_topic_conflict"
    cap_ents = extract_caption_entities(im.caption or "", im.page_text_snippet or "")
    purity = topic_purity_score(intent, cap_ents, [], [])
    if purity < 0.12 and concept_specificity < 30.0:
        return True, "caption_entity_mismatch"
    return False, ""


def _rag_citation_pages(rag_docs: list[Any] | None) -> dict[str, set[int]]:
    from app.services.query_match import document_page

    out: dict[str, set[int]] = {}
    for d in rag_docs or []:
        meta = getattr(d, "metadata", None) or {}
        uid = meta.get("textbook_upload_id")
        if uid:
            out.setdefault(str(uid), set()).add(document_page(d))
    return out


def _figure_referenced_in_rag(fig_number: str | None, rag_docs: list[Any] | None) -> bool:
    if not fig_number or not rag_docs:
        return False
    pats = [f"fig. {fig_number}", f"fig.{fig_number}", f"figure {fig_number}"]
    for d in rag_docs:
        text = (getattr(d, "page_content", None) or "").lower()
        if any(p in text for p in pats):
            return True
    return False


def _figure_referenced_in_context(im: TextbookImage, fig_number: str | None) -> bool:
    if not fig_number:
        return False
    blob = get_figure_context(im).lower()
    pats = [f"fig. {fig_number}", f"fig.{fig_number}", f"figure {fig_number}"]
    return any(p in blob for p in pats)


def classify_mandatory_figure(
    intent: ImageIntent,
    im: TextbookImage,
    rag_docs: list[Any] | None,
    *,
    context_score: float,
    section_score: float,
    page_proximity: float,
    concept_specificity: float,
) -> tuple[bool, bool, str]:
    """
    Returns (is_mandatory, is_topic_anchor, reason).

    Topic-anchor: main teaching illustration (often weak/missing caption).
    """
    reasons: list[str] = []
    topic_anchor = False

    fig_num = getattr(im, "figure_number", None)
    if not fig_num and im.caption:
        m = _FIG_REF_RE.search(im.caption)
        fig_num = m.group(1) if m else None

    ctx = get_figure_context(im).lower()
    core = (intent.core_concept or "").lower()

    if _figure_referenced_in_rag(fig_num, rag_docs) or _figure_referenced_in_context(im, fig_num):
        reasons.append("educational_content_reference")

    if core and core in ctx and context_score >= 50.0:
        reasons.append("concept_centered_explanation")
        if context_score >= 58.0 and section_score >= 45.0:
            reasons.append("same_section_and_topic")

    citation_pages = _rag_citation_pages(rag_docs)
    on_citation = str(im.textbook_upload_id) in citation_pages and im.page_index in citation_pages.get(
        str(im.textbook_upload_id), set()
    )

    if not get_has_caption(im) and context_score >= 48.0 and (on_citation or section_score >= 42.0):
        reasons.append("uncaptioned_teaching_figure")
        topic_anchor = True

    from app.services.figure_context_gates import is_minimal_figure_caption

    if is_minimal_figure_caption(im.caption) and context_score >= 52.0 and core and core in ctx:
        topic_anchor = True
        reasons.append("topic_anchor_minimal_caption")

    if context_score >= 58.0 and section_score >= 50.0 and concept_specificity >= 40.0:
        if not get_has_caption(im) or len((im.caption or "")) < 40:
            topic_anchor = True
            reasons.append("topic_anchor_figure")

    role = getattr(im, "educational_role", None) or ""
    if role == "primary_concept" and context_score >= 55.0 and section_score >= 45.0:
        topic_anchor = True
        reasons.append("primary_concept_anchor")

    if on_citation and context_score >= 54.0 and page_proximity >= 6.0:
        reasons.append("citation_page_teaching_figure")

    img_type = getattr(im, "image_type", None) or "unknown"
    if img_type in ("weather_station", "instrument") and intent.query_type == "concept_definition":
        qlow = (intent.query or "").lower()
        if not re.search(r"\b(station|aws|instrument|sensor|gauge|anemometer)\b", qlow):
            return False, False, "instrument_not_requested"

    is_mandatory = bool(reasons) and context_score >= 45.0 and (
        concept_specificity >= 20.0
        or topic_anchor
        or (core and core in ctx and context_score >= 55.0)
    )
    return is_mandatory, topic_anchor, ";".join(reasons)


def compute_supporting_score(
    rec: FigureRankRecord,
) -> float:
    """Weighted score for non-mandatory images only."""
    total = (
        _RANK_WEIGHTS["context"] * rec.figure_context_score
        + _RANK_WEIGHTS["caption"] * rec.caption_score
        + _RANK_WEIGHTS["section"] * rec.section_score
        + _RANK_WEIGHTS["type"] * rec.type_score
        + _RANK_WEIGHTS["role"] * rec.role_score
        + _RANK_WEIGHTS["page"] * rec.page_score
        + _RANK_WEIGHTS["clip"] * rec.clip_score
    )
    rec.final_score = round(total, 2)
    return rec.final_score


def record_to_debug_dict(rec: FigureRankRecord, im: TextbookImage) -> dict:
    return {
        "image_id": rec.image_id,
        "file_name": rec.file_name,
        "mandatory": rec.mandatory,
        "topic_anchor": rec.topic_anchor,
        "figure_context_score": round(rec.figure_context_score, 1),
        "caption_score": round(rec.caption_score, 1),
        "section_score": round(rec.section_score, 1),
        "type_score": round(rec.type_score, 1),
        "role_score": round(rec.role_score, 1),
        "page_score": round(rec.page_score, 1),
        "clip_score": round(rec.clip_score, 4),
        "final_score": round(rec.final_score, 1),
        "selected_reason": rec.selected_reason,
        "rejected_reason": rec.rejected_reason,
        "concept_specificity": round(rec.concept_specificity, 1),
        "topic_purity": round(rec.topic_purity, 3),
        "caption": (im.caption or "")[:120],
        "educational_role": getattr(im, "educational_role", "unknown"),
    }


def run_topic_centric_retrieval(
    intent: ImageIntent,
    candidates: list[tuple[TextbookImage, SymbolicMatchInfo]],
    rag_docs: list[Any] | None,
    pages_by_upload: dict[str, list[int]],
    *,
    concept_specificity_fn: Any,
    page_proximity_fn: Any,
    clip_sims: dict[str, float] | None = None,
    min_supporting_score: float = 40.0,
    max_images: int = 3,
) -> RetrievalStages:
    """
    Split candidates into mandatory (no ranking) and ranked supporting figures.
    """
    from app.config import MIN_TOPIC_PURITY

    stages = RetrievalStages()
    qtext = _query_text(intent, rag_docs)
    qvec = embed_query(qtext)
    clip_map = clip_sims or {}
    seen_mandatory: set[str] = set()

    scored_supporting: list[tuple[TextbookImage, FigureRankRecord]] = []

    for im, sym in candidates:
        rec = FigureRankRecord(
            image_id=str(im.id),
            file_name=im.file_name,
        )
        spec = float(concept_specificity_fn(intent, im))
        rec.concept_specificity = spec

        role = getattr(im, "educational_role", None) or "unknown"
        if role == "decorative":
            rec.hard_rejected = True
            rec.rejected_reason = "decorative_role"
            stages.rejected.append(rec)
            continue

        cap_reject, cap_reason = caption_conflicts_with_topic(intent, im, spec)
        if cap_reject:
            rec.hard_rejected = True
            rec.rejected_reason = cap_reason
            stages.rejected.append(rec)
            logger.debug("[GROUND] reject %s: %s", im.file_name, cap_reason)
            continue

        rec.figure_context_score = figure_context_bge_score(qtext, im, query_vec=qvec)
        rec.section_score = section_match_score(intent, im)
        prox = float(page_proximity_fn(str(im.textbook_upload_id), im.page_index, pages_by_upload))
        rec.page_score = min(100.0, prox / 24.0 * 100.0)
        rec.caption_score = caption_bge_score(
            intent, im, rag_docs, qvec, context_fallback=rec.figure_context_score
        )
        rec.type_score = image_type_score(im, intent)
        rec.role_score = educational_role_score(im)
        rec.clip_score = round(float(clip_map.get(str(im.id), 0.0)), 4)

        cap_ents = extract_caption_entities(im.caption or "", im.page_text_snippet or "")
        req, _ = count_required_term_matches(
            intent,
            f"{normalize_caption(im.caption or '')} {get_figure_context(im)}".lower(),
        )
        rec.topic_purity = topic_purity_score(intent, cap_ents, req, sym.entity_matches if sym else [])

        if rec.topic_purity < MIN_TOPIC_PURITY and rec.figure_context_score < 42.0:
            rec.hard_rejected = True
            rec.rejected_reason = f"low_topic_purity={rec.topic_purity:.2f}"
            stages.rejected.append(rec)
            continue

        is_mand, topic_anchor, mand_reason = classify_mandatory_figure(
            intent, im, rag_docs,
            context_score=rec.figure_context_score,
            section_score=rec.section_score,
            page_proximity=prox,
            concept_specificity=spec,
        )

        if is_mand and str(im.id) not in seen_mandatory:
            rec.mandatory = True
            rec.topic_anchor = topic_anchor
            rec.selected_reason = f"mandatory_stage:{mand_reason}"
            rec.final_score = rec.figure_context_score
            seen_mandatory.add(str(im.id))
            stages.mandatory_images.append((im, rec))
            logger.info(
                "[GROUND] MANDATORY %s anchor=%s ctx=%.0f reason=%s",
                im.file_name, topic_anchor, rec.figure_context_score, mand_reason,
            )
            continue

        compute_supporting_score(rec)
        if rec.final_score < min_supporting_score:
            rec.rejected_reason = f"below_min_supporting={rec.final_score:.1f}"
            stages.rejected.append(rec)
            continue

        scored_supporting.append((im, rec))

    if len(stages.mandatory_images) > max_images:
        stages.mandatory_images.sort(
            key=lambda x: (
                -int(x[1].topic_anchor),
                -x[1].figure_context_score,
                -x[1].concept_specificity,
            )
        )
        for im, rec in stages.mandatory_images[max_images:]:
            rec.rejected_reason = "mandatory_cap_exceeded"
            stages.rejected.append(rec)
        stages.mandatory_images = stages.mandatory_images[:max_images]

    scored_supporting.sort(key=lambda x: -x[1].final_score)

    max_supporting = max(0, max_images - len(stages.mandatory_images))
    if max_supporting == 0:
        return stages

    if not stages.mandatory_images and scored_supporting:
        im, rec = scored_supporting[0]
        rec.selected_reason = "supporting_stage:single_clear_match"
        stages.supporting_images.append((im, rec))
        return stages

    top_mand_ctx = (
        max(r.figure_context_score for _, r in stages.mandatory_images)
        if stages.mandatory_images else 0.0
    )

    added = 0
    for im, rec in scored_supporting:
        if added >= max_supporting:
            break
        if len(stages.mandatory_images) >= max_images:
            break
        if stages.mandatory_images and rec.final_score < top_mand_ctx * 0.40:
            rec.rejected_reason = "too_weak_vs_mandatory"
            stages.rejected.append(rec)
            continue
        if added >= 1 and rec.final_score < min_supporting_score * 0.88:
            continue
        rec.selected_reason = "supporting_stage:ranked"
        stages.supporting_images.append((im, rec))
        added += 1
        logger.info(
            "[GROUND] SUPPORTING %s ctx=%.0f cap=%.0f score=%.1f",
            im.file_name, rec.figure_context_score, rec.caption_score, rec.final_score,
        )

    return stages


def assemble_payload_rows(
    stages: RetrievalStages,
) -> list[tuple[TextbookImage, float, dict]]:
    """Mandatory first, then supporting — never interleaved by score."""
    rows: list[tuple[TextbookImage, float, dict]] = []
    for im, rec in stages.mandatory_images:
        rows.append((im, rec.final_score, record_to_debug_dict(rec, im)))
    for im, rec in stages.supporting_images:
        rows.append((im, rec.final_score, record_to_debug_dict(rec, im)))
    return rows
