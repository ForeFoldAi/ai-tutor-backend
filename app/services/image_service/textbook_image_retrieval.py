"""
Topic-centric textbook figure retrieval with figure grounding.

Pipeline:
  Question → RAG → topic intent → symbolic filters → figure_context BGE
  → mandatory figure detection → supporting rank → adaptive 1–3 images

Ranking (non-mandatory):
  0.35 figure_context + 0.20 caption + 0.15 section + 0.10 type
  + 0.10 role + 0.05 page + 0.05 CLIP

Mandatory figures are selected in a separate stage and never compete in ranking.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.modules.catalog.models import TextbookImage, TextbookUpload
from app.services.query_match import document_page, keyword_match_score
from app.services.image_service.textbook_image_display import image_has_visible_content
from app.services.image_service.textbook_image_extraction import (
    image_disk_path,
    classify_image_type,
    compute_educational_salience,
    ensure_textbook_images_extracted,
    normalize_caption,
)
from app.services.image_service.image_intent_extractor import ImageIntent, extract_image_intent
from app.services.image_service.symbolic_image_filters import (
    SymbolicMatchInfo,
    apply_symbolic_hard_filters,
    count_required_term_matches,
    entity_match_score,
    level3_overlap_allowed,
    topic_purity_score,
    extract_caption_entities,
)
from app.services.image_service.figure_grounding import (
    assemble_payload_rows,
    record_to_debug_dict,
    run_topic_centric_retrieval,
)

logger = logging.getLogger(__name__)

_DEFAULT_TOP = 3
_SEMANTIC_CAP = 36
_MMR_LAMBDA = 0.70          # relevance vs diversity trade-off

# Pedagogical salience weights per image type (0-100)
_TYPE_PEDAGOGICAL_SALIENCE: dict[str, float] = {
    "weather_station": 100.0,
    "instrument":      90.0,
    "diagram":         90.0,
    "process":         85.0,
    "map":             80.0,
    "chart":           75.0,
    "satellite_image": 70.0,
    "activity":        70.0,
    "landscape":       60.0,
    "historical_photo": 55.0,
    "wildlife":        40.0,
    "table":           85.0,
    "formula":         80.0,
    "unknown":         50.0,
}

# Educational-role salience multiplier
_ROLE_MULTIPLIER: dict[str, float] = {
    "primary_concept":    1.20,
    "supporting_example": 1.00,
    "activity":           0.90,
    "sidebar_example":    0.75,
    "decorative":         0.30,
    "unknown":            0.85,
}
_GENERIC_ASSET_NAME = re.compile(r"^(im\d+|p\d+_\d+|image\d+)\.", re.I)
_IMAGE_REQUEST_PATTERNS = re.compile(
    r"\b("
    r"with\s+(?:an?\s+)?(?:images?|diagrams?|pictures?|figures?|maps?|illustrations?)|"
    r"show\s+(?:me\s+)?(?:an?\s+|the\s+)?(?:diagrams?|figures?|pictures?|maps?|illustrations?)|"
    r"include\s+(?:an?\s+)?images?|using\s+(?:diagrams?|pictures?|illustrations?)|"
    r"(?:see|want|need)\s+(?:an?\s+)?(?:images?|diagrams?|pictures?|figures?)|"
    r"textbook\s+(?:diagram|figure|image)s?"
    r")\b",
    re.I,
)

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
    "and", "or", "to", "for", "with", "by", "from", "this", "that", "it",
    "its", "as", "be", "been", "being", "have", "has", "had", "not", "no",
    "its", "these", "those", "some", "also", "which", "where", "when",
})


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _tokenize(text: str) -> frozenset[str]:
    tokens = set(re.findall(r"\b[a-z][a-z]+\b", (text or "").lower()))
    return frozenset(tokens - _STOPWORDS)


def _chapter_label_match_score(upload_chapter: str | None, chapter_hints: list[str]) -> float:
    if not upload_chapter or not chapter_hints:
        return 0.0
    u = _norm(upload_chapter)
    best = 0.0
    for h in chapter_hints:
        hn = _norm(h)
        if not hn:
            continue
        if hn in u or u in hn:
            best = max(best, 55.0)
        ut = set(re.findall(r"\w+", u))
        ht = set(re.findall(r"\w+", hn))
        if ut & ht:
            best = max(best, 22.0 + 3.0 * len(ut & ht))
    return best


def _topic_keyword_score(query: str, context_excerpt: str, image: TextbookImage) -> float:
    from app.services.image_service.content_kind_retrieval import asset_retrieval_text

    blob = asset_retrieval_text(image)
    q = f"{query}\n{context_excerpt[:2400]}"
    return float(keyword_match_score(q, blob)) * 2.8


def _pages_by_upload_from_docs(docs: list[Any]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for d in docs or []:
        meta = getattr(d, "metadata", None) or {}
        uid = meta.get("textbook_upload_id")
        if not uid:
            continue
        out[str(uid)].append(document_page(d))
    return dict(out)


def _page_proximity_score(upload_id: str, page_index: int, pages_by_upload: dict[str, list[int]]) -> float:
    pages = pages_by_upload.get(upload_id) or []
    if not pages:
        return 0.0
    best = 0.0
    for p in pages:
        dist = abs(int(page_index) - int(p))
        # Decay to 0 at 4 pages distance (was 7.5).  Images more than 4 pages from
        # any RAG citation page should not receive a page-proximity boost.
        best = max(best, max(0.0, 24.0 - dist * 6.0))
    return best


def _cosine_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    qn = np.linalg.norm(query_vec)
    if qn < 1e-9:
        return np.zeros(doc_matrix.shape[0])
    dn = np.linalg.norm(doc_matrix, axis=1)
    dn = np.where(dn < 1e-9, 1e-9, dn)
    return (doc_matrix @ query_vec) / (dn * qn)


def _context_excerpt_from_docs(docs: list[Any]) -> str:
    parts: list[str] = []
    for d in docs or []:
        t = (getattr(d, "page_content", None) or "").strip()
        if t:
            parts.append(t[:900])
        if sum(len(p) for p in parts) > 4000:
            break
    return "\n".join(parts)


def _is_generic_asset_filename(name: str) -> bool:
    return bool(_GENERIC_ASSET_NAME.match((name or "").strip()))


def _query_requests_images(text: str) -> bool:
    return bool(_IMAGE_REQUEST_PATTERNS.search(text or ""))


def _build_retrieval_text(
    query: str,
    context_excerpt: str,
    answer_text: str = "",
) -> str:
    """Fuse student question and retrieved context for BGE semantic bonus.

    answer_text is accepted for backward-compat but intentionally excluded
    from this composite — keeping image retrieval clean of LLM hallucination.
    """
    q = (query or "").strip()
    ctx = (context_excerpt or "").strip()[:2400]
    parts: list[str] = []
    if q:
        parts.append(f"Student question:\n{q}")
    if ctx:
        parts.append(f"Textbook context:\n{ctx}")
    return "\n\n".join(parts) if parts else q


# ---------------------------------------------------------------------------
# Low-level figure attribute accessors (handle legacy rows gracefully)
# ---------------------------------------------------------------------------

def _get_image_type(im: TextbookImage) -> str:
    """Return stored image_type, classifying on-the-fly for legacy rows."""
    t = getattr(im, "image_type", None)
    if t and t != "unknown":
        return t
    return classify_image_type(im.caption or "", im.page_text_snippet or "")


def _get_caption_normalized(im: TextbookImage) -> str:
    """Return stored caption_normalized, normalizing on-the-fly for legacy rows."""
    cn = getattr(im, "caption_normalized", None)
    if cn:
        return cn
    return normalize_caption(im.caption or "")


def _get_salience(im: TextbookImage) -> float:
    sal = getattr(im, "educational_salience", None)
    if sal is not None:
        return float(sal)
    return compute_educational_salience(im.caption or "")


def _get_educational_role(im: TextbookImage) -> str:
    role = getattr(im, "educational_role", None)
    if role and role != "unknown":
        return role
    img_type = _get_image_type(im)
    from app.services.image_service.textbook_image_extraction import classify_educational_role
    return classify_educational_role(img_type, im.caption or "")


def _get_section_title(im: TextbookImage) -> str:
    return (getattr(im, "section_title", None) or "").strip()


# ---------------------------------------------------------------------------
# Precision-level 1: Concept specificity score (dominant 40% signal)
# ---------------------------------------------------------------------------

def _concept_specificity_score(intent: "ImageIntent", im: TextbookImage) -> float:
    """
    0–100 score measuring how SPECIFICALLY the figure matches the core educational concept.

    Tiered match levels:
      Level 0 (100): full core_concept phrase in caption/snippet
      Level 1 (70-95): required multi-word phrase match
      Level 2 (20-65): single distinctive required keyword
      Level 3 (0-30): concept-token overlap only (individual common words)
      Level -: negative terms present → penalty applied

    This deliberately scores "cloudy weather" near 0 for "automated weather station" queries:
    - "automated", "weather", "station" are all concept tokens but none form a phrase
    - No required phrases like "weather station" appear
    - Result: ~10-15 (well below the 25-point hard-reject threshold)
    """
    from app.services.image_service.content_kind_retrieval import (
        asset_retrieval_text,
        concept_tag_overlap_score,
        get_content_kind,
        referenced_asset_matches,
    )
    from app.services.image_service.figure_context_gates import (
        figure_descriptive_text_for_gates,
        is_core_definition_figure,
        is_tangential_weather_mention,
    )

    cap = _get_caption_normalized(im)
    sec = _get_section_title(im)
    subsec = (getattr(im, "subsection_title", None) or "").strip()
    gate_text = figure_descriptive_text_for_gates(im, cap)
    if not gate_text.strip() and get_content_kind(im) in ("table", "formula"):
        gate_text = asset_retrieval_text(im)
    # Let stored keywords contribute when enrichment already wrote them.
    sem = (getattr(im, "semantic_keywords", None) or "").strip()
    if sem:
        gate_text = f"{gate_text} {sem}".strip()
    topic_text = gate_text
    caption_text = " ".join(
        p for p in (cap, sec, subsec) if p
    ).lower()

    if referenced_asset_matches(intent, im):
        return 98.0

    if not topic_text.strip():
        tag_only = concept_tag_overlap_score(intent, im)
        return tag_only if tag_only > 0 else 0.0

    core = getattr(intent, "core_concept", None) or ""
    query_type = getattr(intent, "query_type", "") or ""

    from app.services.image_service.figure_context_gates import is_offtopic_for_broad_definition_query

    if is_offtopic_for_broad_definition_query(
        im, gate_text, query_type=query_type, core_concept=core
    ):
        return 0.0
    if is_tangential_weather_mention(im, gate_text, query_type=query_type, core_concept=core):
        return 0.0

    if is_core_definition_figure(im, core):
        return 95.0

    # ── helpers ──────────────────────────────────────────────────────────────
    def _neg_penalty() -> float:
        if not hasattr(intent, "negative_terms"):
            return 0.0
        return sum(15.0 for t in (intent.negative_terms or []) if t.lower() in topic_text)

    def _sup_bonus() -> float:
        if not hasattr(intent, "supporting_terms"):
            return 0.0
        matches = sum(1 for t in (intent.supporting_terms or []) if t.lower() in topic_text)
        return min(15.0, matches * 5.0)

    required_terms: list[str] = getattr(intent, "required_terms", [])
    concept_tokens: frozenset[str] = getattr(intent, "concept_tokens", frozenset())

    # ── Level 0: exact core concept in caption OR nearby layout text ─────────
    if core and core.lower() in topic_text:
        raw = 100.0 + _sup_bonus() - _neg_penalty()
        return max(70.0, min(100.0, raw))

    # ── Level 1: required multi-word phrase match ─────────────────────────────
    phrase_matches = [t for t in required_terms if len(t.split()) > 1 and t.lower() in topic_text]
    if phrase_matches:
        base = 70.0 + len(phrase_matches) * 8.0
        raw = min(95.0, base) + _sup_bonus() - _neg_penalty()
        return max(50.0, min(95.0, raw))

    # ── Level 2: single required term — abbreviation (2-5 alpha chars) OR 8+ char word ─
    # Intentionally excludes 6-7 char generic words like "weather", "station",
    # "rainfall" — those only contribute to Level-3 (concept-token overlap, capped 30).
    def _is_distinctive_single(t: str) -> bool:
        return (len(t) <= 5 and t.isalpha()) or len(t) >= 8

    single_matches = [
        t for t in required_terms
        if len(t.split()) == 1 and _is_distinctive_single(t) and t.lower() in topic_text
    ]
    if single_matches:
        base = 50.0 + len(single_matches) * 6.0
        raw = min(70.0, base) + _sup_bonus() - _neg_penalty()
        return max(15.0, min(70.0, raw))

    # ── Level 3: concept-token overlap — STRICT (no single-token escape) ───────
    cap_tokens = _tokenize(topic_text)
    tok_match = concept_tokens & cap_tokens
    if tok_match and isinstance(intent, ImageIntent) and level3_overlap_allowed(intent, tok_match):
        ratio = len(tok_match) / max(1, len(concept_tokens))
        raw = ratio * 30.0 + _sup_bonus() - _neg_penalty()
        base = max(0.0, min(30.0, raw))
        tag_boost = concept_tag_overlap_score(intent, im)
        return min(100.0, base + tag_boost * 0.45)

    tag_boost = concept_tag_overlap_score(intent, im)
    if tag_boost >= 40.0:
        return min(85.0, tag_boost)

    # ── Level 4: nothing matches ──────────────────────────────────────────────
    return max(0.0, 0.0 - _neg_penalty())


# ---------------------------------------------------------------------------
# Supporting scoring helpers
# ---------------------------------------------------------------------------

def _section_overlap_score(intent: "ImageIntent", im: TextbookImage) -> float:
    """0–100 overlap: RAG section tokens ↔ figure's section_title + caption tokens."""
    rag_tokens: frozenset[str] = getattr(intent, "rag_section_tokens", frozenset())
    if not rag_tokens:
        return 0.0
    # Include stored section_title for extra signal
    from app.services.image_service.content_kind_retrieval import asset_retrieval_text

    sec = _get_section_title(im)
    img_tokens = _tokenize(f"{asset_retrieval_text(im)} {sec}")
    if not img_tokens:
        return 0.0
    common = img_tokens & rag_tokens
    ratio = len(common) / max(1, len(rag_tokens))
    return min(100.0, ratio * 200.0)


def _type_and_salience_score(intent: "ImageIntent", im: TextbookImage) -> float:
    """
    Combined image-type match + pedagogical salience (0-100).
    Fuses two previously separate signals into a single 10% component.
    """
    img_type = _get_image_type(im)
    preferred: list[str] = getattr(intent, "preferred_types", [])
    excluded: list[str] = getattr(intent, "excluded_types", [])

    # Type match score (0-100)
    if not preferred:
        type_score = 50.0  # neutral
    elif img_type in preferred:
        idx = preferred.index(img_type)
        type_score = max(80.0, 100.0 - idx * 10.0)
    elif img_type == "unknown":
        type_score = 40.0
    elif img_type in excluded:
        type_score = 0.0
    else:
        type_score = 20.0

    # Pedagogical salience (type-specific teaching value)
    ped_sal = _TYPE_PEDAGOGICAL_SALIENCE.get(img_type, 50.0)

    # Educational role multiplier
    role = _get_educational_role(im)
    role_mult = _ROLE_MULTIPLIER.get(role, 0.85)

    return min(100.0, (type_score * 0.5 + ped_sal * 0.5) * role_mult)


# ---------------------------------------------------------------------------
# Hard filters
# ---------------------------------------------------------------------------

def _hard_negative(intent: "ImageIntent", im: TextbookImage) -> bool:
    """Backward-compat wrapper for the old TopicIntent-based filter."""
    excluded: list[str] = getattr(intent, "excluded_types", [])
    if not excluded:
        return False
    img_type = _get_image_type(im)
    if img_type not in excluded:
        return False
    cap_tokens = _tokenize(f"{_get_caption_normalized(im)} {im.page_text_snippet or ''}")
    concept_tokens: frozenset[str] = getattr(intent, "concept_tokens", frozenset())
    return not bool(concept_tokens & cap_tokens)


def _hard_concept_filter(intent: "ImageIntent", im: TextbookImage) -> tuple[bool, str]:
    """
    Symbolic-first hard filter pipeline. Returns (reject, reason).
    """
    from app.config import HARD_REJECT_SPECIFICITY, MIN_TOPIC_PURITY, SIDEBAR_MIN_SPECIFICITY

    cap_norm = _get_caption_normalized(im)
    spec = _concept_specificity_score(intent, im)
    img_type = _get_image_type(im)
    role = _get_educational_role(im)

    if isinstance(intent, ImageIntent):
        sym = apply_symbolic_hard_filters(
            intent,
            im,
            caption_normalized=cap_norm,
            image_type=img_type,
            educational_role=role,
            concept_specificity=spec,
            min_topic_purity=MIN_TOPIC_PURITY,
            sidebar_min_specificity=SIDEBAR_MIN_SPECIFICITY,
        )
        if sym.hard_rejected:
            return True, sym.rejection_reason or "symbolic_filter"

    # Legacy fallback for TopicIntent or very weak matches.
    from app.services.image_service.figure_context_gates import (
        context_supports_topic,
        figure_descriptive_text_for_gates,
        is_core_definition_figure,
        is_offtopic_for_broad_definition_query,
        is_tangential_weather_mention,
    )

    gate_text = figure_descriptive_text_for_gates(im, cap_norm)
    qt = getattr(intent, "query_type", "") or ""
    core = getattr(intent, "core_concept", "") or ""
    if is_offtopic_for_broad_definition_query(
        im, gate_text, query_type=qt, core_concept=core
    ):
        return True, "offtopic_broad_definition"
    if is_tangential_weather_mention(im, gate_text, query_type=qt, core_concept=core):
        return True, "tangential_weather_mention"

    concept_tokens: frozenset[str] = getattr(intent, "concept_tokens", frozenset())
    cap_tokens = _tokenize(f"{cap_norm} {_get_section_title(im)}")
    has_any_overlap = bool(concept_tokens & cap_tokens)
    has_any_overlap = has_any_overlap or context_supports_topic(intent, gate_text)
    if is_core_definition_figure(im, getattr(intent, "core_concept", "") or ""):
        has_any_overlap = True
    excluded: list[str] = getattr(intent, "excluded_types", [])

    if img_type in excluded and not has_any_overlap:
        return True, f"excluded_type={img_type}, no concept overlap"

    if spec < HARD_REJECT_SPECIFICITY and not has_any_overlap:
        return True, f"specificity={spec:.1f} < {HARD_REJECT_SPECIFICITY}, no concept overlap"

    return False, ""


def _build_symbolic_match_info(intent: ImageIntent, im: TextbookImage, spec: float) -> SymbolicMatchInfo:
    """Compute symbolic match details for scoring and debug (survivors only)."""
    from app.config import MIN_TOPIC_PURITY, SIDEBAR_MIN_SPECIFICITY

    cap_norm = _get_caption_normalized(im)
    sec = getattr(im, "section_title", None) or ""
    # Caption-based evidence (matches the hard-gate logic; excludes page snippet).
    caption_text = f"{cap_norm} {sec}".lower()
    req_matches, _ = count_required_term_matches(intent, caption_text)
    ent_matches, ent_conflicts, _ = entity_match_score(intent, caption_text)
    cap_ents = extract_caption_entities(im.caption or "", "")
    purity = topic_purity_score(intent, cap_ents, req_matches, ent_matches)

    return SymbolicMatchInfo(
        required_term_matches=req_matches,
        entity_matches=ent_matches,
        entity_conflicts=ent_conflicts,
        topic_purity=round(purity, 3),
        concept_specificity=spec,
        passes_required_gate=True,
    )


def filter_chapter_candidates(
    intent: ImageIntent,
    images: list[TextbookImage],
) -> list[tuple[TextbookImage, SymbolicMatchInfo]]:
    """
    Symbolic-first candidate pool: all chapter images → ordered hard filters → survivors.
  CLIP/BGE run only on this reduced set.
    """
    survivors: list[tuple[TextbookImage, SymbolicMatchInfo]] = []
    for im in images:
        reject, reason = _hard_concept_filter(intent, im)
        if reject:
            logger.debug("[SYMBOLIC] rejected %s — %s", im.file_name, reason)
            continue
        disk_path = image_disk_path(im.textbook_upload_id, im.file_name, upload=getattr(im, "upload", None))
        if not image_has_visible_content(disk_path):
            continue
        spec = _concept_specificity_score(intent, im)
        sym = _build_symbolic_match_info(intent, im, spec)
        survivors.append((im, sym))
    return survivors


def _bge_scores_batch(
    intent_text: str,
    candidates: list[TextbookImage],
) -> list[float]:
    """Batch BGE cosine similarity between intent_text and each figure's caption."""
    try:
        from app.services.vector_service import _get_embedding_model, is_embedding_model_loaded

        if not is_embedding_model_loaded() or not candidates:
            return [0.0] * len(candidates)
        model = _get_embedding_model()
        if model is None:
            return [0.0] * len(candidates)
    except Exception:
        return [0.0] * len(candidates)

    try:
        qv = np.array(model.embed_query(intent_text), dtype=np.float32)
    except Exception:
        return [0.0] * len(candidates)

    from app.services.image_service.content_kind_retrieval import asset_retrieval_text

    texts = [(asset_retrieval_text(im))[:1600] or im.file_name for im in candidates]
    try:
        mat = np.array(model.embed_documents(texts), dtype=np.float32)
        sims = _cosine_matrix(qv, mat)
        return [max(0.0, float(s)) * 100.0 for s in sims]
    except Exception as exc:
        logger.debug("BGE batch scoring skipped: %s", exc)
        return [0.0] * len(candidates)


def _semantic_bonus(
    query: str,
    context_excerpt: str,
    candidates: list[TextbookImage],
) -> list[float]:
    """BGE semantic bonus using query+context (kept for multimodal path compatibility)."""
    return _bge_scores_batch(query + "\n" + context_excerpt[:1200], candidates)


def _pedagogy_score(
    intent: "ImageIntent",
    im: TextbookImage,
    pages_by_upload: dict[str, list[int]],
    bge_score: float,
    *,
    clip_score: float = 0.0,
    symbolic: SymbolicMatchInfo | None = None,
) -> tuple[float, dict]:
    """
    Symbolic-first composite pedagogy score (0-100).

    Weights:
      0.35 * concept_specificity
      0.25 * section_match
      0.20 * caption_match (BGE)
      0.10 * image_type_score
      0.05 * salience
      0.03 * clip_similarity
      0.02 * page_proximity

    Entity boost (+25) / conflict penalty (-25 per conflict) applied after weighting.
    Topic purity scales concept component when < 0.5.
    """
    s_concept  = _concept_specificity_score(intent, im)
    s_caption  = bge_score
    s_section  = _section_overlap_score(intent, im)
    s_type     = _type_and_salience_score(intent, im)
    raw_prox   = _page_proximity_score(
        str(im.textbook_upload_id), im.page_index, pages_by_upload
    )
    s_page     = min(raw_prox / 24.0 * 100.0, 100.0)
    s_salience = _get_salience(im) * 100.0
    # clip_score: cosine similarity 0-1 from caller
    s_clip = min(100.0, max(0.0, float(clip_score) * 100.0))

    purity = symbolic.topic_purity if symbolic else 0.0
    if purity > 0 and purity < 0.5:
        s_concept *= (0.5 + purity)  # down-weight weak topic purity

    total = (
        0.35 * s_concept +
        0.25 * s_section +
        0.20 * s_caption +
        0.10 * s_type +
        0.05 * s_salience +
        0.03 * s_clip +
        0.02 * s_page
    )

    entity_boost = 0.0
    if isinstance(intent, ImageIntent):
        cap_norm = _get_caption_normalized(im)
        full_text = f"{cap_norm} {im.page_text_snippet or ''}".lower()
        _, conflicts, ent_net = entity_match_score(intent, full_text)
        entity_boost = ent_net
        total += entity_boost

    img_type = _get_image_type(im)
    ed_role  = _get_educational_role(im)

    req_matches = symbolic.required_term_matches if symbolic else []
    ent_matches = symbolic.entity_matches if symbolic else []
    ent_conflicts = symbolic.entity_conflicts if symbolic else []

    breakdown: dict = {
        "concept_specificity": round(s_concept, 1),
        "caption_match":       round(s_caption, 1),
        "section_match":       round(s_section, 1),
        "image_type_score":    round(s_type, 1),
        "page_proximity":      round(raw_prox, 1),
        "clip_similarity":     round(float(clip_score), 4),
        "pedagogical_salience": round(_TYPE_PEDAGOGICAL_SALIENCE.get(img_type, 50.0) / 10.0, 1),
        "educational_role":    ed_role,
        "image_type":          img_type,
        "topic_purity":        round(purity, 3),
        "required_term_matches": req_matches,
        "entity_matches":      ent_matches,
        "entity_conflicts":    ent_conflicts,
        "entity_boost":        round(entity_boost, 1),
        "total":               round(total, 1),
    }
    return max(0.0, total), breakdown


# ---------------------------------------------------------------------------
# MMR diversity + strict count selection
# ---------------------------------------------------------------------------

def _mmr_select(
    scored: list[tuple[float, TextbookImage, dict]],
    *,
    max_n: int,
    min_score: float,
    bge_model_fn: Any | None = None,
) -> list[tuple[float, TextbookImage, dict]]:
    """
    Maximal Marginal Relevance selection for diversity.
    scored: list of (total_score, image, breakdown), descending by total_score.
    """
    if not scored:
        return []

    cap_texts = [
        (_get_caption_normalized(im) + " " + (im.page_text_snippet or ""))[:800]
        for _, im, _ in scored
    ]

    cap_vecs: np.ndarray | None = None
    try:
        from app.services.vector_service import _get_embedding_model, is_embedding_model_loaded

        if is_embedding_model_loaded():
            model = _get_embedding_model()
            if model:
                raw = model.embed_documents(cap_texts)
                mat = np.array(raw, dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms = np.where(norms < 1e-9, 1e-9, norms)
                cap_vecs = mat / norms
    except Exception:
        pass

    selected: list[tuple[float, TextbookImage, dict]] = []
    remaining = list(range(len(scored)))

    while remaining and len(selected) < max_n:
        if cap_vecs is not None and selected:
            sel_idx = [scored.index(s) for s in selected]
            sel_vecs = cap_vecs[sel_idx]

            best_idx = None
            best_mmr = -1e9
            for r in remaining:
                rel = scored[r][0]
                if rel < min_score:
                    continue
                sims = cap_vecs[r] @ sel_vecs.T
                max_sim = float(sims.max()) if len(sims) > 0 else 0.0
                mmr = _MMR_LAMBDA * rel - (1.0 - _MMR_LAMBDA) * max_sim * 100.0
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = r
            if best_idx is None:
                break
            selected.append(scored[best_idx])
            remaining.remove(best_idx)
        else:
            r = remaining[0]
            if scored[r][0] < min_score:
                break
            selected.append(scored[r])
            remaining.pop(0)

    return selected


def _strict_select_with_debug(
    scored_ped: list[tuple[float, TextbookImage, dict]],
    *,
    min_score: float,
    max_primary: int = 2,
    extra_score_ratio: float = 0.92,
    max_total: int = 4,
) -> tuple[list[tuple[float, TextbookImage, dict]], list[dict]]:
    """
    Strict pedagogical selection:
      - Primary: up to max_primary images above min_score
      - Extras: only if score >= primary_top * extra_score_ratio
      - Total cap: max_total

    Returns (selected_items, all_debug_log_entries).

    Debug log entries follow this schema:
      {image, caption, concept_specificity, caption_match, section_match,
       image_type_score, clip_similarity, page_proximity, pedagogical_salience,
       educational_role, final_score, selected, rejection_reason}
    """
    if not scored_ped:
        return [], []

    above_min = [(s, im, bd) for s, im, bd in scored_ped if s >= min_score]
    if not above_min and scored_ped:
        # Graceful: at least consider top image even below threshold
        above_min = scored_ped[:1]

    primary = above_min[:max_primary]
    top_score = primary[0][0] if primary else 0.0
    extras = [
        item for item in above_min[max_primary:max_total]
        if item[0] >= top_score * extra_score_ratio
    ]
    selected = primary + extras
    selected_keys = {(im.textbook_upload_id, im.file_name) for _, im, _ in selected}

    debug_logs: list[dict] = []
    for s, im, bd in scored_ped:
        key = (im.textbook_upload_id, im.file_name)
        is_sel = key in selected_keys
        if is_sel:
            reason = None
        elif s < min_score:
            reason = f"below_min_score ({s:.1f} < {min_score:.1f})"
        elif primary and s < top_score * extra_score_ratio:
            reason = f"below_extra_ratio ({s:.1f} < {top_score * extra_score_ratio:.1f})"
        else:
            reason = "count_limit"

        debug_logs.append({
            "image":                 im.file_name,
            "caption":               (im.caption or "")[:120],
            "concept_specificity":   bd.get("concept_specificity", 0),
            "required_term_matches": bd.get("required_term_matches", []),
            "entity_matches":        bd.get("entity_matches", []),
            "entity_conflicts":      bd.get("entity_conflicts", []),
            "topic_purity":          bd.get("topic_purity", 0),
            "section_match":         bd.get("section_match", 0),
            "caption_match":         bd.get("caption_match", 0),
            "image_type_score":      bd.get("image_type_score", 0),
            "clip_similarity":       bd.get("clip_similarity", 0),
            "page_proximity":        bd.get("page_proximity", 0),
            "pedagogical_salience":  bd.get("pedagogical_salience", 0),
            "educational_role":      bd.get("educational_role", "unknown"),
            "hard_rejected":         bd.get("hard_rejected", False),
            "final_score":           round(s, 1),
            "selected":              is_sel,
            "rejection_reason":      reason,
        })

    return selected, debug_logs


# ---------------------------------------------------------------------------
# Selection + payload helpers
# ---------------------------------------------------------------------------

def _selection_thresholds(requests_images: bool) -> dict[str, float]:
    from app.config import (
        MIN_CLIP_IMAGE_SIMILARITY,
        MIN_IMAGE_RELATIVE_TO_TOP,
        MIN_IMAGE_RELEVANCE_SCORE,
        MIN_TOPIC_SCORE_GENERIC_ASSET,
    )

    if requests_images:
        return {
            "min_score": min(MIN_IMAGE_RELEVANCE_SCORE, 18.0),
            "min_relative": min(MIN_IMAGE_RELATIVE_TO_TOP, 0.55),
            "min_clip": min(MIN_CLIP_IMAGE_SIMILARITY, 0.16),
            "min_topic_generic": min(MIN_TOPIC_SCORE_GENERIC_ASSET, 3.0),
        }
    return {
        "min_score": MIN_IMAGE_RELEVANCE_SCORE,
        "min_relative": MIN_IMAGE_RELATIVE_TO_TOP,
        "min_clip": MIN_CLIP_IMAGE_SIMILARITY,
        "min_topic_generic": MIN_TOPIC_SCORE_GENERIC_ASSET,
    }


def _payload_row(
    im: TextbookImage,
    score: float,
    clip_similarity: float | None = None,
    *,
    subtopic: str | None = None,
) -> dict:
    from app.services.image_service.content_kind_retrieval import get_content_kind
    from app.services.image_service.pdf_figure_context import resolve_display_caption

    from app.services.image_service.pdf_figure_context import _CAPTION_FIG_PREFIX_RE

    effective_caption = resolve_display_caption(im, subtopic=subtopic)
    if getattr(im, "figure_number", None):
        effective_caption = _CAPTION_FIG_PREFIX_RE.sub("", effective_caption).strip(" .:;-")
    row = {
        "url": f"/auth/catalog/textbook-images/{im.textbook_upload_id}/{im.file_name}",
        "caption": effective_caption,
        "page": int(im.page_index) + 1,
        "textbook_upload_id": str(im.textbook_upload_id),
        "relevance": round(score, 2),
        "content_kind": get_content_kind(im),
        "file_name": im.file_name,
    }
    if clip_similarity is not None:
        row["clip_similarity"] = round(clip_similarity, 4)
    if getattr(im, "concept_tags", None):
        row["concept_tags"] = [t for t in im.concept_tags.split("|") if t]
    if getattr(im, "title", None):
        row["title"] = im.title
    if getattr(im, "figure_number", None):
        row["figure_number"] = im.figure_number
    structured = getattr(im, "structured_content", None)
    kind = get_content_kind(im)
    if structured and kind in ("formula", "table"):
        row["structured_content"] = structured.strip()[:2000]
        row["content_kind"] = kind
    return row


def _select_relevant_images(
    scored: list[tuple[float, TextbookImage, float, float]],
    *,
    max_n: int,
    thresholds: dict[str, float] | None = None,
    pages_by_upload: dict[str, list[int]] | None = None,
) -> list[dict]:
    """
    Backward-compat selector used by multimodal path.
    Each item: (total_score, image, topic_score, clip_similarity_or_-1).
    """
    if not scored:
        return []

    th = thresholds or _selection_thresholds(False)
    min_score = th["min_score"]
    min_relative = th["min_relative"]
    min_clip = th["min_clip"]
    min_topic_generic = th["min_topic_generic"]

    scored = sorted(scored, key=lambda x: -x[0])
    top_score = scored[0][0]
    seen: set[tuple[int, str]] = set()
    out: list[dict] = []
    prev_score: float | None = None

    for total, im, topic, clip_raw in scored:
        if total < min_score:
            break
        if top_score > 0 and total < top_score * min_relative:
            break
        if prev_score is not None and total < prev_score * 0.72:
            break
        if clip_raw >= 0 and clip_raw < min_clip:
            continue

        page_prox = 0.0
        if pages_by_upload:
            page_prox = _page_proximity_score(str(im.textbook_upload_id), im.page_index, pages_by_upload)
        if _is_generic_asset_filename(im.file_name):
            if topic < min_topic_generic and page_prox < 14.0:
                continue

        key = (im.textbook_upload_id, im.file_name)
        if key in seen:
            continue
        disk_path = image_disk_path(im.textbook_upload_id, im.file_name, upload=getattr(im, "upload", None))
        if not image_has_visible_content(disk_path):
            continue

        seen.add(key)
        clip_sim = clip_raw if clip_raw >= 0 else None
        out.append(_payload_row(im, total, clip_sim))
        prev_score = total
        if len(out) >= max_n:
            break

    return out


def _symbolically_rejected(intent: "ImageIntent | None", im: TextbookImage) -> bool:
    """
    True if *im* fails the symbolic concept/entity/geographic hard filter.

    Applied to page-proximity and citation fallbacks so unrelated page-mates
    (e.g. a Gharial photo on the same page as Thar Desert text) cannot leak in.
    """
    if intent is None or not isinstance(intent, ImageIntent):
        return False
    try:
        reject, reason = _hard_concept_filter(intent, im)
        if reject:
            logger.debug("[FALLBACK-SYMBOLIC] rejected %s — %s", im.file_name, reason)
        return reject
    except Exception:
        return False


def _page_proximity_fallback(
    images: list[TextbookImage],
    pages_by_upload: dict[str, list[int]],
    retrieval_text: str,
    *,
    max_n: int,
    thresholds: dict[str, float],
    intent: "ImageIntent | None" = None,
) -> list[dict]:
    """Use figures from the same pages as the RAG text chunks when main gate returns nothing."""
    from app.config import MIN_PAGE_PROXIMITY_FALLBACK

    if not images or not pages_by_upload:
        return []

    scored: list[tuple[float, TextbookImage, float, float]] = []
    for im in images:
        if _symbolically_rejected(intent, im):
            continue
        prox = _page_proximity_score(str(im.textbook_upload_id), im.page_index, pages_by_upload)
        if prox < MIN_PAGE_PROXIMITY_FALLBACK:
            continue
        topic = _topic_keyword_score(retrieval_text, "", im)
        scored.append((prox + topic * 1.5, im, topic, -1.0))

    return _select_relevant_images(
        scored, max_n=max_n, thresholds=thresholds, pages_by_upload=pages_by_upload
    )


def _guaranteed_citation_figures(
    images: list[TextbookImage],
    pages_by_upload: dict[str, list[int]],
    *,
    max_n: int,
    query: str = "",
    intent: "ImageIntent | None" = None,
    subtopic: str | None = None,
) -> list[dict]:
    """
    Last-resort: visible figures nearest to RAG citation pages that share
    at least one keyword with the query.  Pure page-proximity picks
    unrelated figures (e.g. Gharial when asking about Thar Desert), so we
    require both symbolic concept survival AND minimum topic overlap.
    """
    retrieval_text = query.strip()
    require_keyword = bool(retrieval_text)

    ranked: list[tuple[float, TextbookImage]] = []
    for im in images:
        if _symbolically_rejected(intent, im):
            continue
        prox = _page_proximity_score(str(im.textbook_upload_id), im.page_index, pages_by_upload)
        if prox < 6.0:
            continue
        if require_keyword:
            topic = _topic_keyword_score(retrieval_text, "", im)
            if topic <= 0.0:
                continue
            score = prox + topic * 2.0
        else:
            score = prox
        disk_path = image_disk_path(im.textbook_upload_id, im.file_name, upload=getattr(im, "upload", None))
        if not image_has_visible_content(disk_path):
            continue
        ranked.append((score, im))
    ranked.sort(key=lambda x: -x[0])
    return [
        _payload_row(im, score + 20.0, None, subtopic=subtopic)
        for score, im in ranked[:max_n]
    ]


def finalize_related_images(
    scored: list[tuple[float, TextbookImage, float, float]],
    *,
    query: str,
    answer_text: str = "",
    max_n: int,
    context_excerpt: str = "",
    pages_by_upload: dict[str, list[int]] | None = None,
    pool_images: list[TextbookImage] | None = None,
    intent: "ImageIntent | None" = None,
) -> list[dict]:
    """Primary selection + page-proximity fallback + citation guarantee.

    Used by multimodal_image_retrieval.py; answer_text is accepted for
    backward-compat but not used in scoring.  *intent* enables symbolic
    rejection inside the fallback tiers so off-topic page-mates cannot leak.
    """
    wants = _query_requests_images(query)
    th = _selection_thresholds(wants)
    pages = pages_by_upload or {}
    retrieval_text = _build_retrieval_text(query, context_excerpt)

    if intent is None and query:
        try:
            intent = extract_image_intent(query, None)
        except Exception:
            intent = None

    out = _select_relevant_images(scored, max_n=max_n, thresholds=th, pages_by_upload=pages)
    if not out and pool_images:
        if intent is not None:
            from app.services.image_service.content_kind_retrieval import resolve_content_kind_pool

            pool_images = resolve_content_kind_pool(pool_images, intent)
        out = _page_proximity_fallback(
            pool_images, pages, retrieval_text, max_n=max_n, thresholds=th, intent=intent
        )
        if out:
            logger.info("[IMAGES] page-proximity fallback kept %d (query=%r)", len(out), (query or "")[:60])
    elif out:
        logger.info(
            "[IMAGES] selected %d (top=%.1f, wants_images=%s, query=%r)",
            len(out), out[0].get("relevance", 0), wants, (query or "")[:60],
        )
    if not out and pool_images and pages:
        out = _guaranteed_citation_figures(
            pool_images, pages, max_n=max_n, query=query, intent=intent
        )
        if out:
            logger.info("[IMAGES] citation-page guarantee kept %d (query=%r)", len(out), (query or "")[:60])
    elif not out:
        logger.info("[IMAGES] none passed gates (query=%r)", (query or "")[:60])
    return out


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_uploads(db: Session, chapter_ids: list[str] | None) -> dict[int, TextbookUpload]:
    if not chapter_ids:
        return {}
    ids: list[int] = []
    for raw in chapter_ids:
        try:
            ids.append(int(raw))
        except Exception:
            continue
    if not ids:
        return {}
    rows = list(db.scalars(select(TextbookUpload).where(TextbookUpload.id.in_(ids))))
    return {r.id: r for r in rows}


def _list_images(db: Session, upload_ids: list[int]) -> list[TextbookImage]:
    if not upload_ids:
        return []
    return list(
        db.scalars(select(TextbookImage).where(TextbookImage.textbook_upload_id.in_(upload_ids))).all()
    )


def _attach_upload_refs(
    images: list[TextbookImage],
    uploads: dict[int, TextbookUpload],
) -> None:
    """Attach upload rows so PDF text backfill can resolve file paths."""
    for im in images:
        up = uploads.get(im.textbook_upload_id)
        if up is not None:
            im.upload = up


# ---------------------------------------------------------------------------
# Primary pedagogy ranker
# ---------------------------------------------------------------------------

def _pedagogy_rank(
    intent: "ImageIntent",
    images: list[TextbookImage],
    uploads: dict[int, TextbookUpload],
    chapter_hints: list[str],
    pages_by_upload: dict[str, list[int]],
    *,
    max_n: int,
    min_ped_score: float,
    clip_sims: dict[str, float] | None = None,
    rag_docs: list[Any] | None = None,
) -> list[dict]:
    """
    Mandatory stage → supporting rank stage → adaptive assembly.
    Mandatory figures never enter the scoring competition.
    """
    from app.config import MAX_GROUNDED_IMAGES

    from app.services.image_service.content_kind_retrieval import resolve_content_kind_pool

    kind_pool = resolve_content_kind_pool(images, intent)
    filtered = filter_chapter_candidates(intent, kind_pool)
    if not filtered:
        logger.info(
            "[RANK] 0 candidates after symbolic filters (query=%r kinds=%s pool=%d)",
            intent.query[:60],
            getattr(intent, "preferred_content_kinds", ["figure"]),
            len(kind_pool),
        )
        return []

    stages = run_topic_centric_retrieval(
        intent,
        filtered,
        rag_docs,
        pages_by_upload,
        concept_specificity_fn=_concept_specificity_score,
        page_proximity_fn=_page_proximity_score,
        clip_sims=clip_sims,
        min_supporting_score=min_ped_score,
        max_images=min(max_n, MAX_GROUNDED_IMAGES),
    )

    rows = assemble_payload_rows(stages)
    logger.info(
        "[RANK] query=%r | kinds=%s | pool=%d | filtered=%d | mandatory=%d | supporting=%d | rejected=%d",
        intent.query[:60],
        getattr(intent, "preferred_content_kinds", ["figure"]),
        len(kind_pool),
        len(filtered),
        len(stages.mandatory_images),
        len(stages.supporting_images),
        len(stages.rejected),
    )
    for im, rec in stages.mandatory_images:
        logger.info(
            "[RANK] MANDATORY %s anchor=%s ctx=%.0f | %s",
            im.file_name, rec.topic_anchor, rec.figure_context_score, rec.selected_reason,
        )
    for im, rec in stages.supporting_images:
        logger.info(
            "[RANK] SUPPORTING %s score=%.1f ctx=%.0f cap=%.0f | %s",
            im.file_name, rec.final_score, rec.figure_context_score, rec.caption_score, rec.selected_reason,
        )
    for rec in stages.rejected[:8]:
        logger.debug("[RANK] REJECTED %s | %s", rec.file_name, rec.rejected_reason)

    return [
        _payload_row(im, score, bd.get("clip_score"))
        for im, score, bd in rows
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def related_images_payload(
    chapter_ids: list[str] | None,
    chapter_names: list[str],
    chapter_single: str,
    query: str,
    retrieved_docs: list[Any],
    *,
    answer_text: str = "",
    top_n: int = _DEFAULT_TOP,
    conversation_context: Any | None = None,
    ensure_extract: bool = True,
) -> list[dict]:
    """
    Heuristic (non-CLIP) precision pedagogy ranker.

    answer_text is accepted for API backward-compat but never used for scoring.
    Set ensure_extract=False to use already-extracted figures only (lesson planner).
    """
    from app.config import MIN_FINAL_SCORE

    if not chapter_ids:
        return []

    hints = [h for h in (chapter_names or []) if h and h.strip()]
    if chapter_single and chapter_single.strip():
        hints.append(chapter_single.strip())

    # Build structured educational intent from question + chunks (NO LLM answer)
    intent = extract_image_intent(
        query, retrieved_docs, conversation_context=conversation_context
    )
    context_excerpt = _context_excerpt_from_docs(retrieved_docs)
    pages_by_upload = _pages_by_upload_from_docs(retrieved_docs)

    with SessionLocal() as db:
        uploads = _load_uploads(db, chapter_ids)
        if not uploads:
            return []

        if ensure_extract:
            for u in uploads.values():
                try:
                    ensure_textbook_images_extracted(db, u)
                    from app.services.image_service.textbook_image_extraction import ensure_figure_context_bge_indexed

                    ensure_figure_context_bge_indexed(db, u)
                except Exception as exc:
                    logger.debug("ensure_textbook_images_extracted: %s", exc)

        images = _list_images(db, list(uploads.keys()))
        if not images:
            return []

        _attach_upload_refs(images, uploads)
        min_score = MIN_FINAL_SCORE * 0.85 if intent.requested_visuals else MIN_FINAL_SCORE

        out = _pedagogy_rank(
            intent, images, uploads, hints, pages_by_upload,
            max_n=top_n, min_ped_score=min_score,
            rag_docs=retrieved_docs,
        )
        if out:
            return out

        from app.config import DISABLE_PAGE_PROXIMITY_FALLBACK
        if DISABLE_PAGE_PROXIMITY_FALLBACK:
            logger.info("[IMAGES] no grounded figures; page fallback disabled (query=%r)", query[:60])
            return []

        retrieval_text = _build_retrieval_text(query, context_excerpt)
        th = _selection_thresholds(intent.requested_visuals)
        from app.services.image_service.content_kind_retrieval import resolve_content_kind_pool

        kind_pool = resolve_content_kind_pool(images, intent)
        fb = _page_proximity_fallback(
            kind_pool, pages_by_upload, retrieval_text, max_n=top_n, thresholds=th, intent=intent
        )
        if fb:
            logger.info("[IMAGES] page-proximity fallback (query=%r)", query[:60])
            return fb
        return []


def related_images_for_query(
    text_collection_name: str,
    chapter_ids: list[str] | None,
    chapter_names: list[str],
    chapter_single: str,
    query: str,
    retrieved_docs: list[Any],
    *,
    answer_text: str = "",
    top_n: int = _DEFAULT_TOP,
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """
    Primary entry for chat/voice: multimodal CLIP retrieval when enabled,
    else pedagogy heuristic ranker.

    answer_text is accepted for API backward-compat but excluded from scoring.
    """
    from app.config import USE_MULTIMODAL_IMAGE_RETRIEVAL
    from app.services.conversation_context import (
        resolve_conversation_context,
        should_retrieve_images,
    )

    conv = resolve_conversation_context(
        query,
        conversation_history=conversation_history,
        chapter=chapter_single,
    )
    if not should_retrieve_images(conv, chapter_ids=chapter_ids):
        logger.info("[IMAGES] visual gate off: %s", conv.to_debug())
        return []

    effective_query = conv.retrieval_query or query
    logger.debug(
        "[IMAGES] retrieve query=%r mode=%s visual=%s",
        effective_query[:80],
        conv.response_mode.value,
        conv.visual_intent.value,
    )

    if USE_MULTIMODAL_IMAGE_RETRIEVAL and chapter_ids:
        try:
            from app.services.image_service.multimodal_image_retrieval import related_images_multimodal

            hits = related_images_multimodal(
                text_collection_name,
                chapter_ids,
                chapter_names,
                chapter_single,
                effective_query,
                retrieved_docs,
                answer_text="",  # intentionally blank — no LLM contamination
                top_n=top_n,
                conversation_context=conv,
            )
            if hits:
                return hits
        except Exception as exc:
            logger.warning("Multimodal image retrieval failed, using fallback: %s", exc)

    return related_images_payload(
        chapter_ids,
        chapter_names,
        chapter_single,
        effective_query,
        retrieved_docs,
        answer_text="",  # intentionally blank
        top_n=top_n,
        conversation_context=conv,
    )


def early_related_images_for_query(
    chapter_ids: list[str] | None,
    retrieved_docs: list[Any],
    query: str = "",
    *,
    top_n: int = _DEFAULT_TOP,
    conversation_history: list[dict] | None = None,
    chapter_single: str = "",
) -> list[dict]:
    """
    Fast early-stream payload: nearest visible figures from citation pages
    that share at least one keyword with the query.

    Used to render images while answer tokens are still streaming.
    """
    from app.services.conversation_context import (
        resolve_conversation_context,
        should_retrieve_images,
    )

    conv = resolve_conversation_context(
        query,
        conversation_history=conversation_history,
        chapter=chapter_single,
    )
    if not should_retrieve_images(conv, chapter_ids=chapter_ids):
        return []

    if not chapter_ids:
        return []
    with SessionLocal() as db:
        uploads = _load_uploads(db, chapter_ids)
        if not uploads:
            return []
        for u in uploads.values():
            try:
                ensure_textbook_images_extracted(db, u)
            except Exception:
                continue
        images = _list_images(db, list(uploads.keys()))
        if not images:
            return []
        _attach_upload_refs(images, uploads)
        pages_by_upload = _pages_by_upload_from_docs(retrieved_docs)
        if not pages_by_upload:
            return []
        effective_query = conv.retrieval_query or query
        intent = extract_image_intent(effective_query, retrieved_docs, conversation_context=conv)
        from app.services.image_service.content_kind_retrieval import resolve_content_kind_pool

        images = resolve_content_kind_pool(images, intent)
        subtopic_hint = (intent.core_concept or intent.intent_text or "").strip() or None
        from app.services.image_service.figure_context_gates import (
            _BROAD_DEFINITION_CORES,
            is_core_definition_figure,
        )

        core = (intent.core_concept or "").lower()
        if intent.query_type == "concept_definition" and core in _BROAD_DEFINITION_CORES:
            defs = [
                im
                for im in images
                if is_core_definition_figure(im, core)
                and not _symbolically_rejected(intent, im)
            ]
            if defs:
                defs.sort(
                    key=lambda im: -_page_proximity_score(
                        str(im.textbook_upload_id), im.page_index, pages_by_upload
                    )
                )
                return [
                    _payload_row(im, 72.0, None, subtopic=subtopic_hint)
                    for im in defs[:top_n]
                    if image_has_visible_content(
                        image_disk_path(im.textbook_upload_id, im.file_name, upload=getattr(im, "upload", None))
                    )
                ]
        return _guaranteed_citation_figures(
            images,
            pages_by_upload,
            max_n=top_n,
            query=effective_query,
            intent=intent,
            subtopic=subtopic_hint,
        )


# ---------------------------------------------------------------------------
# Debug: expose per-figure score breakdown
# ---------------------------------------------------------------------------

def debug_rank_figures(
    chapter_ids: list[str] | None,
    chapter_names: list[str],
    chapter_single: str,
    query: str,
    retrieved_docs: list[Any],
    *,
    top_n: int = 24,
) -> list[dict]:
    """
    Per-figure debug breakdown for /auth/debug/figure-rank.

    Uses mandatory / supporting stages; includes symbolic rejects and stage labels.
    """
    from app.config import MIN_FINAL_SCORE, MAX_GROUNDED_IMAGES

    if not chapter_ids:
        return []

    intent = extract_image_intent(query, retrieved_docs)
    pages_by_upload = _pages_by_upload_from_docs(retrieved_docs)

    with SessionLocal() as db:
        uploads = _load_uploads(db, chapter_ids)
        if not uploads:
            return []
        for u in uploads.values():
            try:
                ensure_textbook_images_extracted(db, u)
                from app.services.image_service.textbook_image_extraction import ensure_figure_context_bge_indexed

                ensure_figure_context_bge_indexed(db, u)
            except Exception:
                pass
        images = _list_images(db, list(uploads.keys()))
        if not images:
            return []

    visible = [
        im for im in images
        if image_has_visible_content(
            image_disk_path(im.textbook_upload_id, im.file_name, upload=getattr(im, "upload", None))
        )
    ]
    filtered = filter_chapter_candidates(intent, visible)
    filtered_ids = {str(im.id) for im, _ in filtered}

    stages = run_topic_centric_retrieval(
        intent,
        filtered,
        retrieved_docs,
        pages_by_upload,
        concept_specificity_fn=_concept_specificity_score,
        page_proximity_fn=_page_proximity_score,
        clip_sims=None,
        min_supporting_score=MIN_FINAL_SCORE,
        max_images=MAX_GROUNDED_IMAGES,
    )

    selected_ids: dict[str, str] = {}
    for im, rec in stages.mandatory_images:
        selected_ids[str(im.id)] = rec.selected_reason or "mandatory_stage"
    for im, rec in stages.supporting_images:
        selected_ids[str(im.id)] = rec.selected_reason or "supporting_stage"

    ranked_debug: dict[str, dict] = {}
    for im, rec in stages.mandatory_images + stages.supporting_images:
        ranked_debug[str(im.id)] = record_to_debug_dict(rec, im)
    for rec in stages.rejected:
        ranked_debug[rec.image_id] = {
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
        }

    rows: list[dict] = []
    for im in visible:
        sym_reject, sym_reason = _hard_concept_filter(intent, im)
        if sym_reject:
            rows.append({
                "image_id": str(im.id),
                "file_name": im.file_name,
                "stage": "symbolic_filter",
                "mandatory": False,
                "selected_reason": None,
                "rejected_reason": sym_reason,
                "final_score": 0.0,
                "url": f"/auth/catalog/textbook-images/{im.textbook_upload_id}/{im.file_name}",
            })
            continue

        if str(im.id) not in filtered_ids:
            rows.append({
                "image_id": str(im.id),
                "file_name": im.file_name,
                "stage": "symbolic_filter",
                "mandatory": False,
                "rejected_reason": "not_in_filtered_pool",
                "final_score": 0.0,
                "url": f"/auth/catalog/textbook-images/{im.textbook_upload_id}/{im.file_name}",
            })
            continue

        dbg = ranked_debug.get(str(im.id), {})
        stage = "ranking"
        if str(im.id) in selected_ids:
            stage = "mandatory" if dbg.get("mandatory") else "supporting"
        elif dbg.get("rejected_reason"):
            stage = "ranking_rejected"

        rows.append({
            **dbg,
            "image": im.file_name,
            "caption": (im.caption or "")[:200],
            "page": im.page_index + 1,
            "stage": stage,
            "selected_reason": selected_ids.get(str(im.id)) or dbg.get("selected_reason"),
            "rejected_reason": dbg.get("rejected_reason"),
            "url": f"/auth/catalog/textbook-images/{im.textbook_upload_id}/{im.file_name}",
        })

    rows.sort(
        key=lambda x: (
            0 if x.get("stage") == "mandatory" else 1 if x.get("stage") == "supporting" else 2,
            -float(x.get("final_score") or 0),
        )
    )
    return rows[:top_n]
