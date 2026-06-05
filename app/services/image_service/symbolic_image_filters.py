"""
Symbolic (keyword/entity) filters for precision educational image retrieval.

Runs BEFORE semantic (BGE/CLIP) scoring so visually similar but pedagogically
wrong figures never enter the candidate pool.

Pipeline order (see ``apply_symbolic_hard_filters``):
  1. excluded image types
  2. decorative / sidebar role gate
  3. required-term gate
  4. geographic conflict filter
  5. concept purity gate
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.modules.catalog.models import TextbookImage
    from app.services.image_service.image_intent_extractor import ImageIntent

# ---------------------------------------------------------------------------
# Geographic conflict table (query concept → conflicting place/region tokens)
# ---------------------------------------------------------------------------

GEOGRAPHIC_CONFLICTS: dict[str, list[str]] = {
    "thar desert": [
        "ladakh", "himalaya", "himalayas", "glacier", "sikkim", "meghalaya",
        "kashmir", "leh", "lamayuru", "moonland", "yak", "everest", "karakoram",
        "aravalli", "kumbhalgarh", "fort", "northeast", "assam",
    ],
    "thar": [
        "ladakh", "himalaya", "glacier", "sikkim", "meghalaya", "yak", "moonland",
        "kashmir", "leh",
    ],
    "ladakh": [
        "thar", "rajasthan", "jaisalmer", "desert", "sahara", "gobi",
        "meghalaya", "kerala", "tamil", "assam",
    ],
    "moonland": [
        "thar", "rajasthan", "jaisalmer", "sahara", "meghalaya", "kerala",
    ],
    "himalaya": [
        "thar", "desert", "rajasthan", "jaisalmer", "coastal", "beach",
    ],
    "automated weather station": [
        "coconut", "oil", "solidif", "winter example", "cloudy weather",
        "rain example",
    ],
    "weather station": [
        "coconut", "oil", "solidif", "cloudy weather",
    ],
    "desert": [
        "glacier", "himalaya", "rainforest", "mangrove", "monsoon forest",
    ],
}

# Tokens treated as proper nouns / entities when capitalized in source text
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
# Indian geography cues often lowercased in captions
_GEO_ENTITY_HINTS = frozenset({
    "thar", "ladakh", "himalaya", "himalayas", "rajasthan", "jaisalmer",
    "meghalaya", "sikkim", "kashmir", "aravalli", "gangetic", "deccan",
    "moonland", "lamayuru", "leh", "everest", "karakoram", "aravallis",
})

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
    "and", "or", "to", "for", "with", "by", "from", "this", "that", "it",
    "its", "as", "be", "been", "being", "have", "has", "had", "not", "no",
    "explain", "describe", "what", "how", "why", "when", "where", "about",
    "india", "indian", "figure", "diagram", "shows", "showing", "depicts",
})


@dataclass
class SymbolicMatchInfo:
    """Per-image symbolic match details for scoring and debug logs."""

    required_term_matches: list[str] = field(default_factory=list)
    entity_matches: list[str] = field(default_factory=list)
    entity_conflicts: list[str] = field(default_factory=list)
    topic_purity: float = 0.0
    concept_specificity: float = 0.0
    passes_required_gate: bool = True
    hard_rejected: bool = False
    rejection_reason: str | None = None


def tokenize(text: str) -> frozenset[str]:
    tokens = set(re.findall(r"\b[a-z][a-z]+\b", (text or "").lower()))
    return frozenset(tokens - _STOPWORDS)


def extract_educational_entities(question: str, core_concept: str) -> list[str]:
    """
    Lightweight entity extraction (no spaCy): proper nouns + geography hints + phrases.
    """
    entities: list[str] = []
    seen: set[str] = set()

    def _add(e: str) -> None:
        el = e.lower().strip()
        if el and el not in _STOPWORDS and el not in seen and len(el) >= 3:
            seen.add(el)
            entities.append(el)

    # Capitalized phrases from original question
    for m in _PROPER_NOUN_RE.finditer(question or ""):
        _add(m.group(1))

    # Multi-word and distinctive tokens from core concept
    cc = (core_concept or "").lower()
    for phrase in re.findall(r"\b[a-z]+(?:\s+[a-z]+){1,3}\b", cc):
        if phrase not in _STOPWORDS:
            _add(phrase)

    for w in re.findall(r"\b[a-z]{4,}\b", cc):
        if w in _GEO_ENTITY_HINTS or len(w) >= 5:
            _add(w)

    # ALL-CAPS abbreviations
    for abbrev in re.findall(r"\b([A-Z]{2,5})\b", question or ""):
        _add(abbrev)

    return entities


def extract_caption_entities(caption: str, snippet: str = "") -> list[str]:
    """Entities / noun-like tokens present in figure text."""
    text = f"{caption or ''} {snippet or ''}"
    entities: list[str] = []
    seen: set[str] = set()

    def _add(e: str) -> None:
        el = e.lower().strip()
        if el and el not in _STOPWORDS and el not in seen:
            seen.add(el)
            entities.append(el)

    for m in _PROPER_NOUN_RE.finditer(text):
        _add(m.group(1))

    for w in re.findall(r"\b[a-z]{4,}\b", text.lower()):
        if w in _GEO_ENTITY_HINTS:
            _add(w)

    return entities


def _figure_full_text(im: "TextbookImage", caption_normalized: str) -> str:
    sec = (getattr(im, "section_title", None) or "").strip()
    return f"{caption_normalized} {im.page_text_snippet or ''} {sec}".lower()


def _figure_caption_text(im: "TextbookImage", caption_normalized: str) -> str:
    """
    Authoritative descriptor of WHAT the figure depicts: caption + section title.

    Deliberately EXCLUDES ``page_text_snippet`` because that snippet is the
    shared text of the whole page — every figure on the page inherits it, which
    made unrelated page-mates (e.g. a gharial photo on the Thar Desert page)
    falsely match the topic. Page text is only a weak ranking signal, never
    evidence for a hard concept gate.
    """
    sec = (getattr(im, "section_title", None) or "").strip()
    return f"{caption_normalized} {sec}".lower()


def _get_conflicts_for_concept(core_concept: str) -> list[str]:
    cc = (core_concept or "").lower().strip()
    conflicts: list[str] = []
    seen: set[str] = set()
    # Longest keys first
    for key in sorted(GEOGRAPHIC_CONFLICTS.keys(), key=len, reverse=True):
        if key in cc or cc in key:
            for c in GEOGRAPHIC_CONFLICTS[key]:
                if c not in seen:
                    seen.add(c)
                    conflicts.append(c)
    return conflicts


def count_required_term_matches(
    intent: "ImageIntent",
    full_text: str,
) -> tuple[list[str], bool]:
    """
    Returns (matched_terms, passes_strict_gate).

    Strict gate (multi-word core_concept): pass if ANY of:
      A) exact core_concept substring
      B) any multi-word required_term substring
      C) 2+ required terms matched (phrase or distinctive single)
    """
    core = (intent.core_concept or "").lower().strip()
    required = intent.required_terms or []
    matches: list[str] = []

    if core and core in full_text:
        matches.append(core)

    for t in required:
        tl = t.lower()
        if len(t.split()) > 1 and tl in full_text and t not in matches:
            matches.append(t)
        elif len(t.split()) == 1:
            if (len(t) <= 5 and t.isalpha()) or len(t) >= 8:
                if tl in full_text and t not in matches:
                    matches.append(t)

    if len(core.split()) < 2:
        return matches, True  # single-word queries: no strict phrase gate

    passes = bool(
        (core and core in full_text)
        or any(len(m.split()) > 1 for m in matches)
        or len(matches) >= 2
    )
    return matches, passes


def topic_purity_score(
    intent: "ImageIntent",
    caption_entities: list[str],
    required_matches: list[str],
    entity_matches: list[str],
) -> float:
    """
    Fraction of caption entities that align with the educational topic (0–1).
    """
    if not caption_entities:
        return 0.0

    topic_tokens: set[str] = set()
    topic_tokens.update(intent.concept_tokens)
    topic_tokens.update(e.lower() for e in (intent.entities or []))
    topic_tokens.update(m.lower() for m in required_matches)
    topic_tokens.update(e.lower() for e in entity_matches)
    for t in intent.required_terms or []:
        for w in t.lower().split():
            if w not in _STOPWORDS:
                topic_tokens.add(w)

    if not topic_tokens:
        return 0.0

    matched = sum(
        1 for e in caption_entities
        if e.lower() in topic_tokens
        or any(e.lower() in tm or tm in e.lower() for tm in topic_tokens if len(tm) >= 4)
    )
    return min(1.0, matched / max(1, len(caption_entities)))


def entity_match_score(intent: "ImageIntent", full_text: str) -> tuple[list[str], list[str], float]:
    """
    Returns (entity_matches, entity_conflicts, boost_penalty_net).

    +25 if any query entity in caption; -25 per geographic conflict hit.
    """
    entities = intent.entities or []
    matches = [e for e in entities if e.lower() in full_text]
    conflicts_found: list[str] = []
    for c in _get_conflicts_for_concept(intent.core_concept):
        if c in full_text:
            conflicts_found.append(c)

    net = 0.0
    if matches:
        net += 25.0
    net -= 25.0 * len(conflicts_found)
    return matches, conflicts_found, net


def level3_overlap_allowed(
    intent: "ImageIntent",
    tok_match: frozenset[str],
) -> bool:
    """
    Level-3 token overlap is allowed only when:
      - overlap count >= 2, AND
      - at least one token is len>=8 OR is a known entity/proper noun.
    """
    if len(tok_match) < 2:
        return False
    entities = {e.lower() for e in (intent.entities or [])}
    for t in tok_match:
        if len(t) >= 8 or t in entities or t in _GEO_ENTITY_HINTS:
            return True
    return False


def apply_symbolic_hard_filters(
    intent: "ImageIntent",
    im: "TextbookImage",
    *,
    caption_normalized: str,
    image_type: str,
    educational_role: str,
    concept_specificity: float,
    min_topic_purity: float,
    sidebar_min_specificity: float,
) -> SymbolicMatchInfo:
    """
    Run ordered hard filters. Returns SymbolicMatchInfo with rejection_reason set if rejected.
    """
    from app.config import MIN_TOPIC_PURITY
    from app.services.image_service.figure_context_gates import (
        context_supports_topic,
        figure_descriptive_text_for_gates,
        is_core_definition_figure,
        is_minimal_figure_caption,
        is_offtopic_for_broad_definition_query,
        is_tangential_weather_mention,
    )

    caption_text = _figure_caption_text(im, caption_normalized)
    gate_text = figure_descriptive_text_for_gates(im, caption_normalized)
    full_text = _figure_full_text(im, caption_normalized)
    minimal_cap = is_minimal_figure_caption(im.caption)
    info = SymbolicMatchInfo(concept_specificity=concept_specificity)

    qt = getattr(intent, "query_type", "") or ""
    core = getattr(intent, "core_concept", "") or ""
    if is_offtopic_for_broad_definition_query(
        im, gate_text, query_type=qt, core_concept=core
    ):
        info.hard_rejected = True
        info.rejection_reason = "offtopic_broad_definition"
        return info
    if is_tangential_weather_mention(im, gate_text, query_type=qt, core_concept=core):
        info.hard_rejected = True
        info.rejection_reason = "tangential_weather_mention"
        return info

    # ── 1. Excluded image types (zero concept overlap) ───────────────────────
    cap_tokens = tokenize(caption_text)
    gate_tokens = tokenize(gate_text)
    has_concept_overlap = bool(intent.concept_tokens & cap_tokens) or context_supports_topic(
        intent, gate_text
    )
    if is_core_definition_figure(im, getattr(intent, "core_concept", "") or ""):
        has_concept_overlap = True
    if intent.excluded_types and image_type in intent.excluded_types and not has_concept_overlap:
        info.hard_rejected = True
        info.rejection_reason = f"excluded_type={image_type}, no concept overlap in caption"
        return info

    # ── 2. Decorative / sidebar gate ─────────────────────────────────────────
    if educational_role in ("sidebar_example", "decorative"):
        if concept_specificity < sidebar_min_specificity:
            if not (
                context_supports_topic(intent, gate_text)
                or is_core_definition_figure(im, getattr(intent, "core_concept", "") or "")
            ):
                info.hard_rejected = True
                info.rejection_reason = (
                    f"role={educational_role}, specificity={concept_specificity:.1f} "
                    f"< {sidebar_min_specificity}"
                )
                return info

    # ── 3. Required-term gate (multi-word concepts) ────────────────────────────
    req_source = gate_text
    req_matches, passes_gate = count_required_term_matches(intent, req_source)
    info.required_term_matches = req_matches
    info.passes_required_gate = passes_gate
    if not passes_gate:
        info.hard_rejected = True
        info.rejection_reason = "required_term_gate_failed (caption)"
        return info

    # ── 4. Geographic conflict (hard reject on strong conflict) — caption only ─
    _, conflicts, _ = entity_match_score(intent, caption_text)
    info.entity_matches = [e for e in (intent.entities or []) if e.lower() in caption_text]
    info.entity_conflicts = conflicts
    if conflicts and concept_specificity < 50.0:
        info.hard_rejected = True
        info.rejection_reason = f"geographic_conflict={conflicts[:3]}"
        return info

    # ── 5. Topic purity gate — caption entities only (no page snippet) ────────
    cap_entities = extract_caption_entities(im.caption or "", "")
    purity = topic_purity_score(intent, cap_entities, req_matches, info.entity_matches)
    info.topic_purity = round(purity, 3)
    threshold = min_topic_purity if min_topic_purity > 0 else MIN_TOPIC_PURITY
    core_words = len((intent.core_concept or "").split())
    if core_words >= 2 and purity < threshold and concept_specificity < 40.0:
        if not (minimal_cap and context_supports_topic(intent, gate_text)):
            info.hard_rejected = True
            info.rejection_reason = f"topic_purity={purity:.2f} < {threshold}"
            return info

    return info
