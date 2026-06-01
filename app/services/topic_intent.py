"""
Topic intent extraction for pedagogy-centric image retrieval.

Derives TopicIntent from the student question and optional RAG chunk metadata.
Deliberately avoids the LLM-generated answer to prevent retrieval contamination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Image-type preference / exclusion tables
# ---------------------------------------------------------------------------

# (regex pattern, [image_type, ...]) — checked in order; first match wins per type
_TYPE_PREFERENCE_RULES: list[tuple[str, list[str]]] = [
    (r"\b(satellite|aerial|from\s+above|top.?view|remote\s+sensing)\b", ["satellite_image"]),
    (r"\b(map|distribution|region|boundary|border|territory|location\s+of)\b", ["map", "satellite_image"]),
    (r"\b(desert|sand\s+dune|dunes|arid|thar|rajasthan|sahara|gobi)\b", ["landscape", "map"]),
    (r"\b(mountain|himalaya|peak|glacier|valley|range|everest|karakoram|alps)\b", ["landscape", "satellite_image", "diagram"]),
    (r"\b(river|lake|delta|plain|gangetic|flood|waterfall|falls|estuary)\b", ["landscape", "map", "satellite_image"]),
    (r"\b(coast|beach|island|sea|ocean|gulf|bay|peninsula|archipelago)\b", ["landscape", "satellite_image", "map"]),
    (r"\b(forest|vegetation|biome|ecosystem|biodiversity|jungle|mangrove)\b", ["landscape", "wildlife"]),
    (r"\b(agriculture|crop|farming|irrigation|paddy|wheat|plantation)\b", ["landscape", "diagram"]),
    (r"\b(animal|wildlife|species|bird|tiger|lion|elephant|gharial|crocodile|peacock|macaque|cobra|deer|bear|yak|camel|reptile|insect|fish)\b", ["wildlife"]),
    (r"\b(diagram|process|formation|cycle|structure|cross.?section|flowchart)\b", ["diagram"]),
    (r"\b(graph|chart|table|bar|pie|line\s+graph|data|statistics|percentage)\b", ["chart"]),
    (r"\b(fort|palace|temple|mosque|church|monument|heritage|ancient|ruins|museum|historical)\b", ["historical_photo", "landscape"]),
    (r"\b(city|town|urban|rural|village|market|mela|festival|people|farmer|worker)\b", ["landscape", "historical_photo"]),
    (r"\b(transport|road|railway|train|bridge|dam|mine|factory|port|infrastructure)\b", ["diagram", "landscape", "map"]),
    (r"\b(rock|mineral|fold|fault|earthquake|volcano|tectonic|strata|sediment)\b", ["diagram", "landscape"]),
    (r"\b(climate|weather|rainfall|monsoon|temperature|precipitation|wind)\b", ["diagram", "map"]),
    (r"\b(landform|physical\s+feature|terrain|topography)\b", ["landscape", "satellite_image", "map"]),
]

# If query is about X, these image types are hard-negative (excluded) UNLESS
# they also appear in a preferred rule above.
_HARD_NEGATIVE_RULES: list[tuple[str, list[str]]] = [
    (r"\b(desert|sand\s+dune|dunes|arid|thar)\b", ["wildlife", "chart"]),
    (r"\b(mountain|himalaya)\b", ["chart", "wildlife"]),
    (r"\b(transport|infrastructure|road|railway)\b", ["wildlife"]),
    (r"\b(data|statistics|graph|chart)\b", ["wildlife", "landscape"]),
    (r"\b(climate|weather|monsoon)\b", ["wildlife"]),
    (r"\b(animal|wildlife|species)\b", ["chart", "map"]),
    (r"\b(agriculture|farming|crop)\b", ["chart"]),
    (r"\b(river|lake|waterfall)\b", ["wildlife", "chart"]),
    (r"\b(historical|heritage|fort|temple)\b", ["wildlife", "chart"]),
]

# Stopwords excluded from concept tokens
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "can", "could", "may", "might", "shall", "must", "with", "and", "or",
    "but", "for", "of", "in", "on", "at", "to", "from", "by", "as", "it",
    "its", "this", "that", "these", "those", "my", "your", "our", "their",
    "what", "which", "who", "how", "when", "where", "why", "explain",
    "describe", "tell", "give", "write", "show", "me", "us", "about",
    "explain", "describe", "discuss", "define", "list", "mention", "name",
    "class", "chapter", "textbook", "india", "indian", "example", "examples",
})

_IMAGE_REQUEST_RE = re.compile(
    r"\b(with\s+images?|show\s+(?:me\s+)?(?:a\s+|the\s+)?(?:diagram|figure|picture|map|illustration|photo)s?|"
    r"include\s+(?:an?\s+)?images?|using\s+(?:diagrams?|pictures?|illustrations?)|"
    r"textbook\s+(?:diagram|figure|image)s?)\b",
    re.I,
)

# Question-start phrases to strip when building topic phrases
_QUESTION_STRIP_RE = re.compile(
    r"^(?:explain|describe|tell\s+me\s+about|what\s+is|what\s+are|how\s+does|"
    r"how\s+do|discuss|write\s+(?:a\s+note\s+)?(?:on|about)|give\s+(?:information\s+)?(?:on|about)|"
    r"define|list|name|mention|write\s+about|describe\s+briefly)\s+",
    re.I,
)
_QUESTION_TAIL_RE = re.compile(
    r"\s+(?:with\s+(?:diagram|images?|figure|pictures?)|in\s+detail|briefly|shortly|in\s+short)$",
    re.I,
)


@dataclass
class TopicIntent:
    """
    Distilled topic intent from a student query.

    Used as the *only* source for image retrieval scoring —
    never derived from or polluted by the LLM-generated answer.
    """

    query: str
    topic_phrases: list[str] = field(default_factory=list)
    concept_tokens: frozenset[str] = field(default_factory=frozenset)
    requested_visuals: bool = False
    preferred_types: list[str] = field(default_factory=list)
    excluded_types: list[str] = field(default_factory=list)
    rag_section_tokens: frozenset[str] = field(default_factory=frozenset)

    @property
    def intent_text(self) -> str:
        """Clean query text safe to embed — no LLM answer contamination."""
        return " ".join(self.topic_phrases) if self.topic_phrases else self.query

    @property
    def has_type_preference(self) -> bool:
        return bool(self.preferred_types)


def _tokenize(text: str) -> frozenset[str]:
    tokens = set(re.findall(r"\b[a-z][a-z]+\b", (text or "").lower()))
    return frozenset(tokens - _STOPWORDS)


def _extract_topic_phrase(query: str) -> str:
    """Strip common question prefixes/suffixes, return the raw topic string."""
    q = re.sub(r"\s+", " ", (query or "").strip())
    q = _QUESTION_STRIP_RE.sub("", q)
    q = _QUESTION_TAIL_RE.sub("", q)
    return q.strip() or query.strip()


def build_topic_intent(
    query: str,
    rag_docs: list[Any] | None = None,
) -> TopicIntent:
    """
    Build a TopicIntent from a student question and optional RAG doc metadata.

    Does NOT accept or use the LLM-generated answer text.
    """
    intent = TopicIntent(query=query)
    intent.requested_visuals = bool(_IMAGE_REQUEST_RE.search(query))
    topic_phrase = _extract_topic_phrase(query)
    intent.topic_phrases = [topic_phrase] if topic_phrase else [query]
    intent.concept_tokens = _tokenize(topic_phrase)

    # Preferred image types from query surface-form
    preferred: list[str] = []
    for pattern, types in _TYPE_PREFERENCE_RULES:
        if re.search(pattern, query, re.I):
            for t in types:
                if t not in preferred:
                    preferred.append(t)
    intent.preferred_types = preferred

    # Excluded types (hard negatives) — preferred wins if conflict
    excluded: list[str] = []
    for pattern, types in _HARD_NEGATIVE_RULES:
        if re.search(pattern, query, re.I):
            for t in types:
                if t not in excluded and t not in preferred:
                    excluded.append(t)
    intent.excluded_types = excluded

    # Section tokens from RAG doc metadata (section_hint / content_label)
    if rag_docs:
        section_parts: list[str] = []
        for d in rag_docs:
            meta = getattr(d, "metadata", None) or {}
            hint = meta.get("section_hint") or meta.get("content_label") or ""
            if hint:
                section_parts.append(str(hint))
        if section_parts:
            intent.rag_section_tokens = _tokenize(" ".join(section_parts))

    return intent
