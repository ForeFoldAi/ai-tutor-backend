"""
Structured image intent extraction for precision educational image retrieval.

Converts a student question + retrieved RAG chunks into a rich ImageIntent
that drives concept-specific figure ranking.

Key design decisions:
  - NEVER uses the LLM-generated answer (prevents retrieval contamination).
  - Fully deterministic and lightweight (keyword/regex only, no LLM calls).
  - required_terms enforce hard concept relevance in _concept_specificity_score.
  - negative_terms down-score divergent examples.
  - supporting_terms provide partial-match boost.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Shared stopwords
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "can", "could", "may", "might", "shall", "must", "with", "and", "or",
    "but", "for", "of", "in", "on", "at", "to", "from", "by", "as", "it",
    "its", "this", "that", "these", "those", "my", "your", "our", "their",
    "what", "which", "who", "how", "when", "where", "why", "explain",
    "describe", "tell", "give", "write", "show", "me", "us", "about",
    "discuss", "define", "list", "mention", "name", "class", "chapter",
    "textbook", "example", "examples", "also", "some", "using", "used",
    "use", "not", "very", "more", "most", "much", "well", "just", "only",
})

# ---------------------------------------------------------------------------
# Image-type preference tables  (aligned with classify_image_type in extraction)
# ---------------------------------------------------------------------------

_TYPE_PREFERENCE_RULES: list[tuple[str, list[str]]] = [
    (r"\b(automated\s+weather\s+station|aws|weather\s+station|met\s+station|meteorological)\b",
     ["weather_station", "instrument"]),
    (r"\b(instrument|sensor|gauge|thermometer|barometer|anemometer|hygrometer|equipment|apparatus)\b",
     ["instrument", "diagram"]),
    (r"\b(satellite|aerial|from\s+above|top.?view|remote\s+sensing)\b",
     ["satellite_image"]),
    (r"\b(map|distribution|region|boundary|border|territory|location\s+of)\b",
     ["map", "satellite_image"]),
    (r"\b(desert|sand\s+dune|dunes|arid|thar|rajasthan|sahara|gobi)\b",
     ["landscape", "map"]),
    (r"\b(mountain|himalaya|peak|glacier|valley|range|everest)\b",
     ["landscape", "satellite_image", "diagram"]),
    (r"\b(river|lake|delta|plain|gangetic|flood|waterfall|falls)\b",
     ["landscape", "map", "satellite_image"]),
    (r"\b(coast|beach|island|sea|ocean|gulf|bay|peninsula)\b",
     ["landscape", "satellite_image", "map"]),
    (r"\b(forest|vegetation|biome|ecosystem|biodiversity|jungle|mangrove)\b",
     ["landscape", "wildlife"]),
    (r"\b(agriculture|crop|farming|irrigation|paddy|wheat)\b",
     ["landscape", "diagram"]),
    (r"\b(animal|wildlife|species|bird|tiger|lion|elephant|gharial|crocodile|peacock|deer|bear|yak|camel)\b",
     ["wildlife"]),
    (r"\b(diagram|process|formation|cycle|structure|cross.?section|flowchart)\b",
     ["diagram", "process"]),
    (r"\b(step|procedure|method|sequence|stage|phase|operation|activity)\b",
     ["process", "activity"]),
    (r"\b(graph|chart|bar|pie|line\s+graph|data|statistics|percentage)\b",
     ["chart"]),
    (r"\b(fort|palace|temple|monument|heritage|ancient|ruins|historical)\b",
     ["historical_photo", "landscape"]),
    (r"\b(climate|weather|rainfall|monsoon|temperature|precipitation|wind)\b",
     ["diagram", "map"]),
    (r"\b(landform|terrain|topography|physical\s+feature)\b",
     ["landscape", "satellite_image", "map"]),
    (r"\b(transport|road|railway|bridge|dam|factory|port|infrastructure)\b",
     ["diagram", "landscape", "map"]),
    (r"\b(rock|mineral|fold|fault|earthquake|volcano|tectonic)\b",
     ["diagram", "landscape"]),
]

_HARD_NEGATIVE_RULES: list[tuple[str, list[str]]] = [
    (r"\b(automated\s+weather\s+station|aws|weather\s+station|met\s+station)\b",
     ["wildlife", "chart", "landscape"]),
    (r"\b(instrument|sensor|gauge|thermometer)\b",
     ["wildlife", "chart", "landscape"]),
    (r"\b(desert|sand\s+dune|thar|arid)\b",
     ["wildlife", "chart", "instrument"]),
    (r"\b(mountain|himalaya)\b",
     ["chart", "wildlife", "instrument"]),
    (r"\b(transport|infrastructure|road|railway)\b",
     ["wildlife", "instrument"]),
    (r"\b(data|statistics|graph)\b",
     ["wildlife", "landscape"]),
    (r"\b(climate|weather|monsoon)\b",
     ["wildlife"]),
    (r"\b(animal|wildlife|species)\b",
     ["chart", "map", "instrument"]),
    (r"\b(agriculture|farming|crop)\b",
     ["chart", "instrument"]),
    (r"\b(river|lake|waterfall)\b",
     ["wildlife", "chart", "instrument"]),
    (r"\b(historical|heritage|fort|temple)\b",
     ["wildlife", "chart", "instrument"]),
]

# Query type classification patterns
_QUERY_TYPE_RULES: list[tuple[str, str]] = [
    (r"\b(what\s+is|what\s+are|define|definition|meaning)\b", "concept_definition"),
    (r"\b(how\s+does|how\s+do|how\s+is|how\s+are|mechanism|works?|function)\b", "process"),
    (r"\b(diagram|figure|picture|map|show|draw|illustrat)\b", "diagram_request"),
    (r"\b(compare|difference|distinguish|versus|vs\.?|contrast)\b", "comparison"),
    (r"\b(explain|describe|discuss|elaborate|briefly)\b", "concept_explanation"),
    (r"\b(list|name|mention|give|write)\b", "enumeration"),
    (r"\b(example|instance|case)\b", "example_request"),
]

# Patterns to extract divergent examples from RAG text
_EXAMPLE_PATTERNS = re.compile(
    r"\b(?:for\s+example|for\s+instance|such\s+as|e\.g\.|like|consider)\b"
    r"[,\s]+([^.!?]{5,80})",
    re.I,
)

# Question prefix stripping for core concept extraction
_QUESTION_STRIP_RE = re.compile(
    r"^(?:explain|describe|tell\s+me\s+about|what\s+is|what\s+are|how\s+does|"
    r"how\s+do|discuss|write\s+(?:a\s+note\s+)?(?:on|about)|"
    r"give\s+(?:information\s+|details\s+)?(?:on|about)|"
    r"define|list|name|mention|write\s+about|describe\s+briefly|"
    r"draw\s+a?\s*(?:labelled?\s*)?diagram\s+of|"
    r"with\s+the\s+help\s+of\s+a\s+diagram)\s+",
    re.I,
)
_QUESTION_TAIL_RE = re.compile(
    r"\s+(?:with\s+(?:diagram|images?|figure|pictures?)|"
    r"in\s+detail|briefly|shortly|in\s+short|with\s+example)$",
    re.I,
)

# Known synonyms / abbreviations table
_CONCEPT_SYNONYMS: dict[str, list[str]] = {
    "automated weather station": ["aws", "met station", "meteorological station", "weather station"],
    "weather station": ["aws", "met station", "meteorological instrument"],
    "automated teller machine": ["atm"],
    "light emitting diode": ["led"],
    "central processing unit": ["cpu"],
    "photovoltaic": ["solar panel", "solar cell"],
    "greenhouse gas": ["ghg", "carbon dioxide", "co2"],
    "deoxyribonucleic acid": ["dna"],
    "gross domestic product": ["gdp"],
    "information technology": ["it"],
    "artificial intelligence": ["ai"],
    "solar system": ["planet", "orbit"],
    "water cycle": ["hydrological cycle", "hydrologic cycle", "evaporation", "precipitation"],
    "food chain": ["predator", "prey", "producer", "consumer"],
    "food web": ["predator", "prey", "trophic"],
}


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImageIntent:
    """
    Structured educational image retrieval intent.

    Built from student question + RAG chunks only — never from the LLM answer.
    Drives the concept_specificity_score which dominates final image ranking.
    """

    query: str
    core_concept: str                       # stripped question phrase
    required_terms: list[str]               # phrases/words that MUST match for high specificity
    supporting_terms: list[str]             # partial boost if present
    negative_terms: list[str]               # penalise if present
    query_type: str                         # concept_explanation / process / diagram_request / ...
    preferred_types: list[str]              # image_type values preferred for this query
    excluded_types: list[str]               # hard-negative image_type values
    concept_tokens: frozenset[str]          # tokenised core concept
    rag_section_tokens: frozenset[str] = field(default_factory=frozenset)
    entities: list[str] = field(default_factory=list)
    requested_visuals: bool = False

    @property
    def intent_text(self) -> str:
        """Clean embed-safe text representing the query intent."""
        return self.core_concept if self.core_concept else self.query

    @property
    def has_type_preference(self) -> bool:
        return bool(self.preferred_types)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IMAGE_REQUEST_RE = re.compile(
    r"\b(with\s+images?|show\s+(?:me\s+)?(?:a\s+|the\s+)?(?:diagram|figure|picture|map|illustration|photo)s?|"
    r"include\s+(?:an?\s+)?images?|using\s+(?:diagrams?|pictures?|illustrations?)|"
    r"textbook\s+(?:diagram|figure|image)s?|draw\s+(?:a\s+)?(?:labelled?\s*)?diagram)\b",
    re.I,
)


def _tokenize(text: str) -> frozenset[str]:
    tokens = set(re.findall(r"\b[a-z][a-z]+\b", (text or "").lower()))
    return frozenset(tokens - _STOPWORDS)


def _extract_core_concept(question: str) -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    q = _QUESTION_STRIP_RE.sub("", q)
    q = _QUESTION_TAIL_RE.sub("", q)
    return q.strip() or question.strip()


def _extract_required_terms(core_concept: str, question: str) -> list[str]:
    """
    Build the set of phrases/words that should appear in a relevant figure's
    caption or snippet.  Ordered from most-specific to least-specific so the
    score function can apply tiered weights.
    """
    terms: list[str] = []
    cc = core_concept.lower().strip()

    # 1. Full core concept phrase (highest priority)
    if cc and cc not in terms:
        terms.append(cc)

    # 2. Check synonym table
    for key, synonyms in _CONCEPT_SYNONYMS.items():
        if key in cc or cc in key:
            for syn in synonyms:
                if syn not in terms:
                    terms.append(syn)
            break

    # 3. Bigrams from core concept
    words = [w for w in re.findall(r"\b[a-z][a-z]+\b", cc) if w not in _STOPWORDS]
    for i in range(len(words) - 1):
        bg = f"{words[i]} {words[i+1]}"
        if bg not in terms:
            terms.append(bg)

    # 4. Abbreviations in question (all-caps 2-4 chars)
    for abbrev in re.findall(r"\b([A-Z]{2,5})\b", question):
        al = abbrev.lower()
        if al not in terms and al not in _STOPWORDS:
            terms.append(al)

    # 5. Individual content keywords: ONLY 8+ char words to avoid generic terms.
    # Words like "weather" (7 chars) or "station" (7 chars) are too common to be
    # distinctive required terms by themselves.  They live in concept_tokens for
    # the Level-3 (soft) check only.
    for w in words:
        if len(w) >= 8 and w not in terms:
            terms.append(w)

    return terms


def _extract_supporting_terms(
    core_concept: str,
    required_terms: list[str],
    chunks: list[Any],
) -> list[str]:
    """
    Terms that frequently co-occur with the core concept in RAG chunks.
    These boost the score but are not required.
    """
    core_words = _tokenize(core_concept)
    req_set = set(required_terms)
    freq: dict[str, int] = defaultdict(int)

    for chunk in chunks or []:
        text = (getattr(chunk, "page_content", None) or "").lower()
        if not text:
            continue
        chunk_words = text.split()
        # Only mine from chunks that mention core concept
        if not any(cw in text for cw in core_words):
            continue
        for w in re.findall(r"\b[a-z][a-z]+\b", text):
            if w not in _STOPWORDS and len(w) >= 4 and w not in core_words and w not in req_set:
                freq[w] += 1

    # Top-8 co-occurring terms (min frequency 1)
    return [t for t, _ in sorted(freq.items(), key=lambda x: -x[1])[:8]]


def _extract_negative_terms(
    core_concept: str,
    required_terms: list[str],
    chunks: list[Any],
) -> list[str]:
    """
    Terms extracted from divergent examples in RAG chunks.
    These penalise figures that are about tangential topics.
    """
    core_words = _tokenize(core_concept)
    req_set = set(required_terms)
    neg: set[str] = set()

    for chunk in chunks or []:
        text = (getattr(chunk, "page_content", None) or "")
        for m in _EXAMPLE_PATTERNS.finditer(text):
            example = m.group(1).lower()
            ex_tokens = _tokenize(example)
            # Example is "divergent" if it shares no words with core concept
            if not (core_words & ex_tokens):
                for t in ex_tokens:
                    if len(t) >= 4 and t not in req_set:
                        neg.add(t)

    return list(neg)[:6]


def _classify_query_type(question: str) -> str:
    q = question.lower()
    for pattern, qtype in _QUERY_TYPE_RULES:
        if re.search(pattern, q, re.I):
            return qtype
    return "concept_explanation"


def _get_type_preferences(question: str, query_type: str) -> tuple[list[str], list[str]]:
    preferred: list[str] = []
    excluded: list[str] = []

    for pattern, types in _TYPE_PREFERENCE_RULES:
        if re.search(pattern, question, re.I):
            for t in types:
                if t not in preferred:
                    preferred.append(t)

    for pattern, types in _HARD_NEGATIVE_RULES:
        if re.search(pattern, question, re.I):
            for t in types:
                if t not in excluded and t not in preferred:
                    excluded.append(t)

    # diagram_request queries always prefer diagram/process/instrument
    if query_type == "diagram_request":
        for t in ("diagram", "process", "instrument", "weather_station"):
            if t not in preferred:
                preferred.insert(0, t)

    return preferred, excluded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_image_intent(
    question: str,
    retrieved_chunks: list[Any] | None = None,
    *,
    conversation_context: Any | None = None,
) -> ImageIntent:
    """
    Build an ImageIntent from the student question and optional RAG chunks.

    NEVER uses the LLM-generated answer.
    Fully deterministic: keyword extraction + regex patterns only.

    Args:
        question: Raw student question.
        retrieved_chunks: LangChain Document list from RAG (page_content + metadata).

    Returns:
        ImageIntent with all scoring signals pre-computed.
    """
    chunks = retrieved_chunks or []
    effective_question = question
    if conversation_context is not None:
        rq = getattr(conversation_context, "retrieval_query", None) or ""
        if rq.strip():
            effective_question = rq.strip()

    core_concept = _extract_core_concept(effective_question)
    required_terms = _extract_required_terms(core_concept, effective_question)
    supporting_terms = _extract_supporting_terms(core_concept, required_terms, chunks)
    negative_terms = _extract_negative_terms(core_concept, required_terms, chunks)
    query_type = _classify_query_type(effective_question)
    preferred_types, excluded_types = _get_type_preferences(effective_question, query_type)

    # Broad definition queries ("what is weather?") should not pull instruments/AWS.
    _broad_definition_cores = frozenset({
        "weather", "climate", "monsoon", "rainfall", "temperature", "precipitation",
    })
    if query_type == "concept_definition" and core_concept.lower() in _broad_definition_cores:
        if not re.search(
            r"\b(station|aws|instrument|sensor|gauge|anemometer|meteorological)\b",
            effective_question,
            re.I,
        ):
            for t in ("weather_station", "instrument"):
                if t not in excluded_types:
                    excluded_types.append(t)

    # Section tokens from RAG metadata
    section_parts: list[str] = []
    for d in chunks:
        meta = getattr(d, "metadata", None) or {}
        hint = meta.get("section_hint") or meta.get("content_label") or ""
        if hint:
            section_parts.append(str(hint))
    rag_section_tokens = _tokenize(" ".join(section_parts)) if section_parts else frozenset()

    from app.services.symbolic_image_filters import extract_educational_entities

    entities = extract_educational_entities(effective_question, core_concept)
    if conversation_context is not None:
        for ent in getattr(conversation_context, "inherited_entities", []) or []:
            if ent and ent not in entities:
                entities.append(ent)

    return ImageIntent(
        query=effective_question,
        core_concept=core_concept,
        required_terms=required_terms,
        supporting_terms=supporting_terms,
        negative_terms=negative_terms,
        query_type=query_type,
        preferred_types=preferred_types,
        excluded_types=excluded_types,
        concept_tokens=_tokenize(core_concept),
        entities=entities,
        rag_section_tokens=rag_section_tokens,
        requested_visuals=(
            bool(_IMAGE_REQUEST_RE.search(effective_question))
            or _context_requires_visuals(conversation_context)
        ),
    )


def _context_requires_visuals(conversation_context: Any | None) -> bool:
    if conversation_context is None:
        return False
    try:
        from app.services.conversation_context import VisualIntent

        return conversation_context.visual_intent == VisualIntent.REQUIRED_VISUALS
    except Exception:
        return False
