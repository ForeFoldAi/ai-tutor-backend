"""
Auto-caption generation for textbook figures with missing or minimal labels.

Used when:
  - Image has only a figure number ("Fig. 2.1") with no descriptive text
  - Image is an orphan (no paired caption) but has nearby educational content

Algorithm (no LLM — fully deterministic):
  1. Extract the most informative sentence from nearby_before / nearby_after
  2. Identify the core educational concept from section/subsection titles
  3. Combine with image-type-appropriate prefix phrase
  4. Return a concise, human-readable caption (max 180 chars)

Design principle: prefer precision over completeness. A generated caption that
says "Diagram showing soil layers" is more useful than one that says
"Illustration from Chapter 4 Section 2.1 about soil and rock formation layers."
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Image-type → descriptive verb/noun for generated captions
# ---------------------------------------------------------------------------

_TYPE_PREFIX: dict[str, str] = {
    "weather_station": "Automated weather station",
    "instrument": "Scientific instrument",
    "diagram": "Diagram",
    "process": "Process diagram",
    "map": "Map",
    "chart": "Chart",
    "satellite_image": "Satellite image",
    "activity": "Activity",
    "landscape": "Photograph",
    "historical_photo": "Historical photograph",
    "wildlife": "Photograph",
    "unknown": "Illustration",
}

# Sentence-boundary split (handles abbreviations crudely)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# Fig marker to strip from inline references
_FIG_INLINE_RE = re.compile(r"(?i)\b(?:fig\.?|figure|diagram|illustration|plate)\s*\d+(?:\.\d+)*[.:,]?\s*")
# Filter short, number-only, or noisy lines
_MIN_SENT_WORDS = 6
# Heading stop words (don't embed raw headings as captions)
_HEADING_STOPWORDS = frozenset({
    "activities", "exercises", "questions", "summary", "review", "do you know",
    "think and discuss", "let us recall", "intext questions", "additional questions",
})


def _clean_text(text: str) -> str:
    """Normalise whitespace and strip figure references from surrounding text."""
    t = _FIG_INLINE_RE.sub(" ", text or "")
    return re.sub(r"\s{2,}", " ", t.replace("\n", " ")).strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences; skip very short or noisy ones."""
    parts = _SENT_RE.split(_clean_text(text))
    out: list[str] = []
    for p in parts:
        p = p.strip(" .:;-")
        words = p.split()
        if len(words) < _MIN_SENT_WORDS:
            continue
        if re.fullmatch(r"[\d\s.,:;/\-–—]+", p):
            continue
        out.append(p)
    return out


def _score_sentence(sent: str, concept_hint: str) -> float:
    """
    Score a sentence by educational usefulness.

    Higher score → better candidate for a generated caption.
    """
    s = sent.lower()
    score = 0.0
    # Length: ~15-40 words is ideal
    wc = len(sent.split())
    if 10 <= wc <= 45:
        score += min(1.0, wc / 30.0)
    # Contains concept hint words
    if concept_hint:
        hint_words = set(re.findall(r"\b[a-z]{3,}\b", concept_hint.lower()))
        sent_words = set(re.findall(r"\b[a-z]{3,}\b", s))
        overlap = hint_words & sent_words
        score += min(2.0, len(overlap) * 0.4)
    # Textbook definition sentences (e.g. "Weather is a state of the atmosphere…")
    if re.search(r"(?i)\bwhat\s+is\s+\w+\b", s):
        score += 2.5
    if re.search(
        r"(?i)\b(?:weather|climate)\s+is\s+(?:a\s+)?state\s+of\s+(?:the\s+)?(?:earth['\u2019]?s\s+)?atmosphere\b",
        s,
    ):
        score += 3.0
    if re.search(r"(?i)day[- ]to[- ]day\s+condition\s+of\s+the\s+atmosphere", s):
        score += 2.5

    # Educational signal words
    edu_signals = {
        "shows", "illustrates", "depicts", "represents", "diagram", "figure",
        "consists", "contains", "made", "formed", "located", "found", "used",
        "called", "known", "type", "example", "process", "step", "stage",
        "layer", "part", "structure", "system", "method", "instrument",
    }
    s_words = set(re.findall(r"\b\w+\b", s))
    score += min(1.0, len(s_words & edu_signals) * 0.25)
    # Penalise heading-like sentences
    if s.strip(" .:").lower() in _HEADING_STOPWORDS:
        score -= 2.0
    # Penalise sentences that start with a number (table row / list item)
    if re.match(r"^\d", sent.strip()):
        score -= 0.5
    return score


def _best_sentence(
    nearby_before: str,
    nearby_after: str,
    concept_hint: str,
) -> str | None:
    """Pick the single most informative sentence from surrounding text."""
    candidates: list[tuple[float, str]] = []

    # Prefer text immediately before figure (describes it) …
    for sent in _split_sentences(nearby_before)[-4:]:
        candidates.append((_score_sentence(sent, concept_hint), sent))

    # … then text immediately after (often the caption or first explanatory line)
    for sent in _split_sentences(nearby_after)[:4]:
        candidates.append((_score_sentence(sent, concept_hint), sent))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    score, best = candidates[0]
    if score < 0.3:
        return None
    # Truncate to ~120 chars at a word boundary
    if len(best) > 130:
        best = best[:127].rsplit(" ", 1)[0] + "…"
    return best


def _concept_from_headings(
    section_title: str | None,
    subsection_title: str | None,
    chapter_title: str | None,
) -> str:
    """Return the most specific available heading as the concept hint."""
    for heading in (subsection_title, section_title, chapter_title):
        if heading and heading.strip():
            h = heading.strip()
            # Skip generic headings
            if h.lower() in _HEADING_STOPWORDS or len(h) < 4:
                continue
            return h
    return ""


def _type_prefix(image_type: str) -> str:
    return _TYPE_PREFIX.get(image_type or "unknown", "Illustration")


def generate_contextual_caption(
    *,
    figure_number: str | None,
    image_type: str,
    section_title: str | None,
    subsection_title: str | None,
    chapter_title: str | None,
    nearby_before: str,
    nearby_after: str,
) -> str | None:
    """
    Generate a concise educational caption for figures lacking descriptive text.

    Returns None if insufficient context is available to produce a useful caption
    (prevents hallucinated / meaningless captions).

    Examples
    --------
    - figure_number="2.1", image_type="diagram", section_title="Soil Layers"
      → "Diagram showing soil layers"
    - section "Automated Weather Station", nearby: "consists of anemometer, barometer…"
      → "Automated weather station showing anemometer, barometer and related instruments"
    """
    concept = _concept_from_headings(section_title, subsection_title, chapter_title)
    prefix = _type_prefix(image_type)

    # Attempt 1: pick the best sentence from nearby text
    best_sent = _best_sentence(nearby_before, nearby_after, concept)

    if best_sent:
        # If the best sentence is short and concept exists, prepend concept
        if concept and concept.lower() not in best_sent.lower() and len(best_sent) < 80:
            caption = f"{prefix} of {concept}: {best_sent}"
        else:
            # Use sentence as-is, prepend type prefix if not already present
            if not best_sent.lower().startswith(prefix.lower()):
                caption = f"{prefix} — {best_sent}"
            else:
                caption = best_sent
        return caption[:200]

    # Attempt 2: fall back to concept heading only
    if concept:
        fig_ref = f"Fig. {figure_number}" if figure_number else ""
        parts = [prefix]
        if concept:
            parts.append(f"showing {concept.lower()}")
        if fig_ref:
            parts.append(f"({fig_ref})")
        caption = " ".join(parts)
        return caption[:180]

    # Insufficient context
    return None


# ---------------------------------------------------------------------------
# Semantic keyword extractor (no LLM — TF-IDF-style term scoring)
# ---------------------------------------------------------------------------

_KEYWORD_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "may", "might", "shall", "must", "with", "and", "or", "but", "for", "of",
    "in", "on", "at", "to", "from", "by", "as", "it", "its", "this", "that",
    "these", "those", "also", "some", "such", "both", "each", "other", "more",
    "most", "very", "just", "only", "not", "no", "nor", "so", "yet", "too",
    "figure", "fig", "diagram", "illustration", "photo", "image", "picture",
    "chapter", "section", "unit", "lesson", "page", "textbook", "book",
    "class", "table", "shows", "showing", "see", "refer", "note", "example",
})

_EDUCATIONAL_DOMAIN_TERMS: frozenset[str] = frozenset({
    # Science / Geography
    "photosynthesis", "evaporation", "condensation", "precipitation", "rainfall", "transpiration",
    "osmosis", "diffusion", "digestion", "respiration", "circulation", "reproduction",
    "ecosystem", "habitat", "biodiversity", "conservation", "pollution", "climate",
    "latitude", "longitude", "altitude", "topography", "erosion", "weathering",
    "sediment", "stratum", "strata", "mineral", "volcano", "earthquake", "tectonic",
    "monsoon", "cyclone", "tsunami", "drought", "flood", "irrigation", "agriculture",
    "hygrometer", "anemometer", "barometer", "thermometer",
    # Biology
    "nucleus", "mitochondria", "chloroplast", "chromosome", "enzyme", "protein",
    "membrane", "cytoplasm", "organism", "species", "evolution", "adaptation",
    # Physics / Chemistry
    "conduction", "convection", "radiation", "refraction", "reflection", "velocity",
    "acceleration", "friction", "gravity", "magnetism", "electrode", "electrolysis",
    "oxidation", "reduction", "combustion", "compound", "element", "molecule",
    # Math
    "geometry", "algebra", "fraction", "decimal", "percentage", "ratio", "proportion",
    # History
    "civilization", "dynasty", "empire", "colonialism", "independence", "revolution",
    "constitution", "parliament", "democracy", "republic", "monarchy",
})


def extract_semantic_keywords(
    caption: str,
    generated_caption: str | None,
    nearby_before: str,
    nearby_after: str,
    section_title: str | None,
    chapter_title: str | None,
    image_type: str,
) -> str:
    """
    Extract pipe-separated semantic keywords from figure context.

    Returns top-10 most distinctive educational terms as a pipe-separated string,
    e.g. "weather|station|anemometer|instrument|measurement|aws".
    """
    # Aggregate all text sources
    blob = " ".join(
        t for t in [caption, generated_caption, nearby_before[:600], nearby_after[:600],
                    section_title, chapter_title]
        if t
    ).lower()

    # Tokenise
    tokens = re.findall(r"\b[a-z][a-z]{2,}\b", blob)
    if not tokens:
        return image_type or ""

    # Frequency count
    freq: dict[str, int] = {}
    for t in tokens:
        if t not in _KEYWORD_STOPWORDS and len(t) >= 3:
            freq[t] = freq.get(t, 0) + 1

    # Boost domain-specific terms
    scored: list[tuple[float, str]] = []
    for term, count in freq.items():
        boost = 2.0 if term in _EDUCATIONAL_DOMAIN_TERMS else 1.0
        # Prefer longer, more specific terms
        length_bonus = min(1.5, len(term) / 8.0)
        scored.append((count * boost * length_bonus, term))

    scored.sort(key=lambda x: -x[0])
    top_terms = [term for _, term in scored[:10]]

    # Always include image_type as a keyword
    if image_type and image_type != "unknown" and image_type not in top_terms:
        top_terms = [image_type] + top_terms[:9]

    return "|".join(top_terms)


# ---------------------------------------------------------------------------
# Educational tag generator
# ---------------------------------------------------------------------------

_SUBJECT_TAG_PATTERNS: list[tuple[str, str]] = [
    (r"\b(photosynthesis|chlorophyll|mitosis|meiosis|cell|organism|bacteria|virus|fungi|plant|animal|mammal|bird|reptile|insect|fish|microorganism)\b", "biology"),
    (r"\b(force|gravity|velocity|acceleration|friction|motion|energy|power|heat|light|sound|electricity|magnetism|circuit|wave|optics|thermodynamics)\b", "physics"),
    (r"\b(element|compound|molecule|atom|ion|bond|reaction|acid|base|salt|oxidation|reduction|electrolysis|periodic table|metal|nonmetal)\b", "chemistry"),
    (r"\b(integer|fraction|decimal|percentage|ratio|algebra|geometry|triangle|circle|square|area|volume|probability|statistics|calculus|equation)\b", "mathematics"),
    (r"\b(map|river|mountain|plateau|plain|valley|climate|rainfall|monsoon|soil|mineral|erosion|latitude|longitude|district|state|country|continent|ocean)\b", "geography"),
    (r"\b(civilization|dynasty|empire|war|revolution|independence|treaty|constitution|parliament|colonial|historical|ancient|medieval|modern)\b", "history"),
    (r"\b(government|democracy|parliament|constitution|rights|duties|citizen|election|policy|law|court|judiciary|legislature|executive)\b", "civics"),
    (r"\b(economy|trade|industry|agriculture|gdp|inflation|employment|poverty|market|supply|demand|budget|tax|bank|currency)\b", "economics"),
    (r"\b(poem|prose|story|novel|grammar|sentence|vocabulary|tense|noun|verb|adjective|adverb|punctuation|comprehension)\b", "english"),
    (r"\b(atom|nucleus|proton|neutron|electron|radioactive|nuclear|fission|fusion)\b", "nuclear_physics"),
]

_ROLE_TAGS: dict[str, str] = {
    "primary_concept": "concept|teaching",
    "supporting_example": "example",
    "activity": "activity|experiment",
    "sidebar_example": "reference",
    "decorative": "decorative",
}

_TYPE_TAGS: dict[str, str] = {
    "table": "table|data|comparison|statistics",
    "formula": "formula|equation|expression|calculation",
    "weather_station": "instrument|measurement|meteorology",
    "instrument": "instrument|measurement",
    "diagram": "diagram|structure",
    "process": "process|steps|sequence",
    "map": "map|geography|location",
    "chart": "chart|data|statistics",
    "satellite_image": "remote_sensing|aerial|geography",
    "activity": "experiment|hands_on",
    "landscape": "environment|landform",
    "historical_photo": "history|heritage",
    "wildlife": "fauna|ecosystem",
}


def generate_educational_tags(
    caption: str,
    generated_caption: str | None,
    nearby_before: str,
    nearby_after: str,
    image_type: str,
    educational_role: str,
    section_title: str | None,
) -> str:
    """
    Generate pipe-separated educational tags from figure metadata.

    Tags encode: subject domain, image function, and type.
    Example: "geography|diagram|structure|teaching"
    """
    blob = " ".join(
        t for t in [caption, generated_caption, nearby_before[:400], section_title]
        if t
    ).lower()

    tags: list[str] = []

    # Subject domain tags
    for pattern, subject_tag in _SUBJECT_TAG_PATTERNS:
        if re.search(pattern, blob, re.I):
            if subject_tag not in tags:
                tags.append(subject_tag)

    # Image type tags
    type_tags = _TYPE_TAGS.get(image_type or "unknown", "")
    for t in type_tags.split("|"):
        if t and t not in tags:
            tags.append(t)

    # Educational role tags
    role_tags = _ROLE_TAGS.get(educational_role or "unknown", "")
    for t in role_tags.split("|"):
        if t and t not in tags:
            tags.append(t)

    return "|".join(tags[:12])


# ---------------------------------------------------------------------------
# Stage 7 — Educational title generation (structured, no LLM)
# ---------------------------------------------------------------------------

# Fig-number prefix stripped before short_title extraction
_CAPTION_FIG_PREFIX_RE = re.compile(
    r"(?i)^(?:fig\.?|figure|diagram|illustration|plate|exhibit|scheme)\s*\d+(?:\.\d+)*\s*[.:;\-–—]?\s*"
)

# Curriculum concept tag patterns — maps pattern → concept label
CURRICULUM_CONCEPT_TAGS: dict[str, list[str]] = {
    # Physics
    "light_optics":    ["refraction", "reflection", "lens", "mirror", "prism", "spectrum", "optics"],
    "heat_transfer":   ["conduction", "convection", "radiation", "heat transfer", "thermometer", "calorimeter"],
    "motion_forces":   ["force", "friction", "velocity", "acceleration", "newton", "momentum", "inertia"],
    "electricity":     ["circuit", "current", "voltage", "resistance", "ohm", "conductor", "capacitor"],
    "magnetism":       ["magnet", "magnetic field", "compass", "electromagnetic", "solenoid"],
    "waves_sound":     ["wave", "frequency", "amplitude", "sound", "vibration", "resonance"],
    # Chemistry
    "states_of_matter": ["solid", "liquid", "gas", "melting", "boiling", "evaporation", "sublimation", "condensation"],
    "chemical_reactions": ["oxidation", "reduction", "combustion", "electrolysis", "catalyst", "precipitate"],
    "atomic_structure": ["atom", "proton", "neutron", "electron", "nucleus", "periodic table", "valence"],
    # Biology
    "cell_biology":    ["cell", "nucleus", "membrane", "cytoplasm", "mitochondria", "chloroplast", "organelle"],
    "photosynthesis":  ["photosynthesis", "chlorophyll", "glucose", "carbon dioxide", "sunlight"],
    "human_body":      ["heart", "lung", "kidney", "liver", "brain", "muscle", "blood vessel", "nervous"],
    "plant_biology":   ["root", "stem", "leaf", "flower", "seed", "germination", "pollination", "xylem"],
    "ecology":         ["ecosystem", "food chain", "predator", "prey", "habitat", "biodiversity", "food web"],
    "reproduction":    ["reproduction", "fertilisation", "embryo", "gamete", "ovum", "sperm", "zygote"],
    # Geography
    "weather_climate": ["weather", "climate", "monsoon", "rainfall", "precipitation", "rain gauge", "temperature", "humidity", "wind speed"],
    "landforms":       ["mountain", "plateau", "plain", "valley", "river", "delta", "canyon", "glacier", "peninsula"],
    "water_cycle":     ["water cycle", "evaporation", "precipitation", "rainfall", "rain gauge", "transpiration", "groundwater"],
    "maps_cartography": ["map", "scale", "latitude", "longitude", "contour", "atlas", "legend", "grid"],
    "natural_disasters": ["earthquake", "volcano", "tsunami", "cyclone", "flood", "drought", "landslide"],
    # History
    "ancient_civilizations": ["civilization", "ancient", "mesopotamia", "egypt", "indus", "harappa", "dynasty"],
    "colonial_period": ["colonial", "empire", "trade route", "east india", "british", "mughal", "viceroy"],
    "independence_movement": ["independence", "freedom", "revolution", "nationalist", "civil disobedience", "satyagraha"],
    # Mathematics
    "geometry":        ["triangle", "circle", "polygon", "angle", "perimeter", "area", "volume", "congruence"],
    "algebra":         ["equation", "variable", "expression", "polynomial", "linear", "quadratic", "root"],
    "statistics":      ["graph", "bar chart", "mean", "median", "mode", "frequency", "histogram", "probability"],
    "number_systems":  ["integer", "fraction", "decimal", "rational", "irrational", "prime", "factor"],
    # Environment
    "pollution":       ["pollution", "waste", "sewage", "emission", "acid rain", "ozone", "greenhouse"],
    "conservation":    ["conservation", "reserve", "protected area", "biodiversity", "wildlife", "deforestation"],
}


def generate_educational_title(
    *,
    figure_number: str | None,
    image_type: str,
    caption: str | None,
    section_title: str | None,
    subsection_title: str | None,
    chapter_title: str | None,
    nearby_before: str,
    nearby_after: str,
) -> dict[str, object]:
    """
    Generate structured educational metadata for a figure (Stage 7).

    Always includes chapter and section context in both description and as the
    concept hint for title extraction.  Returns:
        {
            "short_title":   str   (≤ 80 chars, 3–7 words + section prefix),
            "description":   str   (≤ 400 chars, starts with Chapter/Section),
            "keywords":      list[str]   (top-5 semantic keywords),
            "concept_tags":  list[str]   (matched curriculum concept names),
        }
    """
    concept = _concept_from_headings(section_title, subsection_title, chapter_title)
    prefix = _type_prefix(image_type)

    # --- short_title ---
    bare_title: str | None = None

    # Attempt 1: strip Fig-number prefix from official caption
    if caption and caption.strip():
        stripped = _CAPTION_FIG_PREFIX_RE.sub("", caption.strip()).strip(" .:;-")
        if len(stripped.split()) >= 3:
            bare_title = " ".join(stripped.split()[:7])

    # Attempt 2: best sentence from nearby text (first 7 words)
    if not bare_title:
        best = _best_sentence(nearby_before, nearby_after, concept)
        if best:
            bare_title = " ".join(best.split()[:7]).rstrip(",.;:")

    # Always include section context in short_title
    if bare_title:
        if section_title and section_title.lower() not in bare_title.lower():
            short_title = f"{section_title}: {bare_title}"[:80]
        else:
            short_title = bare_title[:80]
    else:
        # Fallback: "{type_label} — {section or chapter}"
        heading = section_title or subsection_title or chapter_title or ""
        short_title = (f"{prefix} — {heading}" if heading else prefix)[:80]

    # --- description (always starts with Chapter / Section context) ---
    desc_parts: list[str] = []
    if chapter_title:
        desc_parts.append(f"Chapter: {chapter_title.strip()}")
    if section_title:
        desc_parts.append(f"Section: {section_title.strip()}")

    # Collect best 1–2 educational sentences from nearby text
    all_sents: list[tuple[float, str]] = []
    for sent in _split_sentences(nearby_before)[-4:]:
        all_sents.append((_score_sentence(sent, concept), sent))
    for sent in _split_sentences(nearby_after)[:4]:
        all_sents.append((_score_sentence(sent, concept), sent))
    all_sents.sort(key=lambda x: -x[0])
    edu_sents = [s for sc, s in all_sents if sc >= 0.3][:2]

    if edu_sents:
        desc_parts.append(" ".join(edu_sents))
    elif caption:
        stripped_cap = _CAPTION_FIG_PREFIX_RE.sub("", caption).strip(" .:;-")
        if stripped_cap:
            desc_parts.append(stripped_cap)

    description = ". ".join(desc_parts)[:400]

    # --- keywords (top-5) ---
    kw_str = extract_semantic_keywords(
        caption or "",
        None,
        nearby_before,
        nearby_after,
        section_title,
        chapter_title,
        image_type,
    )
    keywords: list[str] = [k for k in kw_str.split("|") if k][:5]

    # --- concept_tags ---
    blob = " ".join(
        t for t in [caption, nearby_before[:400], nearby_after[:400], section_title, chapter_title]
        if t
    ).lower()
    concept_tags: list[str] = []
    for tag_name, kw_list in CURRICULUM_CONCEPT_TAGS.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", blob) for kw in kw_list):
            concept_tags.append(tag_name)
            if len(concept_tags) >= 5:
                break

    return {
        "short_title": short_title,
        "description": description,
        "keywords": keywords,
        "concept_tags": concept_tags,
    }
