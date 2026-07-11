"""Interactive science experiment prompt, detection, and JSON extraction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.science_experiment.fallbacks import (
    _GENERIC_TYPES,
    build_fallback_science_experiment,
    merge_catalog_experiment,
)
from app.services.science_experiment.schemas import ScienceExperiment

logger = logging.getLogger(__name__)

_SCIENCE_EXPERIMENT_FENCE_RE = re.compile(
    r"```(?:science-experiment|json:science-experiment|science_experiment)\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)

_INCOMPLETE_SCIENCE_EXPERIMENT_RE = re.compile(
    r"\n?```(?:science-experiment|json:science-experiment|science_experiment)\s*\n[\s\S]*$",
    re.IGNORECASE,
)

_MARKDOWN_FENCE_RE = re.compile(r"```markdown\s*\n([\s\S]*?)```", re.IGNORECASE)

_SKIP_TYPES = frozenset({
    "greeting", "affirmation", "personal-response", "one-word", "brief", "factual",
})

_EXPLICIT_EXPERIMENT_RE = re.compile(
    r"\b("
    r"experiment|hands[\s-]?on|virtual\s+lab|lab\s+activity|"
    r"simulate|interactive|demonstrat|observe\s+in\s+(?:a\s+)?lab|"
    r"try\s+this\s+at\s+home|build\s+a\s+circuit|set\s+up\s+(?:an?\s+)?(?:experiment|circuit)"
    r")\b",
    re.I,
)

_SCIENCE_EXPERIMENT_APPENDIX = """\
INTERACTIVE SCIENCE EXPERIMENT (MANDATORY for every science answer — append after prose):

THREE-VIEW PRINCIPLE — every experiment MUST include synchronized views:
1. realWorld — what students see in a classroom lab
2. microscopic — atoms, molecules, charges, cells (particle level)
3. scientific — concepts, equations, reasoning

GRADE BANDS:
- Classes 1–3: bright illustrations, simple animations, minimal text
- Classes 4–5: detailed animations, basic interactivity, prediction questions
- Classes 6–8: interactive simulations — students modify variables and observe
- Classes 9–10: virtual labs with measurements, graphs, particle + observable levels

RULES:
- Write the FULL science tutor answer FIRST (structured headings for non-math subjects).
- AFTER the answer, append ONE complete ```science-experiment``` JSON block.
- JSON must be valid and closed with ```.
- Do NOT wrap the answer in ```markdown fences.

```science-experiment
{
  "conceptName": "...",
  "classLevel": "Class N",
  "learningObjective": "...",
  "conceptExplanation": "one-sentence summary",
  "experiment": {
    "experimentType": "plant-growth|seed-germination|photosynthesis|respiration|digestion|blood-circulation|magnetism|electricity|light-reflection|refraction|acids-bases|chemical-reaction|states-of-matter|heat-transfer|water-cycle|sound|force-motion|solar-system|human-organs|concept-explorer",
    "title": "...",
    "description": "what to explore",
    "gradeTier": "elementary|primary|middle|advanced",
    "threeViews": {
      "realWorld": {"title": "...", "description": "...", "narration": "..."},
      "microscopic": {"title": "...", "description": "...", "narration": "..."},
      "scientific": {"title": "...", "description": "...", "narration": "...", "equation": "..."}
    },
    "sliders": [{"id": "light", "label": "Light", "min": 0, "max": 100, "step": 5, "default": 50, "unit": "%"}],
    "liveCalculations": [{"id": "rate", "label": "Rate", "formula": "light * 0.5", "unit": ""}],
    "buttons": [{"id": "animate", "label": "▶ Play", "action": "animate"}, {"id": "reset", "label": "Reset", "action": "reset"}],
    "colors": {"primary": "#0EA5E9", "secondary": "#10B981", "accent": "#F59E0B", "background": "#F0F9FF", "text": "#0F172A"},
    "studentInteractions": [{"id": "s1", "type": "slide", "description": "Change a variable", "expectedObservation": "..."}],
    "safetyNotes": ["Wear goggles in the lab."],
    "procedure": ["Step 1...", "Step 2..."],
    "hypothesisPrompt": "What do you predict?"
  },
  "guidedExploration": ["What happens if you increase light?", "Compare real-world and microscopic views."]
}
```

Experiment type guide:
- Plant growth → plant-growth | Seed sprouting → seed-germination
- Photosynthesis → photosynthesis | Breathing/cellular respiration → respiration
- Digestion → digestion | Heart/blood → blood-circulation
- Magnets → magnetism | Circuits → electricity
- Mirror → light-reflection | Light in water → refraction
- Acids/bases/pH → acids-bases | Burning/reactions → chemical-reaction
- Ice/water/steam → states-of-matter | Heat flow → heat-transfer
- Rain cycle → water-cycle | Sound waves → sound
- Push/pull/motion → force-motion | Planets → solar-system
- Body organs → human-organs
Always include sliders AND threeViews. Press ▶ Play via animate button."""


def should_use_interactive_science_experiment(
    *,
    subject_name: str,
    answer_type: str,
    query: str,
    heading_scope: Any | None = None,
    class_level: str = "",
) -> bool:
    if not _is_science_subject(subject_name):
        return False
    if not (query or "").strip():
        return False
    if answer_type in _SKIP_TYPES:
        return False
    from app.services.section_heading import HeadingScope

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        return False

    if _EXPLICIT_EXPERIMENT_RE.search(query):
        return True

    from app.services.science_experiment.fallbacks import _GENERIC_TYPES
    from app.services.science_experiment.experiment_catalog import match_science_experiment

    catalog = match_science_experiment(query, class_level)
    exp_type = str((catalog.get("experiment") or {}).get("experimentType") or "")
    return exp_type not in _GENERIC_TYPES


def _is_science_subject(subject_name: str) -> bool:
    s = (subject_name or "").lower()
    keywords = ("science", "physics", "chemistry", "biology", "evs", "environmental")
    return any(kw in s for kw in keywords)


def get_experiment_appendix_prompt() -> str:
    return _SCIENCE_EXPERIMENT_APPENDIX


def clean_tutor_answer_text(answer: str) -> str:
    text = answer or ""
    text = _MARKDOWN_FENCE_RE.sub(r"\1", text)
    text = _INCOMPLETE_SCIENCE_EXPERIMENT_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def finalize_science_answer(
    answer: str,
    query: str,
    *,
    class_level: str = "",
    subject_name: str = "",
    existing_experiment: dict[str, Any] | None = None,
    allow_fallback: bool = True,
) -> tuple[str, dict[str, Any] | None]:
    clean, experiment = extract_science_experiment_from_answer(answer)
    if experiment is None:
        experiment = existing_experiment
    if not _is_science_subject(subject_name):
        return clean, experiment
    if experiment is None and not allow_fallback:
        return clean, None
    catalog = build_fallback_science_experiment(query, class_level) if allow_fallback else None
    if experiment is None:
        if catalog is None:
            return clean, None
        cat_type = str((catalog.get("experiment") or {}).get("experimentType") or "")
        if cat_type in _GENERIC_TYPES:
            return clean, None
        logger.info("Using science experiment catalog for query=%r", query[:80])
        return clean, catalog
    if catalog is None:
        llm_type = str((experiment.get("experiment") or {}).get("experimentType") or "")
        if llm_type in _GENERIC_TYPES:
            return clean, None
        return clean, experiment
    merged = merge_catalog_experiment(experiment, catalog)
    if merged is not experiment:
        logger.info(
            "Merged catalog experiment %s for query=%r",
            catalog.get("experiment", {}).get("experimentType"),
            query[:80],
        )
    merged_type = str((merged.get("experiment") or {}).get("experimentType") or "")
    if merged_type in _GENERIC_TYPES:
        return clean, None
    return clean, merged


def extract_science_experiment_from_answer(answer: str) -> tuple[str, dict[str, Any] | None]:
    clean, experiment = strip_science_experiment_block(answer)
    clean = clean_tutor_answer_text(clean)
    if experiment is None:
        return clean_tutor_answer_text(answer), None
    try:
        validated = ScienceExperiment.model_validate(experiment)
        return clean, validated.model_dump(mode="json")
    except Exception as exc:
        logger.warning("Science experiment validation failed: %s", exc)
        return clean, experiment


def strip_science_experiment_block(answer: str) -> tuple[str, dict[str, Any] | None]:
    match = _SCIENCE_EXPERIMENT_FENCE_RE.search(answer or "")
    if not match:
        return clean_tutor_answer_text(answer or ""), None
    try:
        raw = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return clean_tutor_answer_text(answer or ""), None
    clean = (answer[: match.start()] + answer[match.end() :]).strip()
    clean = clean_tutor_answer_text(clean)
    return clean, raw if isinstance(raw, dict) else None
