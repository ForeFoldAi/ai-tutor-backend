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
    r"```(?:science-experiment|science-lesson|json:science-experiment|science_experiment|science_lesson)\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)

_INCOMPLETE_SCIENCE_EXPERIMENT_RE = re.compile(
    r"\n?```(?:science-experiment|science-lesson|json:science-experiment|science_experiment|science_lesson)\s*\n[\s\S]*$",
    re.IGNORECASE,
)

# Real-world handling verbs without simulation framing → strip procedure to safe catalog copy.
_UNSAFE_PROCEDURE_RE = re.compile(
    r"\b(pour|light\s+the|heat\s+over|mix\s+acid|taste|swallow|inhale)\b",
    re.I,
)
_SIM_FRAME_RE = re.compile(r"\b(simulation|virtual|on[\s-]?screen|in\s+this\s+lab\s+view)\b", re.I)

_MARKDOWN_FENCE_RE = re.compile(r"```markdown\s*\n([\s\S]*?)```", re.IGNORECASE)

_SKIP_TYPES = frozenset({
    "greeting", "affirmation", "personal-response", "one-word", "brief",
    # "factual" intentionally NOT skipped — Health/Science teaching Qs often
    # classify as factual; appendix + finalize still need to run.
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
INTERACTIVE SCIENCE LAB (MANDATORY for science answers — append after prose):

Choose subject: physics | chemistry | biology | evs
Choose kind: concept (understand) | experiment (virtual guided lab). Prefer experiment when the student says show/test/demonstrate/experiment.

THREE SYNCHRONIZED VIEWS (always):
1. realWorld — "What you see"
2. microscopic — "What's inside" (particles/cells/charges)
3. scientific — "Why it happens" (+ equation when useful)

SAFETY (non-negotiable):
- Every lab is a VIRTUAL on-screen simulation. Procedure steps must say "In this simulation…"
- Never give real hazardous handling, dosages, or "do this at home with chemicals/flames".
- Adolescence / reproduction topics: factual NCERT tone only; use life-cycle-animator or human-body-basics — never graphic.
- Disease / natural disasters: factual, non-dramatized.

RULES:
- Full tutor answer FIRST, then ONE ```science-experiment``` JSON block (valid + closed).
- Do NOT wrap the answer in ```markdown fences.
- Use catalog experimentType exactly when a catalog hint is provided.
- Prefer ontology types: plant-anatomy-lab, circuit-builder, acid-base-indicator-lab,
  states-of-matter-lab, motion-grapher, light-optics-bench, cell-structure-3d,
  molecule-builder-3d, atom-structure-3d, sound-wave-lab, force-pressure-lab,
  water-cycle-animator, food-chain-simple, human-body-system-3d, reaction-simulator,
  magnet-field-visualizer, solar-system-3d, periodic-table-explorer, heredity-punnett-lab,
  ecosystem-lab, microscope-lab, energy-transformation-lab, concept-explorer, …

Worked concept (2D food chain):
```science-experiment
{"conceptName":"Food chain","classLevel":"Class 5","subject":"biology","kind":"concept","learningObjective":"Trace energy flow","conceptExplanation":"Energy moves from producers to consumers.","experiment":{"experimentType":"food-chain-simple","title":"Food chain steps","description":"Step through sun→plant→deer→tiger","gradeTier":"primary","subject":"biology","kind":"concept","threeViews":{"realWorld":{"title":"Chain","description":"Living things linked by who eats whom"},"microscopic":{"title":"Energy","description":"Energy packets move along arrows"},"scientific":{"title":"Idea","description":"Producers make food; consumers eat","equation":""}},"sliders":[{"id":"level","label":"Chain step","min":0,"max":3,"step":1,"default":0}],"buttons":[{"id":"animate","label":"Play","action":"animate"},{"id":"reset","label":"Reset","action":"reset"}],"safetyNotes":[],"procedure":[],"hypothesisPrompt":"What happens if plants disappear?"},"guidedExploration":["Move the chain step","Explain each arrow"]}
```

Worked experiment (acid–base indicator — virtual only):
```science-experiment
{"conceptName":"Acids and bases","classLevel":"Class 7","subject":"chemistry","kind":"experiment","learningObjective":"Read pH with indicator colour","conceptExplanation":"Indicators change colour with pH.","experiment":{"experimentType":"acid-base-indicator-lab","title":"Virtual pH beaker","description":"Slide pH and watch colour","gradeTier":"middle","subject":"chemistry","kind":"experiment","aim":"See how indicators show acid/base","apparatus":["Virtual beaker","Virtual indicator"],"safetyLevel":"caution","threeViews":{"realWorld":{"title":"Beaker","description":"Colour of the liquid"},"microscopic":{"title":"Ions","description":"More H+ means lower pH"},"scientific":{"title":"Rule","description":"Acid pH<7, base pH>7","equation":"pH = -log[H+]"}},"sliders":[{"id":"ph","label":"pH","min":0,"max":14,"step":0.5,"default":7}],"buttons":[{"id":"animate","label":"Play","action":"animate"},{"id":"reset","label":"Reset","action":"reset"}],"safetyNotes":["Virtual chemicals only — never taste unknowns."],"procedure":["In this simulation, move the pH slider from 2 to 12.","In this simulation, pause at pH 7 and note the colour."],"hypothesisPrompt":"What colour do you expect for a strong acid?"},"guidedExploration":["Find the neutral colour","Compare acid vs base"]}
```

Always include sliders AND threeViews. Use ▶ Play via animate button."""


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

    from app.services.science_experiment.experiment_catalog import match_science_experiment
    from app.services.science_experiment.topic_ontology import pick_ontology_topic

    if pick_ontology_topic(query, class_level) is not None:
        return True

    catalog = match_science_experiment(query, class_level)
    exp_type = str((catalog.get("experiment") or {}).get("experimentType") or "")
    if exp_type not in _GENERIC_TYPES and exp_type != "concept-explorer":
        return True
    # Soft teach: any real science question (≥4 tokens) still gets the appendix;
    # finalize suppresses pure concept-explorer when affinity is empty.
    return len((query or "").split()) >= 4


def _is_science_subject(subject_name: str) -> bool:
    s = (subject_name or "").lower()
    keywords = ("science", "physics", "chemistry", "biology", "evs", "environmental")
    return any(kw in s for kw in keywords)


def get_experiment_appendix_prompt() -> str:
    return _SCIENCE_EXPERIMENT_APPENDIX


def _lint_procedure_safety(experiment: dict[str, Any]) -> dict[str, Any]:
    """Ensure procedure steps are framed as virtual simulation."""
    exp = experiment.get("experiment")
    if not isinstance(exp, dict):
        return experiment
    out = dict(experiment)
    exp2 = dict(exp)
    cleaned_proc = []
    for step in exp2.get("procedure") or []:
        text = str(step)
        if _UNSAFE_PROCEDURE_RE.search(text) and not _SIM_FRAME_RE.search(text):
            cleaned_proc.append(f"In this simulation: {text}")
        else:
            cleaned_proc.append(text)
    exp2["procedure"] = cleaned_proc
    steps_obj = []
    for step in exp2.get("procedureSteps") or []:
        if not isinstance(step, dict):
            continue
        s = dict(step)
        instr = str(s.get("instruction") or "")
        if _UNSAFE_PROCEDURE_RE.search(instr) and not _SIM_FRAME_RE.search(instr):
            s["instruction"] = f"In this simulation: {instr}"
        steps_obj.append(s)
    if steps_obj:
        exp2["procedureSteps"] = steps_obj
    if exp2.get("kind") == "experiment" and not exp2.get("safetyNotes"):
        exp2["safetyNotes"] = ["This is a virtual simulation — do not try hazardous steps at home."]
    out["experiment"] = exp2
    return out


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
    allow_llm_pass2: bool = True,
) -> tuple[str, dict[str, Any] | None]:
    """Extract/build science experiment grounded on the answer; suppress on mismatch."""
    from app.services.interactive_grounding import match_text, resolve_grounded_panel

    clean, experiment = extract_science_experiment_from_answer(answer)
    if experiment is None:
        experiment = existing_experiment
    if not _is_science_subject(subject_name):
        return clean, experiment
    if not allow_fallback and experiment is None:
        return clean, None

    text = match_text(clean, query)
    catalog = build_fallback_science_experiment(text, class_level) if allow_fallback else None
    if experiment is not None and catalog is not None:
        experiment = merge_catalog_experiment(experiment, catalog)

    grounded = resolve_grounded_panel(
        answer=clean,
        query=query,
        kind="science",
        class_level=class_level,
        first_panel=experiment,
        catalog_panel=catalog,
        allow_llm_pass2=allow_llm_pass2 and allow_fallback,
    )
    if grounded is None and catalog is not None and allow_fallback:
        # Only fail-open when the QUERY itself maps to a real lab type (not when
        # answer prose alone dragged a catalog hit — keeps "What is science?" empty).
        query_catalog = build_fallback_science_experiment(query, class_level)
        exp_type = str((query_catalog.get("experiment") or {}).get("experimentType") or "")
        if exp_type not in _GENERIC_TYPES and exp_type != "concept-explorer":
            grounded = query_catalog
    if grounded is None:
        return clean, None
    return clean, _lint_procedure_safety(grounded)


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
