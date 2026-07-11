"""Tests for interactive science experiment extraction and catalog matching."""

from app.services.science_experiment.experiment_catalog import match_science_experiment
from app.services.science_experiment.fallbacks import build_fallback_science_experiment, merge_catalog_experiment
from app.services.science_experiment.schemas import ScienceExperiment
from app.services.science_experiment.service import (
    extract_science_experiment_from_answer,
    finalize_science_answer,
    should_use_interactive_science_experiment,
    strip_science_experiment_block,
)

SAMPLE_JSON = """
{
  "conceptName": "Photosynthesis",
  "classLevel": "Class 7",
  "learningObjective": "Understand how plants make food",
  "conceptExplanation": "Plants use light to make glucose.",
  "experiment": {
    "experimentType": "photosynthesis",
    "title": "Leaf Lab",
    "description": "Explore CO2 and light",
    "gradeTier": "middle",
    "threeViews": {
      "realWorld": {"title": "Leaf in sun", "description": "Bubbles of O2"},
      "microscopic": {"title": "Chloroplast", "description": "Light reactions"},
      "scientific": {"title": "Equation", "description": "6CO2 + 6H2O → glucose", "equation": "6CO2 + 6H2O → C6H12O6 + 6O2"}
    },
    "sliders": [{"id": "light", "label": "Light", "min": 0, "max": 100, "step": 5, "default": 50}],
    "buttons": [{"id": "animate", "label": "Play", "action": "animate"}]
  },
  "guidedExploration": ["What if light is zero?"]
}
"""


def test_strip_science_experiment_block():
    answer = f"**Topic**\nPlants\n\n```science-experiment\n{SAMPLE_JSON}\n```"
    clean, exp = strip_science_experiment_block(answer)
    assert "science-experiment" not in clean
    assert exp is not None
    assert exp["conceptName"] == "Photosynthesis"


def test_extract_validates_experiment():
    answer = f"Explain photosynthesis.\n\n```science-experiment\n{SAMPLE_JSON}\n```"
    clean, exp = extract_science_experiment_from_answer(answer)
    assert exp is not None
    validated = ScienceExperiment.model_validate(exp)
    assert validated.experiment.experimentType == "photosynthesis"
    assert validated.experiment.threeViews.realWorld.title == "Leaf in sun"


def test_should_use_for_science_subject():
    assert should_use_interactive_science_experiment(
        subject_name="Science",
        answer_type="paragraph",
        query="Explain photosynthesis experiment",
    )


def test_should_skip_factual_materials_question():
    assert not should_use_interactive_science_experiment(
        subject_name="Science",
        answer_type="factual",
        query="What kind of materials do we need to make a lamp glow?",
        class_level="CLASS_9",
    )


def test_lamp_glow_matches_electricity_catalog():
    lesson = build_fallback_science_experiment(
        "What kind of materials do we need to make a lamp glow?",
        "CLASS_9",
    )
    assert lesson["experiment"]["experimentType"] == "electricity"


def test_finalize_skips_generic_concept_explorer():
    clean, exp = finalize_science_answer(
        "Conductors carry current.",
        "What is science?",
        class_level="CLASS_8",
        subject_name="Science",
    )
    assert exp is None


def test_should_skip_for_greeting():
    assert not should_use_interactive_science_experiment(
        subject_name="Science",
        answer_type="greeting",
        query="Hello",
    )


def test_finalize_science_answer_uses_fallback():
    clean, exp = finalize_science_answer(
        "Plants make food using sunlight.",
        "Explain photosynthesis with an experiment",
        class_level="CLASS_7",
        subject_name="Science",
    )
    assert exp is not None
    assert exp["experiment"]["experimentType"] == "photosynthesis"
    assert exp["experiment"]["threeViews"]["microscopic"]["title"]


def test_photosynthesis_catalog_match():
    lesson = build_fallback_science_experiment("Explain photosynthesis in plants", "CLASS_7")
    assert lesson["experiment"]["experimentType"] == "photosynthesis"


def test_chemical_reaction_catalog_match():
    lesson = build_fallback_science_experiment("Magnesium ribbon burning experiment", "CLASS_9")
    assert lesson["experiment"]["experimentType"] == "chemical-reaction"


def test_merge_catalog_overrides_weak_llm():
    llm = {
        "conceptName": "Reaction",
        "experiment": {"experimentType": "concept-explorer", "title": "Generic"},
    }
    catalog = build_fallback_science_experiment("What happens when magnesium burns?", "CLASS_9")
    merged = merge_catalog_experiment(llm, catalog)
    assert merged["experiment"]["experimentType"] == "chemical-reaction"


EXPERIMENT_CASES = [
    ("photosynthesis", "Explain photosynthesis with molecules entering leaves"),
    ("respiration", "How does cellular respiration work?"),
    ("electricity", "Show me a simple electric circuit experiment"),
    ("acids-bases", "Acids and bases pH indicator lab"),
    ("chemical-reaction", "Magnesium ribbon burning chemical reaction"),
    ("water-cycle", "Explain the water cycle experiment"),
    ("magnetism", "Magnetic field lines around a bar magnet"),
    ("force-motion", "Newton force and motion experiment"),
    ("solar-system", "Planets in the solar system orbit"),
    ("human-organs", "Human body organs and systems"),
]


def test_experiment_type_matching():
    for expected, query in EXPERIMENT_CASES:
        lesson = match_science_experiment(query, "CLASS_8")
        vtype = lesson["experiment"]["experimentType"]
        assert vtype == expected, f"Query: {query} got {vtype}"
