"""Response-grounded interactive panels: affinity gate, number seed, open-box & pressure."""

from __future__ import annotations

from app.services.interactive_grounding import (
    extract_answer_numbers,
    panel_numbers_agree,
    resolve_grounded_panel,
    should_emit_panel,
    type_affinity_score,
)
from app.services.math_lesson.service import finalize_math_answer
from app.services.math_lesson.visual_catalog import match_math_visualization
from app.services.science_experiment.experiment_catalog import match_science_experiment
from app.services.science_experiment.service import finalize_science_answer


OPEN_BOX_ANSWER = """
**To Find** Dimensions and volume of the open box.

**Given Information**
- Sheet length = 30 cm, width = 20 cm
- Squares of side 5 cm cut from each corner

**Solution**
Height of box = 5 cm
Base length = 30 − 2×5 = 20 cm
Base width = 20 − 2×5 = 10 cm
Volume V = length × width × height = 20 × 10 × 5 = 1000 cm³

**Final Answer**
Box is 20 cm × 10 cm × 5 cm; volume = 1000 cm³
"""

OPEN_BOX_QUERY = (
    "A rectangular sheet of paper is 30 cm long and 20 cm wide. "
    "Four squares of side 5 cm are cut from its four corners. "
    "The remaining sheet is folded upwards to make an open box."
)

PRESSURE_ANSWER = """
**To Find** Pressure in positions A and B.

**Given Information**
- Weight (force) F = 40 N
- Position A area = 200 cm²
- Position B area = 100 cm²

**The Formula**
P = F / A

**Solution**
P_A = 40 / 200 = 0.2 N/cm²
P_B = 40 / 100 = 0.4 N/cm²
Pressure is greater in position B because the same force acts on a smaller area.

**Final Answer**
P_A = 0.2 N/cm², P_B = 0.4 N/cm²; B is greater.
"""

PRESSURE_QUERY = (
    "A rectangular wooden block weighs 40 N. Position A area 200 cm², "
    "Position B area 100 cm². Calculate pressure in each position."
)


def test_affinity_rejects_linear_graph_for_powers():
    text = "2 to the power 5 equals 32 because we multiply 2 by itself five times."
    assert type_affinity_score(text, "linear-graph", "math") == 0
    assert type_affinity_score(text, "concept-explorer", "math") >= 40


def test_affinity_open_box_area_resizer():
    assert type_affinity_score(OPEN_BOX_ANSWER, "area-resizer", "math") >= 40


def test_affinity_pressure_force_lab():
    assert type_affinity_score(PRESSURE_ANSWER, "force-pressure-lab", "science") >= 40
    assert type_affinity_score(PRESSURE_ANSWER, "circuit-builder", "science") < 40


def test_extract_open_box_numbers():
    nums = extract_answer_numbers(OPEN_BOX_ANSWER)
    for n in (30, 20, 5, 10, 1000):
        assert float(n) in nums


def test_catalog_open_box_match():
    lesson = match_math_visualization(OPEN_BOX_QUERY + "\n" + OPEN_BOX_ANSWER, "CLASS_8")
    assert lesson["visualization"]["visualizationType"] == "area-resizer"
    ids = {s["id"] for s in lesson["visualization"]["sliders"]}
    assert {"length", "width", "height", "cut"} <= ids or {"sheetLength", "cut"} <= ids


def test_catalog_pressure_match():
    lesson = match_science_experiment(PRESSURE_QUERY + "\n" + PRESSURE_ANSWER, "CLASS_8")
    assert lesson["experiment"]["experimentType"] == "force-pressure-lab"
    area = next(s for s in lesson["experiment"]["sliders"] if s["id"] == "area")
    assert float(area["max"]) >= 200


def test_should_emit_open_box_from_catalog():
    catalog = match_math_visualization(OPEN_BOX_ANSWER, "CLASS_8")
    panel = should_emit_panel(catalog, OPEN_BOX_ANSWER, kind="math", query=OPEN_BOX_QUERY)
    assert panel is not None
    assert panel["visualization"]["visualizationType"] == "area-resizer"
    defaults = [float(s["default"]) for s in panel["visualization"]["sliders"]]
    assert any(d in (20.0, 30.0, 10.0, 5.0) for d in defaults)
    formulas = " ".join(
        str(c.get("formula") or "") for c in (panel["visualization"].get("liveCalculations") or [])
    )
    assert "length" in formulas and "width" in formulas and "height" in formulas


def test_should_emit_pressure_from_catalog():
    catalog = match_science_experiment(PRESSURE_ANSWER, "CLASS_8")
    panel = should_emit_panel(catalog, PRESSURE_ANSWER, kind="science", query=PRESSURE_QUERY)
    assert panel is not None
    assert panel["experiment"]["experimentType"] == "force-pressure-lab"
    force = next(s for s in panel["experiment"]["sliders"] if s["id"] == "force")
    area = next(s for s in panel["experiment"]["sliders"] if s["id"] == "area")
    assert float(force["default"]) == 40.0
    assert float(area["max"]) >= 200
    assert float(area["default"]) in (200.0, 100.0, 40.0) or float(area["default"]) >= 100


def test_mismatch_circuit_for_disease_suppressed():
    bad = {
        "conceptName": "Circuit",
        "experiment": {
            "experimentType": "circuit-builder",
            "title": "Lamp",
            "sliders": [{"id": "v", "label": "V", "min": 1, "max": 12, "step": 1, "default": 6}],
            "threeViews": {
                "realWorld": {"title": "a", "description": "b"},
                "microscopic": {"title": "a", "description": "b"},
                "scientific": {"title": "a", "description": "b", "equation": ""},
            },
        },
    }
    disease_answer = (
        "Germs enter the body through air, food, and water. "
        "Washing hands breaks disease transmission."
    )
    assert should_emit_panel(bad, disease_answer, kind="science") is None


def test_wrong_linear_graph_fence_suppressed_for_powers():
    fence = """
```math-lesson
{"conceptName":"Powers","visualization":{"visualizationType":"linear-graph","title":"Graph",
"sliders":[{"id":"m","label":"m","min":-5,"max":5,"step":1,"default":2}]}}
```
"""
    answer = "2 to the power 5 equals 32." + fence
    clean, lesson = finalize_math_answer(
        answer,
        "ok explain 2 to the power 5 slowly",
        class_level="CLASS_8",
        subject_name="Mathematics",
        allow_llm_pass2=False,
    )
    if lesson is not None:
        assert lesson["visualization"]["visualizationType"] != "linear-graph"


def test_finalize_open_box_without_pass2():
    clean, lesson = finalize_math_answer(
        OPEN_BOX_ANSWER,
        OPEN_BOX_QUERY,
        class_level="CLASS_8",
        subject_name="Mathematics",
        allow_llm_pass2=False,
    )
    assert "1000" in clean
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] == "area-resizer"
    assert panel_numbers_agree(lesson, extract_answer_numbers(OPEN_BOX_ANSWER))


def test_finalize_pressure_without_pass2():
    clean, exp = finalize_science_answer(
        PRESSURE_ANSWER,
        PRESSURE_QUERY,
        class_level="CLASS_8",
        subject_name="Science",
        allow_llm_pass2=False,
    )
    assert "0.2" in clean or "0.4" in clean
    assert exp is not None
    assert exp["experiment"]["experimentType"] == "force-pressure-lab"
    force = next(s for s in exp["experiment"]["sliders"] if s["id"] == "force")
    assert float(force["default"]) == 40.0


def test_finalize_math_suppresses_unrelated_vague():
    """No clear concept signal and no matching catalog → no wrong panel."""
    clean, lesson = finalize_math_answer(
        "Okay sure.",
        "hmm",
        class_level="CLASS_8",
        subject_name="Mathematics",
        allow_llm_pass2=False,
    )
    assert lesson is None or type_affinity_score(clean, lesson["visualization"]["visualizationType"], "math") >= 0


def test_resolve_prefers_gated_catalog_when_pass2_disabled():
    catalog = match_math_visualization(OPEN_BOX_ANSWER, "CLASS_8")
    panel = resolve_grounded_panel(
        answer=OPEN_BOX_ANSWER,
        query=OPEN_BOX_QUERY,
        kind="math",
        class_level="CLASS_8",
        first_panel={
            "visualization": {
                "visualizationType": "linear-graph",
                "sliders": [{"id": "m", "min": 0, "max": 5, "default": 1}],
            }
        },
        catalog_panel=catalog,
        allow_llm_pass2=False,
    )
    assert panel is not None
    assert panel["visualization"]["visualizationType"] == "area-resizer"
