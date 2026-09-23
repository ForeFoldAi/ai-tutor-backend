"""Production templates + typed extractors."""

from app.services.interactive_grounding import resolve_grounded_panel, should_emit_panel
from app.services.interactive_templates import (
    build_template_panel,
    extract_typed_params,
    params_complete,
)
from app.services.math_lesson.service import finalize_math_answer
from app.services.science_experiment.service import finalize_science_answer

OPEN_BOX = """
**Solution**
Sheet 30 cm by 20 cm. Squares of side 5 cm cut from corners.
Base length = 20 cm. Base width = 10 cm. Height = 5 cm.
Volume V = 20 × 10 × 5 = 1000 cm³
**Final Answer**
20 cm × 10 cm × 5 cm; volume = 1000 cm³
"""

POWERS = """
2 to the power 5 means 2 × 2 × 2 × 2 × 2.
2 × 2 = 4, 4 × 2 = 8, 8 × 2 = 16, 16 × 2 = 32.
So 2^5 = 32.
"""

PRESSURE = """
Force F = 40 N.
Position A area = 200 cm². Position B area = 100 cm².
P = F / A
P_A = 40 / 200 = 0.2 N/cm²
P_B = 40 / 100 = 0.4 N/cm²
"""


def test_extract_open_box_complete():
    vt, params, pack = extract_typed_params(OPEN_BOX, "math")
    assert vt == "area-resizer"
    assert pack == "open_box"
    assert params_complete(vt, params, pack=pack)
    assert params["sheetLength"] == 30
    assert params["length"] == 20
    assert params["width"] == 10
    assert params["height"] == 5


def test_extract_open_box_ignores_equation_intermediate():
    """'Base length = 30 − 2×5 = 20' must not trap length on 30."""
    text = (
        "Original sheet dimensions: length = 30 cm, width = 20 cm. "
        "Squares cut from each corner: side = 5 cm. Open box. "
        "Base length = 30 − 2×5 = 20 cm. Base width = 20 − 2×5 = 10 cm. Height = 5 cm."
    )
    vt, params, pack = extract_typed_params(text, "math")
    assert pack == "open_box"
    assert params["length"] == 20 and params["width"] == 10 and params["cut"] == 5


def test_extract_open_box_ignores_practice_question_nums():
    text = (
        OPEN_BOX
        + "\n\n**Practice Question**\n"
        "A sheet is 40 cm by 25 cm. Squares of side 4 cm are cut from each corner."
    )
    vt, params, pack = extract_typed_params(text, "math")
    assert pack == "open_box"
    assert params["cut"] == 5 and params["length"] == 20 and params["width"] == 10


def test_extract_powers_complete():
    vt, params, pack = extract_typed_params(POWERS, "math")
    assert vt == "concept-explorer"
    assert pack == "powers"
    assert params_complete(vt, params, pack=pack)
    assert params == {"base": 2.0, "exponent": 5.0, "result": 32.0}


def test_extract_pressure_complete():
    vt, params, pack = extract_typed_params(PRESSURE, "science")
    assert vt == "force-pressure-lab"
    assert params["force"] == 40
    assert params["area"] == 200
    assert params.get("area_b") == 100
    assert params_complete(vt, params, pack=pack)


def test_template_open_box_defaults():
    vt, params, pack = extract_typed_params(OPEN_BOX, "math")
    panel = build_template_panel(vt, params, kind="math", class_level="CLASS_8", pack=pack)
    assert panel is not None
    by = {s["id"]: float(s["default"]) for s in panel["visualization"]["sliders"]}
    assert by["sheetLength"] == 30
    assert by["sheetWidth"] == 20
    assert by["cut"] == 5
    assert by["length"] == 20
    assert by["width"] == 10
    assert by["height"] == 5


def test_template_powers_defaults():
    vt, params, pack = extract_typed_params(POWERS, "math")
    panel = build_template_panel(vt, params, kind="math", pack=pack)
    by = {s["id"]: float(s["default"]) for s in panel["visualization"]["sliders"]}
    assert by == {"base": 2.0, "exponent": 5.0, "result": 32.0}


def test_extract_health_habits_not_disease():
    text = (
        "Our body needs nutritious food, clean water, exercise, and rest to stay healthy. "
        "Eating a balanced diet keeps us strong."
    )
    vt, params, pack = extract_typed_params(text, "science")
    assert vt == "human-body-system-3d"
    assert pack == "health"
    assert params_complete(vt, params, pack=pack)


def test_extract_wash_hands_is_disease():
    text = "We should wash our hands so germs do not spread from one person to another."
    vt, _params, pack = extract_typed_params(text, "science")
    assert vt == "disease-transmission-simulator"
    assert pack == "disease"


def test_template_pressure_defaults():
    vt, params, pack = extract_typed_params(PRESSURE, "science")
    panel = build_template_panel(vt, params, kind="science", pack=pack)
    by = {s["id"]: float(s["default"]) for s in panel["experiment"]["sliders"]}
    assert by["force"] == 40
    assert by["area"] == 200
    assert float(next(s["max"] for s in panel["experiment"]["sliders"] if s["id"] == "area")) >= 200


def test_resolve_tier_a_skips_pass2_open_box():
    panel = resolve_grounded_panel(
        answer=OPEN_BOX,
        query="open box from sheet cut corners",
        kind="math",
        class_level="CLASS_8",
        allow_llm_pass2=False,
    )
    assert panel is not None
    by = {s["id"]: float(s["default"]) for s in panel["visualization"]["sliders"]}
    assert by["length"] == 20 and by["width"] == 10 and by["height"] == 5


def test_resolve_tier_a_powers():
    panel = resolve_grounded_panel(
        answer=POWERS,
        query="explain 2 to the power 5",
        kind="math",
        class_level="CLASS_8",
        first_panel={
            "visualization": {
                "visualizationType": "algebra-stepper",
                "title": "x",
                "sliders": [],
                "algebraSteps": [],
            }
        },
        allow_llm_pass2=False,
    )
    assert panel is not None
    assert panel["visualization"]["visualizationType"] == "concept-explorer"
    by = {s["id"]: float(s["default"]) for s in panel["visualization"]["sliders"]}
    assert by["result"] == 32


def test_finalize_open_box_production_numbers():
    clean, lesson = finalize_math_answer(
        OPEN_BOX,
        "sheet 30 by 20 cut 5 open box volume",
        class_level="CLASS_8",
        subject_name="Mathematics",
        allow_llm_pass2=False,
    )
    assert lesson is not None
    by = {s["id"]: float(s["default"]) for s in lesson["visualization"]["sliders"]}
    assert by["length"] == 20 and by["width"] == 10 and by["height"] == 5


def test_finalize_powers_production():
    clean, lesson = finalize_math_answer(
        POWERS,
        "ok explain 2 to the power 5 slowly",
        class_level="CLASS_8",
        subject_name="Mathematics",
        allow_llm_pass2=False,
    )
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] == "concept-explorer"
    by = {s["id"]: float(s["default"]) for s in lesson["visualization"]["sliders"]}
    assert by["base"] == 2 and by["exponent"] == 5 and by["result"] == 32


SQRT5_QUERY = (
    "draw a number line and by geometrical construction mark square root 5 on the number line"
)

SQRT5_ANSWER = """
**To Find**
Locate √5 on a number line by geometrical construction.

**Given Information**
Use a right triangle with sides 1 and 2 (since 1² + 2² = 5). Compass and straightedge.

**Concept Behind It**
A number multiplied by itself gives a square. Here 1² + 2² = 5, so the hypotenuse is √5.

**The Formula**
a² + b² = c²

**Steps**
1. Draw a number line, mark O at 0 and A at 2.
2. Erect perpendicular AB = 1.
3. Join OB; OB = √5.
4. With centre O and radius OB, mark C on the line. C represents √5.
"""


def test_extract_sqrt5_construction_not_powers():
    text = f"{SQRT5_ANSWER}\n{SQRT5_QUERY}"
    vt, params, pack = extract_typed_params(text, "math")
    assert pack == "sqrt_line"
    assert vt == "sqrt-number-line"
    assert params["n"] == 5.0
    assert params.get("a") == 2.0 and params.get("b") == 1.0


def test_resolve_sqrt5_keeps_interactive_not_power_play():
    catalog = {
        "conceptName": "Square Root on the Number Line",
        "visualization": {
            "visualizationType": "sqrt-number-line",
            "title": "Construct √n on the number line",
            "sliders": [
                {"id": "n", "label": "n", "min": 2, "max": 50, "step": 1, "default": 5},
                {"id": "a", "label": "a", "min": 1, "max": 20, "step": 1, "default": 2},
                {"id": "b", "label": "b", "min": 1, "max": 20, "step": 1, "default": 1},
            ],
        },
    }
    panel = resolve_grounded_panel(
        answer=SQRT5_ANSWER,
        query=SQRT5_QUERY,
        kind="math",
        class_level="CLASS_8",
        first_panel=catalog,
        catalog_panel=catalog,
        allow_llm_pass2=False,
    )
    assert panel is not None
    assert panel["visualization"]["visualizationType"] == "sqrt-number-line"
    assert panel["visualization"].get("title") != "Power Play"


def test_finalize_sqrt5_construction():
    clean, lesson = finalize_math_answer(
        SQRT5_ANSWER,
        SQRT5_QUERY,
        class_level="CLASS_8",
        subject_name="Mathematics",
        allow_llm_pass2=False,
    )
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] == "sqrt-number-line"
    assert lesson["visualization"].get("title") != "Power Play"


def test_fractions_pack_not_from_bare_ratio_in_equation():
    text = (
        "Solve 2x + 5 = 15.\n"
        "Subtract 5: 2x = 10.\n"
        "Divide by 2: x = 10/2 = 5.\n"
        "So x = 5."
    )
    vt, _params, pack = extract_typed_params(text, "math")
    assert pack != "fractions"


def test_finalize_equation_keeps_algebra_not_fractions():
    answer = (
        "Solve 2x + 5 = 15.\n"
        "2x = 10, so x = 10/2 = 5.\n"
        "Check: 2(5)+5=15."
    )
    clean, lesson = finalize_math_answer(
        answer,
        "solve 2x + 5 = 15",
        class_level="CLASS_8",
        subject_name="Mathematics",
        allow_llm_pass2=False,
    )
    assert lesson is not None
    assert lesson["visualization"]["visualizationType"] != "fractions"


def test_finalize_pressure_production():
    clean, exp = finalize_science_answer(
        PRESSURE,
        "block 40 N areas 200 and 100 pressure",
        class_level="CLASS_8",
        subject_name="Science",
        allow_llm_pass2=False,
    )
    assert exp is not None
    assert exp["experiment"]["experimentType"] == "force-pressure-lab"
    by = {s["id"]: float(s["default"]) for s in exp["experiment"]["sliders"]}
    assert by["force"] == 40 and by["area"] == 200
