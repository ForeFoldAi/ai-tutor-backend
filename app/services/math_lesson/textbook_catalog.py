"""
Class 9 EM Mathematics textbook — topic → interactive visualization catalog.

Aligned with chapters: Polynomials, Geometry Elements, Lines & Angles,
Linear Equations, Triangles, Quadrilaterals, Statistics, Surface Areas,
Areas, Constructions, Probability, Circles.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.services.math_lesson.math_tokens import class_level_to_default_palette, default_colors


def _cube_scene() -> dict[str, Any]:
    return {
        "camera": {"position": [4.8, 3.2, 5.6], "target": [0, 0.65, 0], "fov": 40},
        "groundGrid": True,
        "objects": [
            {
                "id": "cube",
                "type": "box",
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "scaleDrivenBy": "s",
                "color": "primary",
                "roughness": 0.35,
                "metalness": 0.15,
                "wireframeAccent": True,
            }
        ],
    }


def _cylinder_scene() -> dict[str, Any]:
    return {
        "camera": {"position": [3.5, 2.8, 4.5], "target": [0, 0.8, 0], "fov": 45},
        "groundGrid": True,
        "objects": [
            {
                "id": "cyl",
                "type": "cylinder",
                "position": [0, 0.8, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "scaleDrivenBy": "r",
                "color": "secondary",
                "wireframeAccent": True,
            }
        ],
    }


def _heights_scene() -> dict[str, Any]:
    return {
        "camera": {"position": [6, 4, 8], "target": [2, 1, 0], "fov": 45},
        "groundGrid": True,
        "objects": [
            {
                "id": "tower",
                "type": "box",
                "position": [4, 1.2, 0],
                "scale": [0.5, 2.4, 0.5],
                "scaleDrivenBy": "angle",
                "color": "primary",
                "wireframeAccent": True,
            },
            {
                "id": "observer",
                "type": "sphere",
                "position": [0, 0.2, 0],
                "scale": [0.2, 0.2, 0.2],
                "color": "accent",
                "wireframeAccent": False,
            },
        ],
        "labels": [{"id": "angle", "text": "θ", "position": [1.5, 0.8, 0]}],
    }


def _viz(
    vtype: str,
    title: str,
    description: str,
    *,
    sliders: list[dict] | None = None,
    buttons: list[dict] | None = None,
    calcs: list[dict] | None = None,
    draggables: list[dict] | None = None,
    objects: list[dict] | None = None,
    interactions: list[dict] | None = None,
    animations: list[dict] | None = None,
    steps: list[dict] | None = None,
    algebra_steps: list[dict] | None = None,
    render_mode: str = "2d",
    graph_mode: str = "",
    curve_type: str = "",
    coefficients: list[float] | None = None,
    scene: dict[str, Any] | None = None,
    palette_id: str | None = None,
    finance_spec: dict[str, Any] | None = None,
    class_level: str = "Class 9",
) -> dict[str, Any]:
    pid = palette_id or class_level_to_default_palette(class_level)
    out: dict[str, Any] = {
        "visualizationType": vtype,
        "title": title,
        "description": description,
        "renderMode": render_mode,
        "paletteId": pid,
        "sliders": sliders or [],
        "buttons": buttons or [],
        "liveCalculations": calcs or [],
        "draggableObjects": draggables or [],
        "interactiveObjects": objects or [],
        "studentInteractions": interactions or [],
        "animations": animations or [],
        "colors": default_colors(pid),
    }
    if scene is not None:
        out["scene"] = scene
    if algebra_steps:
        out["algebraSteps"] = algebra_steps
    elif steps:
        # legacy → will be normalized by schema validator
        out["steps"] = steps
    ct = curve_type or (graph_mode if graph_mode in ("linear", "quadratic") else "")
    if ct:
        out["curveType"] = ct
        out["graphMode"] = ct  # compat
    if coefficients:
        out["coefficients"] = coefficients
    if finance_spec:
        out["financeSpec"] = finance_spec
    return out


def _lesson(
    concept: str,
    objective: str,
    explanation: str,
    visualization: dict[str, Any],
    level: str,
) -> dict[str, Any]:
    return {
        "conceptName": concept,
        "classLevel": level,
        "learningObjective": objective,
        "conceptExplanation": explanation,
        "visualization": visualization,
        "guidedExploration": [],
        "practiceMode": {"easy": "", "medium": "", "hard": "", "challenge": ""},
        "commonMistakes": [],
        "aiHints": [],
        "assessment": [],
    }


def _algebra_tiles_spec() -> dict[str, Any]:
    return _viz(
        "algebra-tiles",
        "Algebra Tiles — (a + b)²",
        "Drag the sliders to expand (a+b)². Discover why a²+b² is missing the 2ab term.",
        sliders=[
            {"id": "a", "label": "Side a", "min": 1, "max": 5, "step": 1, "default": 3},
            {"id": "b", "label": "Side b", "min": 1, "max": 5, "step": 1, "default": 2},
        ],
        calcs=[
            {"id": "correct", "label": "(a+b)²", "formula": "a * a + 2 * a * b + b * b", "unit": ""},
            {"id": "wrong", "label": "a²+b² (wrong)", "formula": "a * a + b * b", "unit": ""},
            {"id": "missing", "label": "Missing term 2ab", "formula": "2 * a * b", "unit": ""},
        ],
        buttons=[
            {"id": "expand", "label": "▶ Expand & Collapse", "action": "animate"},
            {"id": "reset", "label": "Reset", "action": "reset"},
        ],
        interactions=[
            {
                "id": "tiles",
                "type": "slide",
                "description": "Change a and b to see the area model grow.",
                "expectedObservation": "The middle rectangles (2ab) are the missing terms.",
            }
        ],
    )


def _factor_rectangle_spec() -> dict[str, Any]:
    return _viz(
        "factor-rectangle",
        "Rectangle Factorisation Model",
        "Drag the side sliders until the rectangle area matches x² + 7x + 12.",
        sliders=[
            {"id": "width", "label": "Width factor", "min": 1, "max": 12, "step": 1, "default": 3},
            {"id": "height", "label": "Height factor", "min": 1, "max": 12, "step": 1, "default": 4},
        ],
        calcs=[
            {"id": "product", "label": "Constant term (c)", "formula": "width * height", "unit": ""},
            {"id": "sum", "label": "Coefficient of x (b)", "formula": "width + height", "unit": ""},
        ],
        buttons=[{"id": "check", "label": "Check factors", "action": "animate"}],
        interactions=[
            {
                "id": "factor",
                "type": "drag",
                "description": "Find width and height whose product is 12 and sum is 7.",
                "expectedObservation": "3 and 4 work: (x+3)(x+4) = x²+7x+12.",
            }
        ],
    )


def _geometry_basics_spec() -> dict[str, Any]:
    return _viz(
        "geometry-basics",
        "Point · Line · Segment · Ray",
        "Select a tool and drag on the canvas. Extend a segment to see a ray or line.",
        buttons=[
            {"id": "point", "label": "Point", "action": "point"},
            {"id": "segment", "label": "Segment", "action": "segment"},
            {"id": "ray", "label": "Ray", "action": "ray"},
            {"id": "line", "label": "Line", "action": "line"},
            {"id": "animate", "label": "▶ Extend", "action": "animate"},
        ],
        draggables=[
            {"id": "A", "label": "A", "type": "point", "initialX": 40, "initialY": 80, "color": "#3B82F6"},
            {"id": "B", "label": "B", "type": "point", "initialX": 160, "initialY": 80, "color": "#10B981"},
        ],
    )


def _angle_explorer_spec() -> dict[str, Any]:
    return _viz(
        "angle-explorer",
        "Angle Explorer",
        "Drag the two rays and classify the angle: acute, right, obtuse or straight.",
        sliders=[
            {"id": "ray1", "label": "Ray 1 (degrees)", "min": 0, "max": 360, "step": 1, "default": 0},
            {"id": "ray2", "label": "Ray 2 (degrees)", "min": 0, "max": 360, "step": 1, "default": 45},
        ],
        calcs=[{"id": "angle", "label": "Angle measure", "formula": "ray2 - ray1", "unit": "°"}],
        buttons=[{"id": "animate", "label": "▶ Sweep", "action": "animate"}],
    )


def _parallel_transversal_spec() -> dict[str, Any]:
    return _viz(
        "parallel-transversal",
        "Parallel Lines & Transversal",
        "Drag the transversal and observe corresponding, alternate and interior angles.",
        sliders=[
            {"id": "transversal", "label": "Transversal angle", "min": 10, "max": 170, "step": 1, "default": 60},
        ],
        buttons=[
            {"id": "corresponding", "label": "Highlight corresponding", "action": "corresponding"},
            {"id": "alternate", "label": "Highlight alternate", "action": "alternate"},
            {"id": "animate", "label": "▶ Drag demo", "action": "animate"},
        ],
    )


def _linear_graph_spec() -> dict[str, Any]:
    return _viz(
        "linear-graph",
        "Graph y = mx + c",
        "Move the slope (m) and intercept (c) sliders to see the line change.",
        sliders=[
            {"id": "m", "label": "Slope (m)", "min": -5, "max": 5, "step": 0.5, "default": 1},
            {"id": "c", "label": "Intercept (c)", "min": -10, "max": 10, "step": 1, "default": 0},
        ],
        calcs=[{"id": "eq", "label": "Equation", "formula": "m * 1 + c", "unit": " at x=1"}],
        graph_mode="linear",
    )


def _quadratic_grapher_spec() -> dict[str, Any]:
    """Sub-mode of linear-graph: y = ax² + bx + c (visualizationType stays linear-graph)."""
    return _viz(
        "linear-graph",
        "Quadratic Grapher — y = ax² + bx + c",
        "Change a, b, c to move the parabola. Watch the discriminant and roots.",
        sliders=[
            {"id": "a", "label": "a (opens)", "min": -3, "max": 3, "step": 0.5, "default": 1},
            {"id": "b", "label": "b", "min": -5, "max": 5, "step": 0.5, "default": 0},
            {"id": "c", "label": "c", "min": -5, "max": 5, "step": 0.5, "default": -1},
        ],
        calcs=[
            {"id": "disc", "label": "Discriminant b²−4ac", "formula": "b * b - 4 * a * c", "unit": ""},
        ],
        curve_type="quadratic",
        coefficients=[1, 0, -1],
        class_level="Class 10",
    )


def _algebra_stepper_spec() -> dict[str, Any]:
    return _viz(
        "algebra-stepper",
        "Algebra Stepper — solve step by step",
        "Press Next to apply one inverse operation at a time.",
        algebra_steps=[
            {
                "id": "s0",
                "expressionBefore": "2x + 5 = 15",
                "expressionAfter": "2x + 5 = 15",
                "operation": "Start with the equation.",
                "highlightTerms": ["2x", "5", "15"],
            },
            {
                "id": "s1",
                "expressionBefore": "2x + 5 = 15",
                "expressionAfter": "2x = 10",
                "operation": "Subtract 5 from both sides",
                "highlightTerms": ["5"],
            },
            {
                "id": "s2",
                "expressionBefore": "2x = 10",
                "expressionAfter": "x = 5",
                "operation": "Divide both sides by 2",
                "highlightTerms": ["2"],
            },
        ],
        buttons=[
            {"id": "next", "label": "Next step", "action": "animate"},
            {"id": "reset", "label": "Reset", "action": "reset"},
        ],
        interactions=[
            {
                "id": "step",
                "type": "click",
                "description": "Walk through each algebraic move.",
                "expectedObservation": "Both sides stay equal after every step.",
            }
        ],
        class_level="Class 8",
    )


def _compound_interest_spec() -> dict[str, Any]:
    return _viz(
        "compound-interest-visual",
        "Compound Interest Growth",
        "Change principal, rate and years. Watch the amount stack year by year.",
        sliders=[
            {"id": "P", "label": "Principal (₹)", "min": 1000, "max": 50000, "step": 1000, "default": 10000},
            {"id": "r", "label": "Rate % p.a.", "min": 1, "max": 20, "step": 0.5, "default": 8},
            {"id": "n", "label": "Years", "min": 1, "max": 10, "step": 1, "default": 5},
        ],
        calcs=[
            {"id": "amount", "label": "Amount A", "formula": "P * ((1 + r / 100) ** n)", "unit": "₹"},
            {"id": "ci", "label": "CI = A − P", "formula": "P * ((1 + r / 100) ** n) - P", "unit": "₹"},
        ],
        buttons=[{"id": "animate", "label": "▶ Grow year by year", "action": "animate"}],
        finance_spec={
            "principal": 10000,
            "rate": 8,
            "timeYears": 5,
            "mode": "compound-interest",
            "compoundingFrequency": "annually",
        },
        class_level="Class 8",
    )


def _heights_distances_spec() -> dict[str, Any]:
    return _viz(
        "heights-distances-scene",
        "Heights & Distances (3D)",
        "Set the angle of elevation and distance. See the tower height update in 3D.",
        render_mode="3d",
        scene=_heights_scene(),
        sliders=[
            {"id": "angle", "label": "Angle of elevation", "min": 15, "max": 75, "step": 1, "default": 30},
            {"id": "distance", "label": "Distance (m)", "min": 10, "max": 100, "step": 5, "default": 40},
        ],
        calcs=[
            {
                "id": "height",
                "label": "Height",
                "formula": "distance * tan(angle * 3.14159 / 180)",
                "unit": "m",
            }
        ],
        buttons=[{"id": "animate", "label": "▶ Sight line", "action": "animate"}],
        class_level="Class 10",
    )


def _line_intersection_spec() -> dict[str, Any]:
    return _viz(
        "line-intersection",
        "Two Lines Intersection",
        "Adjust both lines and find their point of intersection.",
        sliders=[
            {"id": "m1", "label": "Line 1 slope", "min": -3, "max": 3, "step": 0.5, "default": 1},
            {"id": "c1", "label": "Line 1 intercept", "min": -8, "max": 8, "step": 1, "default": 2},
            {"id": "m2", "label": "Line 2 slope", "min": -3, "max": 3, "step": 0.5, "default": -1},
            {"id": "c2", "label": "Line 2 intercept", "min": -8, "max": 8, "step": 1, "default": 5},
        ],
    )


def _triangle_explorer_spec() -> dict[str, Any]:
    return _viz(
        "triangle-explorer",
        "Triangle Builder",
        "Drag the vertices and watch side lengths and angles update live.",
        draggables=[
            {"id": "A", "label": "A", "initialX": 60, "initialY": 40, "color": "#3B82F6"},
            {"id": "B", "label": "B", "initialX": 20, "initialY": 120, "color": "#10B981"},
            {"id": "C", "label": "C", "initialX": 140, "initialY": 120, "color": "#F59E0B"},
        ],
    )


def _triangle_angle_sum_spec() -> dict[str, Any]:
    return _viz(
        "triangle-angle-sum",
        "Triangle Angle Sum = 180°",
        "Press Animate to rearrange the three angles into a straight line.",
        buttons=[
            {"id": "animate", "label": "▶ Rearrange angles", "action": "animate"},
            {"id": "reset", "label": "Reset", "action": "reset"},
        ],
        calcs=[{"id": "sum", "label": "Angle sum", "formula": "60 + 60 + 60", "unit": "°"}],
    )


def _quadrilateral_morph_spec() -> dict[str, Any]:
    return _viz(
        "quadrilateral-morph",
        "Quadrilateral Shape Morph",
        "Drag corners to make a square, rectangle, rhombus or parallelogram.",
        draggables=[
            {"id": "P1", "label": "Corner 1", "initialX": 50, "initialY": 30, "color": "#3B82F6"},
            {"id": "P2", "label": "Corner 2", "initialX": 150, "initialY": 30, "color": "#3B82F6"},
            {"id": "P3", "label": "Corner 3", "initialX": 160, "initialY": 110, "color": "#10B981"},
            {"id": "P4", "label": "Corner 4", "initialX": 40, "initialY": 110, "color": "#10B981"},
        ],
    )


def _statistics_lab_spec() -> dict[str, Any]:
    return _viz(
        "statistics-lab",
        "Statistics Lab",
        "Edit the dataset below. Watch mean, median, mode and the bar graph update instantly.",
        buttons=[
            {"id": "add", "label": "+ Add mark", "action": "add"},
            {"id": "remove", "label": "− Remove", "action": "remove"},
        ],
    )


def _mensuration_cube_spec() -> dict[str, Any]:
    return _viz(
        "mensuration-cube",
        "3D Cube Explorer",
        "Change the side length and see surface area and volume update.",
        render_mode="3d",
        scene=_cube_scene(),
        sliders=[{"id": "s", "label": "Side length (cm)", "min": 1, "max": 10, "step": 1, "default": 5}],
        calcs=[
            {"id": "sa", "label": "Surface area", "formula": "6 * s * s", "unit": " cm^2"},
            {"id": "vol", "label": "Volume", "formula": "s * s * s", "unit": " cm^3"},
        ],
        buttons=[{"id": "rotate", "label": "▶ Rotate", "action": "animate"}],
        class_level="Class 9",
    )


def _mensuration_cylinder_spec() -> dict[str, Any]:
    return _viz(
        "mensuration-cylinder",
        "3D Cylinder Explorer",
        "Adjust radius and height to see volume and curved surface area.",
        render_mode="3d",
        scene=_cylinder_scene(),
        sliders=[
            {"id": "r", "label": "Radius", "min": 1, "max": 8, "step": 1, "default": 3},
            {"id": "h", "label": "Height", "min": 1, "max": 12, "step": 1, "default": 5},
        ],
        calcs=[
            {"id": "vol", "label": "Volume", "formula": "3.14159 * r * r * h", "unit": " cu units"},
            {"id": "csa", "label": "Curved SA", "formula": "2 * 3.14159 * r * h", "unit": " sq units"},
        ],
        class_level="Class 9",
    )


def _area_resizer_spec() -> dict[str, Any]:
    return _viz(
        "area-resizer",
        "Area Resizer",
        "Resize the rectangle and triangle. Watch areas update in real time.",
        sliders=[
            {"id": "width", "label": "Rectangle width", "min": 1, "max": 12, "step": 1, "default": 6},
            {"id": "height", "label": "Rectangle height", "min": 1, "max": 12, "step": 1, "default": 4},
            {"id": "base", "label": "Triangle base", "min": 1, "max": 12, "step": 1, "default": 8},
            {"id": "triHeight", "label": "Triangle height", "min": 1, "max": 12, "step": 1, "default": 5},
        ],
        calcs=[
            {"id": "rectArea", "label": "Rectangle area", "formula": "width * height", "unit": ""},
            {"id": "triArea", "label": "Triangle area", "formula": "0.5 * base * triHeight", "unit": ""},
        ],
    )


def _open_box_spec() -> dict[str, Any]:
    """Net → open box: sheet cut at corners, fold up. Sliders match worked solutions."""
    return _viz(
        "area-resizer",
        "Open Box from a Sheet",
        "Adjust sheet, cut, and box dimensions; volume updates live (fold animation deferred).",
        sliders=[
            {"id": "sheetLength", "label": "Sheet length (cm)", "min": 10, "max": 80, "step": 1, "default": 30},
            {"id": "sheetWidth", "label": "Sheet width (cm)", "min": 10, "max": 80, "step": 1, "default": 20},
            {"id": "cut", "label": "Cut square side (cm)", "min": 1, "max": 20, "step": 1, "default": 5},
            {"id": "length", "label": "Box length (cm)", "min": 1, "max": 60, "step": 1, "default": 20},
            {"id": "width", "label": "Box width (cm)", "min": 1, "max": 60, "step": 1, "default": 10},
            {"id": "height", "label": "Box height (cm)", "min": 1, "max": 20, "step": 1, "default": 5},
        ],
        calcs=[
            {"id": "volume", "label": "Volume", "formula": "length * width * height", "unit": " cm^3"},
            {
                "id": "baseLength",
                "label": "Base length check",
                "formula": "sheetLength - 2 * cut",
                "unit": " cm",
            },
            {
                "id": "baseWidth",
                "label": "Base width check",
                "formula": "sheetWidth - 2 * cut",
                "unit": " cm",
            },
        ],
        buttons=[{"id": "animate", "label": "▶ Fold box", "action": "animate"}],
    )


def _probability_dice_spec() -> dict[str, Any]:
    return _viz(
        "probability-dice",
        "Dice Simulator (1000 rolls)",
        "Roll the die many times and watch experimental probability approach 1/6.",
        buttons=[
            {"id": "roll", "label": "Roll once", "action": "roll"},
            {"id": "roll1000", "label": "Roll 1000×", "action": "animate"},
            {"id": "reset", "label": "Reset", "action": "reset"},
        ],
    )


def _probability_coin_spec() -> dict[str, Any]:
    return _viz(
        "probability-coin",
        "Coin Toss Simulator",
        "Toss repeatedly and observe how experimental probability approaches 1/2.",
        buttons=[
            {"id": "flip", "label": "Flip", "action": "flip"},
            {"id": "flip100", "label": "Flip 100×", "action": "animate"},
            {"id": "reset", "label": "Reset", "action": "reset"},
        ],
    )


def _circle_tangent_spec() -> dict[str, Any]:
    return _viz(
        "circle-tangent",
        "Circle & Tangent Explorer",
        "Drag the tangent around the circle. It always touches at exactly one point.",
        sliders=[
            {"id": "tangentAngle", "label": "Tangent position", "min": 0, "max": 360, "step": 1, "default": 45},
            {"id": "r", "label": "Radius", "min": 2, "max": 8, "step": 1, "default": 5},
        ],
        calcs=[
            {"id": "circ", "label": "Circumference", "formula": "2 * 3.14159 * r", "unit": ""},
            {"id": "area", "label": "Area", "formula": "3.14159 * r * r", "unit": ""},
        ],
    )


def _circle_explorer_spec() -> dict[str, Any]:
    return _viz(
        "circle",
        "Circle Explorer",
        "Drag the radius slider. Diameter, circumference and area update instantly.",
        sliders=[{"id": "r", "label": "Radius", "min": 1, "max": 10, "step": 1, "default": 5}],
        calcs=[
            {"id": "d", "label": "Diameter", "formula": "2 * r", "unit": ""},
            {"id": "circ", "label": "Circumference", "formula": "2 * 3.14159 * r", "unit": ""},
            {"id": "area", "label": "Area", "formula": "3.14159 * r * r", "unit": ""},
        ],
    )


# (pattern, spec_builder, concept, objective, explanation)
_TOPIC_RULES: list[tuple[re.Pattern[str], Callable[[], dict], str, str, str]] = [
    (
        re.compile(
            r"\b(compound\s+interest|CI\b|amount\s*=\s*P|A\s*=\s*P\s*\(|\bGST\b|\btax\b|discount)\b",
            re.I,
        ),
        _compound_interest_spec,
        "Compound Interest",
        "See how money grows when interest is added to the principal each year.",
        "A = P(1 + r/100)^n",
    ),
    (
        re.compile(
            r"\b(height[s]?\s+and\s+distances?|angle\s+of\s+elevation|angle\s+of\s+depression)\b",
            re.I,
        ),
        _heights_distances_spec,
        "Heights and Distances",
        "Link trigonometry to a real tower / observer scene.",
        "height = distance × tan(θ)",
    ),
    (
        re.compile(
            r"\b(quadratic\s+equation|parabola|discriminant|ax\s*\^?\s*2|ax²|"
            r"roots?\s+of\s+(?:the\s+)?quadratic)\b",
            re.I,
        ),
        _quadratic_grapher_spec,
        "Quadratic Equations",
        "Graph y = ax² + bx + c and relate roots to the discriminant.",
        "Discriminant Δ = b² − 4ac",
    ),
    (
        re.compile(
            r"\b(simplif(?:y|ying)|transpose|expand\s+and\s+simplif)\b",
            re.I,
        ),
        _algebra_stepper_spec,
        "Algebra Stepper",
        "Apply one inverse operation per step until the variable is isolated.",
        "Whatever you do to one side, do to the other.",
    ),
    (
        re.compile(
            r"\bcircles?\b(?!\s*(?:of\s+)?(?:a\s+)?cylinder)|"
            r"circumference|diameter.*circle|circle.*diameter|"
            r"area\s+of\s+(?:a\s+)?circle|circle.*\barea\b",
            re.I,
        ),
        _circle_explorer_spec,
        "Circle Measurements",
        "Explore radius, diameter, circumference and area.",
        "C = 2πr, A = πr².",
    ),
    (
        re.compile(
            r"\b(die|dice)\b|roll\s+(?:a\s+)?die|thrown?\s+once",
            re.I,
        ),
        _probability_dice_spec,
        "Dice Probability",
        "Roll many times; experimental → 1/6.",
        "Law of large numbers for fair dice.",
    ),
    (
        re.compile(
            r"\b(solve|find)\b.*\b\d*\s*[a-z]\s*[\+\-]\s*\d+|\b\d*\s*[a-z]\s*[\+\-]\s*\d+\s*=|"
            r"linear\s+equation|equation\s+in\s+one\s+variable|"
            r"number\s+when\s+added|gives\s+\d+|cost\s+of\s+one",
            re.I,
        ),
        _algebra_stepper_spec,
        "Linear Equations",
        "Isolate the variable using inverse operations, one step at a time.",
        "Whatever you do to one side, do to the other.",
    ),
    (
        re.compile(
            r"\barea\b.*\b(triangle|parallelogram|rectangle)\b|"
            r"\b(triangle|parallelogram)\b.*\barea\b|"
            r"base\s+\d+.*height\s+\d+",
            re.I,
        ),
        _area_resizer_spec,
        "Areas of Plane Figures",
        "Resize shapes and see areas change live.",
        "Area of triangle = ½ × base × height.",
    ),
    (
        re.compile(
            r"\b(triangle|triangles)\b.*\b(angles?|find|sum|180)\b|"
            r"\b(angles?|sum)\b.*\b(triangle|triangles)\b|"
            r"\b(three|3)\s+angles?\b|"
            r"\binterior\s+angles?\s+(of\s+)?(a\s+)?(triangle|triangles)\b|"
            r"\bfind\s+angles?\s+[a-z]\b|"
            r"angles?\s+[a-z]\s*=\s*\d+.*angles?\s+[a-z]\s*=\s*\d+|"
            r"prove\b.*\b(triangle|180)\b",
            re.I,
        ),
        _triangle_angle_sum_spec,
        "Triangle Angle Sum",
        "Prove interior angles of a triangle sum to 180°.",
        "Three angles rearrange to form a straight line.",
    ),
    (
        re.compile(
            r"\b(triangle|triangles)\b.*\b(formula|formulas|property|properties|theorem)\b|"
            r"\b(formula|formulas)\b.*\b(triangle|triangles)\b",
            re.I,
        ),
        _triangle_angle_sum_spec,
        "Triangle Formulas",
        "Learn triangle formulas starting with the angle sum.",
        "The three interior angles always add up to 180°.",
    ),
    (
        re.compile(
            r"\b(?:tell\s+me\s+about|explain|describe|what\s+(?:is|are)|about)\b.*\b(?:the\s+)?(?:triangle|triangles)\b",
            re.I,
        ),
        _triangle_explorer_spec,
        "Triangles",
        "Explore triangles by dragging vertices.",
        "Watch how sides and angles change as you move the corners.",
    ),
    (
        re.compile(r"\b(triangle|triangles)\b", re.I),
        _triangle_angle_sum_spec,
        "Triangles",
        "Explore triangle angle relationships interactively.",
        "Interior angles of any triangle sum to 180°.",
    ),
    (
        re.compile(r"\b(polynomial|monomial|binomial|trinomial|remainder\s+theorem)\b", re.I),
        _factor_rectangle_spec,
        "Polynomials & Factorisation",
        "Explore factors using the rectangle area model.",
        "Polynomials are sums of terms; factorisation splits expressions.",
    ),
    (
        re.compile(
            r"(?:√\s*\d|sqrt\s*\(\s*\d+|square\s*root\s+of\s+\d+).{0,60}number\s*line|"
            r"number\s*line.{0,60}(?:√|sqrt|square\s*root|geometrical?\s+construction)|"
            r"geometrical?\s+construction.{0,40}(?:√|sqrt|square\s*root)|"
            r"(?:mark|represent|locate)\s+(?:√|sqrt|square\s*root)",
            re.I,
        ),
        lambda: _viz(
            "sqrt-number-line",
            "Construct √n on the number line",
            "Build a right triangle, then swing an arc to mark √n.",
            sliders=[
                {"id": "n", "label": "n (√n)", "min": 2, "max": 50, "step": 1, "default": 5},
                {"id": "a", "label": "Leg a", "min": 1, "max": 20, "step": 1, "default": 2},
                {"id": "b", "label": "Leg b", "min": 1, "max": 20, "step": 1, "default": 1},
            ],
            calcs=[{"id": "check", "label": "a² + b²", "formula": "a * a + b * b", "unit": ""}],
            buttons=[{"id": "reset", "label": "Reset", "action": "reset"}],
        ),
        "Square Root on the Number Line",
        "Construct √n using a right triangle and compass arc.",
        "If a² + b² = n, the hypotenuse is √n; transfer it to the line with an arc.",
    ),
    (
        re.compile(r"\b(irrational|surd|rationali[sz]|represent.*sqrt|√\s*\d|number\s*line)\b", re.I),
        lambda: _viz(
            "number-line",
            "Number Line Explorer",
            "Place rational and irrational numbers on the line.",
            sliders=[
                {"id": "start", "label": "Start", "min": -10, "max": 10, "step": 1, "default": 0},
                {"id": "jump", "label": "Jump", "min": -10, "max": 10, "step": 1, "default": 3},
            ],
            calcs=[{"id": "result", "label": "Position", "formula": "start + jump", "unit": ""}],
        ),
        "Real Numbers on the Number Line",
        "Jump along the line to explore positions. For geometric √n construction, ask to mark √n on the number line.",
        "Every real number has a unique point on the number line.",
    ),
    (
        re.compile(r"\(a\s*\+\s*b\)\s*[\²^2]|a\s*[\²^2]\s*\+\s*b\s*[\²^2]|algebra\s*tile|missing\s*term|2ab", re.I),
        _algebra_tiles_spec,
        "Why (a+b)² ≠ a²+b²",
        "Discover the 2ab term using algebra tiles.",
        "(a+b)² = a² + 2ab + b², not a² + b².",
    ),
    (
        re.compile(
            r"(?<!prime\s)factori[sz]ation|(?<!prime\s)factori[sz]e|"
            r"x\s*[\²^2].*\+.*x|rectangle\s*model|x\^2",
            re.I,
        ),
        _factor_rectangle_spec,
        "Factorising Quadratics",
        "Find factors using the rectangle area model.",
        "Split x²+bx+c into two binomial factors.",
    ),
    (
        re.compile(
            r"line\s*segments?|\brays?\b|"
            r"(?:rays?|segments?|lines?)\s*,?\s*(?:or|and)\s+(?:rays?|segments?|lines?)|"
            r"draw.*(?:line|ray|segment)|"
            r"identif(?:y|ies|ying).*(?:ray|segment|line)|"
            r"classif(?:y|ies|ying).*(?:ray|segment|line)|"
            r"elements\s*of\s*geometry|geometry\s*explorer|point.*line",
            re.I,
        ),
        _geometry_basics_spec,
        "Elements of Geometry",
        "Distinguish point, line, segment and ray.",
        "A segment has two endpoints; a ray has one.",
    ),
    (
        re.compile(r"transversal|parallel\s*line|corresponding|alternate\s*interior", re.I),
        _parallel_transversal_spec,
        "Parallel Lines & Transversal",
        "Explore angle pairs when a transversal cuts parallel lines.",
        "Corresponding and alternate angles are equal.",
    ),
    (
        re.compile(r"acute|obtuse|straight\s*angle|angle\s*explorer|drag.*ray", re.I),
        _angle_explorer_spec,
        "Types of Angles",
        "Classify angles by dragging two rays.",
        "Angles can be acute, right, obtuse or straight.",
    ),
    (
        re.compile(r"intersect|two\s*linear|two\s*equations|system\s*of", re.I),
        _line_intersection_spec,
        "Intersection of Lines",
        "Find where two linear equations meet.",
        "The intersection solves both equations.",
    ),
    (
        re.compile(r"y\s*=\s*mx|slope|intercept|linear\s*equation.*variable|two\s*variable", re.I),
        _linear_graph_spec,
        "Linear Equations in Two Variables",
        "See how m and c change the graph of y = mx + c.",
        "Slope m controls steepness; c is the y-intercept.",
    ),
    (
        re.compile(
            r"sum.*interior.*triangle|180.*triangle|angles?\s*rearrang|angles?\s*sums?|"
            r"\b(three|3)\s+angles?\b",
            re.I,
        ),
        _triangle_angle_sum_spec,
        "Triangle Angle Sum",
        "Prove interior angles of a triangle sum to 180°.",
        "Three angles rearrange to form a straight line.",
    ),
    (
        re.compile(r"triangle.*vert|drag.*vert|triangle\s*builder", re.I),
        _triangle_explorer_spec,
        "Triangle Explorer",
        "Drag vertices and observe changing measurements.",
        "Side lengths and angles change as vertices move.",
    ),
    (
        re.compile(r"quadrilateral|parallelogram|rhombus|rectangle.*square|diagonal", re.I),
        _quadrilateral_morph_spec,
        "Quadrilaterals",
        "Morph a quadrilateral by dragging corners.",
        "Different corner positions create different quadrilateral types.",
    ),
    (
        re.compile(r"mean|median|mode|statistics|bar\s*graph|dataset|marks\s*of\s*student", re.I),
        _statistics_lab_spec,
        "Statistics Lab",
        "Edit data and watch mean, median, mode update.",
        "Statistics describe a dataset's centre and spread.",
    ),
    (
        re.compile(r"cylinder|curved\s*surface", re.I),
        _mensuration_cylinder_spec,
        "Cylinder — Surface Area & Volume",
        "Explore how radius and height affect a cylinder.",
        "V = πr²h.",
    ),
    (
        re.compile(
            r"open\s+box|cut\s+from\s+(?:its\s+)?(?:four\s+)?corners?|"
            r"folded?\s+upwards?|squares?\s+of\s+side\s+\d+.*cut|"
            r"sheet\s+of\s+paper.*(?:cut|fold)",
            re.I,
        ),
        _open_box_spec,
        "Open Box from a Rectangular Sheet",
        "Resize length, width, and height to match the folded box; volume updates live.",
        "Box length = sheet length − 2×cut; volume = l × w × h.",
    ),
    (
        re.compile(r"(?<!perfect\s)cube|3d.*side|surface\s*area.*volume", re.I),
        _mensuration_cube_spec,
        "Cube — Surface Area & Volume",
        "Rotate a cube and change its side length.",
        "SA = 6s², V = s³.",
    ),
    (
        re.compile(r"parallelogram.*area|cut.*rearrang|resize.*triangle|area\s*formula", re.I),
        _area_resizer_spec,
        "Areas of Plane Figures",
        "Resize shapes and see areas change live.",
        "Area of triangle = ½ × base × height.",
    ),
    (
        re.compile(r"toss.*coin|coin.*toss|flip.*coin", re.I),
        _probability_coin_spec,
        "Coin Toss Probability",
        "Observe experimental probability → ½.",
        "More tosses bring experimental probability closer to ½.",
    ),
    (
        re.compile(r"tangent|point\s*of\s*contact", re.I),
        _circle_tangent_spec,
        "Tangent to a Circle",
        "A tangent touches the circle at exactly one point.",
        "The radius is perpendicular to the tangent at contact.",
    ),
    (
        re.compile(
            r"\b(sector|circular\s+segment|areas?\s+related\s+to\s+circles?|"
            r"area\s+of\s+(?:a\s+)?sector|segment\s+of\s+(?:a\s+)?circle)\b",
            re.I,
        ),
        lambda: _viz(
            "circle",
            "Sector of a Circle",
            "Change radius and central angle; watch the sector sweep and the area update.",
            sliders=[
                {"id": "r", "label": "Radius r", "min": 1, "max": 12, "step": 1, "default": 5},
                {"id": "theta", "label": "Central angle θ (°)", "min": 0, "max": 360, "step": 5, "default": 90},
            ],
            calcs=[
                {
                    "id": "sectorArea",
                    "label": "Sector area",
                    "formula": "(theta / 360) * 3.14159 * r * r",
                    "unit": "",
                }
            ],
            buttons=[{"id": "reset", "label": "Reset", "action": "reset"}],
        ),
        "Areas Related to Circles",
        "See how sector area grows with central angle.",
        "Sector area = (θ/360) × πr²",
    ),
    (
        re.compile(r"\b(chord|angles?\s+subtended|circle\s+theorems?)\b", re.I),
        lambda: _viz(
            "circle",
            "Circle Explorer",
            "Explore radius and circle relationships for chords and subtended angles.",
            sliders=[
                {"id": "r", "label": "Radius r", "min": 1, "max": 12, "step": 1, "default": 5},
            ],
            calcs=[
                {"id": "d", "label": "Diameter", "formula": "2 * r", "unit": ""},
                {"id": "circ", "label": "Circumference", "formula": "2 * 3.14159 * r", "unit": ""},
            ],
        ),
        "Circles — Chords and Angles",
        "Relate radius and circle measures while studying chords.",
        "A chord is a line segment joining two points on a circle.",
    ),
    (
        re.compile(r"compass|ruler|perpendicular\s*bisector|construction|construct\s*a\s*triangle", re.I),
        lambda: _viz(
            "geometry-construction",
            "Compass & Ruler Construction",
            "Guided perpendicular bisector of segment AB (step scrubber).",
            draggables=[
                {"id": "A", "label": "A", "initialX": 60, "initialY": 100, "color": "#3B82F6"},
                {"id": "B", "label": "B", "initialX": 180, "initialY": 100, "color": "#10B981"},
            ],
            buttons=[
                {"id": "step1", "label": "Step 1: Arc from A", "action": "animate"},
                {"id": "step2", "label": "Step 2: Arc from B", "action": "animate"},
                {"id": "reset", "label": "Reset", "action": "reset"},
            ],
        ),
        "Geometrical Constructions",
        "Learn the perpendicular bisector construction step by step.",
        "Virtual compass arcs meet on the perpendicular bisector of AB.",
    ),
    (
        re.compile(r"quarter\s*turn|half\s*turn|full\s*turn", re.I),
        lambda: _viz(
            "circle",
            "Turns and Angles",
            "Explore quarter, half and full turns.",
            sliders=[{"id": "turnSlider", "label": "Quarter turns", "min": 0, "max": 4, "step": 1, "default": 1}],
            calcs=[{"id": "angle", "label": "Angle", "formula": "turnSlider * 90", "unit": "°"}],
            buttons=[
                {"id": "animate", "label": "▶ Animate Turns", "action": "animate"},
                {"id": "reset", "label": "Reset", "action": "reset"},
            ],
        ),
        "Turns and Angles",
        "Relate turns to degree measures.",
        "A quarter turn = 90°.",
    ),
]


def match_textbook_visualization(query: str, class_level: str = "") -> dict[str, Any] | None:
    """Return a full math-lesson dict for the best-matching topic (Class 1–10)."""
    from app.services.math_lesson.visual_catalog import match_math_visualization

    q = (query or "").strip()
    if not q:
        return None
    return match_math_visualization(q, class_level)
