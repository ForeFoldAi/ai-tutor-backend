"""
Class 9 EM Mathematics textbook — topic → interactive visualization catalog.

Aligned with chapters: Polynomials, Geometry Elements, Lines & Angles,
Linear Equations, Triangles, Quadrilaterals, Statistics, Surface Areas,
Areas, Constructions, Probability, Circles.
"""

from __future__ import annotations

import re
from typing import Any, Callable

_DEFAULT_COLORS = {
    "primary": "#3B82F6",
    "secondary": "#10B981",
    "accent": "#F59E0B",
    "background": "#F8FAFC",
    "text": "#1E293B",
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
    interactions: list[dict] | None = None,
    animations: list[dict] | None = None,
) -> dict[str, Any]:
    return {
        "visualizationType": vtype,
        "title": title,
        "description": description,
        "sliders": sliders or [],
        "buttons": buttons or [],
        "liveCalculations": calcs or [],
        "draggableObjects": draggables or [],
        "studentInteractions": interactions or [],
        "animations": animations or [],
        "colors": dict(_DEFAULT_COLORS),
    }


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
        sliders=[{"id": "s", "label": "Side length", "min": 1, "max": 10, "step": 1, "default": 3}],
        calcs=[
            {"id": "sa", "label": "Surface area", "formula": "6 * s * s", "unit": " sq units"},
            {"id": "vol", "label": "Volume", "formula": "s * s * s", "unit": " cu units"},
        ],
        buttons=[{"id": "rotate", "label": "▶ Rotate", "action": "animate"}],
    )


def _mensuration_cylinder_spec() -> dict[str, Any]:
    return _viz(
        "mensuration-cylinder",
        "3D Cylinder Explorer",
        "Adjust radius and height to see volume and curved surface area.",
        sliders=[
            {"id": "r", "label": "Radius", "min": 1, "max": 8, "step": 1, "default": 3},
            {"id": "h", "label": "Height", "min": 1, "max": 12, "step": 1, "default": 5},
        ],
        calcs=[
            {"id": "vol", "label": "Volume", "formula": "3.14159 * r * r * h", "unit": " cu units"},
            {"id": "csa", "label": "Curved SA", "formula": "2 * 3.14159 * r * h", "unit": " sq units"},
        ],
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
        re.compile(r"\(a\s*\+\s*b\)\s*[\²^2]|a\s*[\²^2]\s*\+\s*b\s*[\²^2]|algebra\s*tile|missing\s*term|2ab", re.I),
        _algebra_tiles_spec,
        "Why (a+b)² ≠ a²+b²",
        "Discover the 2ab term using algebra tiles.",
        "(a+b)² = a² + 2ab + b², not a² + b².",
    ),
    (
        re.compile(r"factori[sz]e|factori[sz]ation|x\s*[\²^2].*\+.*x|rectangle\s*model|x\^2", re.I),
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
        re.compile(r"sum.*interior.*triangle|180.*triangle|angle\s*rearrang|angle\s*sum", re.I),
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
        re.compile(r"cube|3d.*side|surface\s*area.*volume", re.I),
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
        re.compile(r"roll.*die|dice|experimental\s*probability|1000\s*times", re.I),
        _probability_dice_spec,
        "Dice Probability",
        "Roll many times; experimental → 1/6.",
        "Law of large numbers for fair dice.",
    ),
    (
        re.compile(r"tangent|point\s*of\s*contact", re.I),
        _circle_tangent_spec,
        "Tangent to a Circle",
        "A tangent touches the circle at exactly one point.",
        "The radius is perpendicular to the tangent at contact.",
    ),
    (
        re.compile(r"circumference|circle.*radius|diameter.*circle", re.I),
        _circle_explorer_spec,
        "Circle Measurements",
        "Explore radius, diameter, circumference and area.",
        "C = 2πr, A = πr².",
    ),
    (
        re.compile(r"compass|ruler|perpendicular\s*bisector|construction|construct\s*a\s*triangle", re.I),
        lambda: _viz(
            "geometry-construction",
            "Compass & Ruler Construction",
            "Place points A and B, then run the guided perpendicular bisector steps.",
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
        "Learn compass-and-ruler constructions step by step.",
        "Construct perpendicular bisectors and triangles with virtual tools.",
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
