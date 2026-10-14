"""
Class 1–10 mathematics — elementary & middle-school visualization rules.

Complements textbook_catalog.py (Class 6–10 advanced topics).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.services.math_lesson.textbook_catalog import _quadrilateral_morph_spec, _viz

# Locked 2D/3D shape lab — one named shape, never a morphing n-gon.
_SHAPE_LAB_GUIDES: dict[str, list[str]] = {
    "square": [
        "Resize the side. Every side stays equal — it never becomes a rectangle or pentagon.",
        "Watch perimeter (4 × side) and area (side × side) update together.",
        "Switch to 3D: a cube is six of these squares as faces.",
    ],
    "rectangle": [
        "Change length and width separately. Opposite sides stay equal.",
        "When length equals width, the rectangle becomes a square.",
        "Switch to 3D: a cuboid is the solid with this rectangle as its face.",
    ],
    "triangle": [
        "A triangle has 3 sides and 3 corners. Resize a side and keep all three.",
        "Switch to 3D to see a triangular prism built from this triangle.",
    ],
    "pentagon": [
        "A regular pentagon has 5 equal sides. Only this pentagon is shown.",
        "Switch to 3D to see a pentagonal prism.",
    ],
    "hexagon": [
        "A regular hexagon has 6 equal sides. Only this hexagon is shown.",
        "Switch to 3D to see a hexagonal prism.",
    ],
    "cuboid": [
        "A cuboid is the 3D box whose faces are rectangles.",
        "Switch to 2D to see one rectangular face.",
    ],
    "picker": [
        "Tap one shape name. Only that shape is drawn.",
        "Use 2D for the face and 3D for the solid.",
    ],
}


def _shape_lab_spec(shape: str, *, default_view: str = "2d") -> dict[str, Any]:
    """Professional locked shape explorer. `shape` is square|rectangle|triangle|pentagon|hexagon|cuboid|picker."""
    shape = shape.lower()
    titles = {
        "square": "Square Lab — 2D face & 3D cube",
        "rectangle": "Rectangle Lab — 2D face & 3D cuboid",
        "triangle": "Triangle Lab — 2D face & 3D prism",
        "pentagon": "Pentagon Lab — 2D face & 3D prism",
        "hexagon": "Hexagon Lab — 2D face & 3D prism",
        "cuboid": "Cuboid Lab — 3D box & 2D rectangular face",
        "picker": "2D & 3D Shape Lab",
    }
    descriptions = {
        "square": "Only a square. Resize the side, then switch to 3D to see the cube made of 6 squares.",
        "rectangle": "Only a rectangle. Change length and width, then switch to 3D for the cuboid.",
        "triangle": "Only a triangle. Resize a side, then switch to 3D for the prism.",
        "pentagon": "Only a regular pentagon. Resize a side, then switch to 3D for the prism.",
        "hexagon": "Only a regular hexagon. Resize a side, then switch to 3D for the prism.",
        "cuboid": "A cuboid (rectangular box). Adjust length, width and height. Switch to 2D for one face.",
        "picker": "Pick one named shape. Explore that shape in 2D, then its solid in 3D.",
    }
    if shape in ("rectangle", "cuboid"):
        sliders = [
            {"id": "length", "label": "Length", "min": 1, "max": 10, "step": 1, "default": 6},
            {"id": "width", "label": "Width", "min": 1, "max": 10, "step": 1, "default": 4},
            {"id": "height", "label": "Height (3D)", "min": 1, "max": 10, "step": 1, "default": 3},
        ]
        calcs = [
            {"id": "perim", "label": "Perimeter (2D)", "formula": "2 * (length + width)", "unit": ""},
            {"id": "area", "label": "Face area", "formula": "length * width", "unit": ""},
            {"id": "vol", "label": "Volume (3D)", "formula": "length * width * height", "unit": ""},
        ]
    else:
        sliders = [
            {"id": "s", "label": "Side length", "min": 1, "max": 10, "step": 1, "default": 4},
            {"id": "height", "label": "Height (3D)", "min": 1, "max": 10, "step": 1, "default": 4},
        ]
        if shape == "square":
            calcs = [
                {"id": "perim", "label": "Perimeter", "formula": "4 * s", "unit": ""},
                {"id": "area", "label": "Area", "formula": "s * s", "unit": ""},
                {"id": "sa", "label": "Cube surface area", "formula": "6 * s * s", "unit": ""},
                {"id": "vol", "label": "Cube volume", "formula": "s * s * s", "unit": ""},
            ]
        elif shape == "triangle":
            calcs = [
                {"id": "perim", "label": "Perimeter", "formula": "3 * s", "unit": ""},
                {"id": "area", "label": "Area (equilateral)", "formula": "0.433 * s * s", "unit": ""},
            ]
        elif shape == "pentagon":
            calcs = [
                {"id": "perim", "label": "Perimeter", "formula": "5 * s", "unit": ""},
                {"id": "area", "label": "Area", "formula": "1.72 * s * s", "unit": ""},
            ]
        elif shape == "hexagon":
            calcs = [
                {"id": "perim", "label": "Perimeter", "formula": "6 * s", "unit": ""},
                {"id": "area", "label": "Area", "formula": "2.598 * s * s", "unit": ""},
            ]
        else:
            calcs = [
                {"id": "perim", "label": "Perimeter", "formula": "4 * s", "unit": ""},
                {"id": "area", "label": "Area", "formula": "s * s", "unit": ""},
            ]
    draw_label = {
        "square": "▶ Draw square",
        "rectangle": "▶ Draw rectangle",
        "triangle": "▶ Draw triangle",
        "pentagon": "▶ Draw pentagon",
        "hexagon": "▶ Draw hexagon",
        "cuboid": "▶ Build cuboid",
        "picker": "▶ Draw shape",
    }.get(shape, "▶ Draw")
    return _viz(
        "shape-lab",
        titles.get(shape, "Shape Lab"),
        descriptions.get(shape, "Explore this shape in 2D and 3D."),
        objects=[{
            "id": "shape",
            "label": shape.title(),
            "type": shape,
            "properties": {"lock": shape != "picker", "defaultView": default_view},
        }],
        sliders=sliders,
        calcs=calcs,
        buttons=[
            {"id": "animate", "label": draw_label, "action": "animate"},
            {"id": "reset", "label": "Reset", "action": "reset"},
        ],
        interactions=[
            {
                "id": "i1",
                "type": "slide",
                "description": "Use the sliders to resize. The shape name never changes.",
                "expectedObservation": "Only this shape is on screen.",
            },
            {
                "id": "i2",
                "type": "toggle",
                "description": "Switch 2D (face) and 3D (solid).",
                "expectedObservation": "The 3D solid is built from this 2D face.",
            },
        ],
        animations=[{"id": "draw", "trigger": "button", "description": "Draw sides one by one", "duration": 2.4}],
    )


def apply_shape_lab_guide(lesson: dict[str, Any], query: str = "") -> dict[str, Any]:
    """Fill guidedExploration (shape-lab) and pin slider defaults from the question (e.g. side 5)."""
    viz = lesson.get("visualization") or {}
    vtype = str(viz.get("visualizationType") or "")
    if vtype == "shape-lab":
        objs = viz.get("interactiveObjects") or []
        shape = str((objs[0] or {}).get("type") or "picker").lower() if objs else "picker"
        if not lesson.get("guidedExploration"):
            lesson["guidedExploration"] = list(_SHAPE_LAB_GUIDES.get(shape, _SHAPE_LAB_GUIDES["picker"]))
    # Prefer "side … N" / "side length … N", else first in-range number for s/r/length
    side_m = re.search(
        r"\b(?:side(?:\s*length)?|length)\s*(?:of|=|:)?\s*(\d{1,2})\b",
        query or "",
        re.I,
    )
    nums = [int(side_m.group(1))] if side_m else [int(n) for n in re.findall(r"\b(\d{1,5})\b", query or "")]
    if vtype == "factor-tree":
        pair = [n for n in nums if 2 <= n <= 20000][:2]
        sliders = viz.get("sliders") or []
        for s in sliders:
            # Allow pinning large textbook numbers (e.g. 8788)
            if int(s.get("max") or 99) < 20000:
                s["max"] = 20000
        if pair and sliders:
            if len(pair) >= 1:
                sliders[0]["default"] = pair[0]
            if len(pair) >= 2 and len(sliders) > 1:
                sliders[1]["default"] = pair[1]
            elif len(pair) == 1 and len(sliders) > 1:
                # Single-number prime factorisation — mirror on B so ladder still runs
                sliders[1]["default"] = pair[0]
        # Tag perfect cube / square so the FE shows the missing multiplier
        qlow = (query or "").lower()
        if "perfect cube" in qlow:
            viz["title"] = "Perfect Cube — Division Method"
        elif "perfect square" in qlow:
            viz["title"] = "Perfect Square — Division Method"
        return lesson
    # Only simple side/radius labs — not open-box nets (sheetLength + folded length).
    if nums and vtype in ("shape-lab", "mensuration-cube", "mensuration-cylinder", "circle"):
        n = nums[0]
        for slider in viz.get("sliders") or []:
            sid = str(slider.get("id") or "")
            if sid in ("s", "length", "r", "side") and slider.get("min", 1) <= n <= slider.get("max", 10):
                slider["default"] = n
                break
    return lesson


# (pattern, spec_fn, concept, objective, explanation)
ELEMENTARY_TOPIC_RULES: list[tuple[re.Pattern[str], Callable[[], dict], str, str, str]] = [
    (
        re.compile(
            r"\b("
            r"matchsticks?|match\s*sticks?|sharing\s+sides|same\s+matchsticks?|"
            r"more\s+than\s+one\s+square|make\s+(?:two|three|\d+)\s+squares?"
            r")\b",
            re.I,
        ),
        lambda: _viz(
            "matchstick-squares",
            "Make Squares with Matchsticks",
            "Slide to build 1–4 squares in a row. Squares that share a side use fewer matchsticks!",
            sliders=[
                {"id": "squares", "label": "Squares in a row", "min": 1, "max": 4, "step": 1, "default": 2},
            ],
            calcs=[
                {"id": "sticks", "label": "Matchsticks needed", "formula": "3 * squares + 1", "unit": ""},
            ],
            buttons=[{"id": "animate", "label": "▶ Build squares", "action": "animate"}],
        ),
        "Matchstick Squares",
        "Build more than one square using the same matchsticks by sharing sides.",
        "Each new square adds 3 matchsticks when it shares a side (rule: 3n + 1).",
    ),
    (
        re.compile(
            r"(?is)^(?!.*(?:matchsticks?|perfect\s+square|square\s+root|\bsquared\b))"
            r"(?!.*\b(?:rectangle|rhombus|parallelogram)\b)"
            r".*\bsquares?\b"
        ),
        lambda: _shape_lab_spec("square"),
        "Square",
        "A square has 4 equal sides and 4 right angles.",
        "Resize the side — it stays a square. 3D analogue: cube.",
    ),
    (
        re.compile(
            r"(?is)^(?!.*\b(?:square|rhombus|parallelogram)\b).*\brectangles?\b"
        ),
        lambda: _shape_lab_spec("rectangle"),
        "Rectangle",
        "A rectangle has opposite sides equal and 4 right angles.",
        "Change length and width separately. 3D analogue: cuboid.",
    ),
    (
        re.compile(r"\b(cuboid|rectangular\s+prism|rectangular\s+box)\b", re.I),
        lambda: _shape_lab_spec("cuboid", default_view="3d"),
        "Cuboid",
        "A cuboid is a 3D box with rectangular faces.",
        "Volume = length × width × height.",
    ),
    (
        re.compile(r"\bpentagons?\b", re.I),
        lambda: _shape_lab_spec("pentagon"),
        "Pentagon",
        "A regular pentagon has 5 equal sides.",
        "Only a pentagon is shown. 3D analogue: pentagonal prism.",
    ),
    (
        re.compile(r"\bhexagons?\b", re.I),
        lambda: _shape_lab_spec("hexagon"),
        "Hexagon",
        "A regular hexagon has 6 equal sides.",
        "Only a hexagon is shown. 3D analogue: hexagonal prism.",
    ),
    (
        re.compile(r"\b(count|counting|how\s+many|objects?|apples?|stars?)\b", re.I),
        lambda: _viz(
            "counting",
            "Counting Objects",
            "Press + to add objects and count along. Great for Class 1–2!",
            sliders=[{"id": "count", "label": "Objects", "min": 0, "max": 20, "step": 1, "default": 5}],
            buttons=[
                {"id": "add", "label": "+ Add one", "action": "add"},
                {"id": "animate", "label": "▶ Count aloud", "action": "animate"},
                {"id": "reset", "label": "Reset", "action": "reset"},
            ],
        ),
        "Counting",
        "Count objects one by one.",
        "Each object is one count.",
    ),
    (
        re.compile(r"\b(number\s*line|jump|forward|backward|addition|subtract|add|plus|minus)\b", re.I),
        lambda: _viz(
            "number-line",
            "Number Line Explorer",
            "Move the marker along the number line. See addition as jumps forward and subtraction as jumps back.",
            sliders=[
                {"id": "start", "label": "Start at", "min": 0, "max": 20, "step": 1, "default": 3},
                {"id": "jump", "label": "Jump by", "min": -10, "max": 10, "step": 1, "default": 4},
            ],
            calcs=[{"id": "result", "label": "Land on", "formula": "start + jump", "unit": ""}],
            buttons=[{"id": "animate", "label": "▶ Animate jump", "action": "animate"}],
        ),
        "Number Line",
        "Use jumps on a number line for add and subtract.",
        "Moving right adds; moving left subtracts.",
    ),
    (
        re.compile(r"\b(place\s*value|ones|tens|hundreds|thousands|digit|expand)\b", re.I),
        lambda: _viz(
            "place-value",
            "Place Value Blocks",
            "Change the number and see ones, tens and hundreds blocks.",
            sliders=[{"id": "n", "label": "Number", "min": 0, "max": 999, "step": 1, "default": 247}],
            calcs=[
                {"id": "h", "label": "Hundreds", "formula": "n // 100", "unit": ""},
                {"id": "t", "label": "Tens", "formula": "(n // 10) % 10", "unit": ""},
                {"id": "o", "label": "Ones", "formula": "n % 10", "unit": ""},
            ],
        ),
        "Place Value",
        "Understand how digits show value by position.",
        "Each place is worth 10× the place to its right.",
    ),
    (
        re.compile(r"\b(multiply|multiplication|times\s*table|times\s*\d|product|array)\b", re.I),
        lambda: _viz(
            "multiplication-grid",
            "Multiplication Array",
            "Rows × columns = product. Drag sliders to build different arrays.",
            sliders=[
                {"id": "rows", "label": "Rows", "min": 1, "max": 12, "step": 1, "default": 3},
                {"id": "cols", "label": "Columns", "min": 1, "max": 12, "step": 1, "default": 4},
            ],
            calcs=[{"id": "product", "label": "Product", "formula": "rows * cols", "unit": ""}],
            buttons=[{"id": "animate", "label": "▶ Fill array", "action": "animate"}],
        ),
        "Multiplication",
        "Multiplication is repeated addition in rows and columns.",
        "3 × 4 means 3 rows of 4.",
    ),
    (
        re.compile(r"\b(division|divide|share\s*equally|quotient|remainder)\b", re.I),
        lambda: _viz(
            "bar-model",
            "Division Bar Model",
            "Share the bar into equal groups. See quotient and remainder.",
            sliders=[
                {"id": "total", "label": "Total", "min": 2, "max": 24, "step": 1, "default": 12},
                {"id": "groups", "label": "Groups", "min": 2, "max": 12, "step": 1, "default": 3},
            ],
            calcs=[{"id": "each", "label": "Each group gets", "formula": "total / groups", "unit": ""}],
        ),
        "Division",
        "Division splits a total into equal groups.",
        "Total ÷ groups = size of each group.",
    ),
    (
        re.compile(r"\b(decimal|tenth|hundredth|point\s*\d)\b", re.I),
        lambda: _viz(
            "decimal-blocks",
            "Decimal Place Value",
            "See ones, tenths and hundredths with the decimal slider.",
            sliders=[{"id": "d", "label": "Tenths", "min": 0, "max": 99, "step": 1, "default": 35}],
            calcs=[{"id": "val", "label": "Decimal", "formula": "d / 10", "unit": ""}],
        ),
        "Decimals",
        "Decimals extend place value to tenths and hundredths.",
        "The decimal point separates wholes from parts.",
    ),
    (
        re.compile(r"\b(percent|percentage|\%|out\s*of\s*100)\b", re.I),
        lambda: _viz(
            "percent-circle",
            "Percentage Circle",
            "Shade part of 100. See the percent update live.",
            sliders=[{"id": "pct", "label": "Percent", "min": 0, "max": 100, "step": 1, "default": 25}],
            calcs=[{"id": "frac", "label": "As fraction", "formula": "pct / 100", "unit": ""}],
        ),
        "Percentages",
        "Percent means out of 100.",
        "25% = 25 out of 100.",
    ),
    (
        re.compile(r"\b(ratio|proportion|scale\s*factor|mix)\b", re.I),
        lambda: _viz(
            "ratio-bar",
            "Ratio Bar Model",
            "Adjust the two parts and see the ratio change.",
            sliders=[
                {"id": "a", "label": "Part A", "min": 1, "max": 10, "step": 1, "default": 2},
                {"id": "b", "label": "Part B", "min": 1, "max": 10, "step": 1, "default": 3},
            ],
            calcs=[{"id": "total", "label": "Total parts", "formula": "a + b", "unit": ""}],
        ),
        "Ratio",
        "A ratio compares two quantities.",
        "2:3 means 2 parts to 3 parts.",
    ),
    (
        re.compile(r"\b(negative|integer|integers|below\s*zero|number\s*line.*integer)\b", re.I),
        lambda: _viz(
            "integer-line",
            "Integer Number Line",
            "Explore positive and negative numbers on a line.",
            sliders=[
                {"id": "pos", "label": "Start", "min": -10, "max": 10, "step": 1, "default": -2},
                {"id": "move", "label": "Move by", "min": -10, "max": 10, "step": 1, "default": 5},
            ],
            calcs=[{"id": "end", "label": "End at", "formula": "pos + move", "unit": ""}],
        ),
        "Integers",
        "Integers include negative and positive whole numbers.",
        "Moving left on the line means smaller numbers.",
    ),
    (
        re.compile(r"\b(clock|time|hour|minute|am|pm|o'clock)\b", re.I),
        lambda: _viz(
            "clock-time",
            "Clock & Time",
            "Move the hour and minute hands. Read the time.",
            sliders=[
                {"id": "hour", "label": "Hour", "min": 1, "max": 12, "step": 1, "default": 3},
                {"id": "minute", "label": "Minute", "min": 0, "max": 55, "step": 5, "default": 30},
            ],
            buttons=[{"id": "animate", "label": "▶ Watch time pass", "action": "animate"}],
        ),
        "Time",
        "Read hours and minutes on an analog clock.",
        "The short hand shows hours; the long hand shows minutes.",
    ),
    (
        re.compile(r"\b(rupee|rupees|money|coin|note|cost|price|change)\b", re.I),
        lambda: _viz(
            "money",
            "Money Counter",
            "Add coins and notes. See the total amount.",
            sliders=[
                {"id": "coins", "label": "₹ coins (×10)", "min": 0, "max": 20, "step": 1, "default": 5},
                {"id": "notes", "label": "₹ notes (×50)", "min": 0, "max": 10, "step": 1, "default": 2},
            ],
            calcs=[{"id": "total", "label": "Total ₹", "formula": "coins * 10 + notes * 50", "unit": ""}],
        ),
        "Money",
        "Combine coins and notes to make amounts.",
        "Count each type and add for the total.",
    ),
    (
        re.compile(r"\b(pattern|sequence|next\s*term|shape\s*pattern)\b", re.I),
        lambda: _viz(
            "pattern",
            "Pattern Builder",
            "Build a growing pattern. Predict the next term.",
            sliders=[{"id": "step", "label": "Step number", "min": 1, "max": 8, "step": 1, "default": 4}],
            calcs=[{"id": "term", "label": "Matchsticks", "formula": "2 * step + 1", "unit": ""}],
            buttons=[{"id": "animate", "label": "▶ Grow pattern", "action": "animate"}],
        ),
        "Patterns",
        "Look for a rule that repeats or grows.",
        "Find what changes each step.",
    ),
    (
        re.compile(
            r"\b("
            r"difference\s+between|compare|comparison|how\s+are|vs\.?|"
            r"square.*rectangle|rectangle.*square"
            r")\b",
            re.I,
        ),
        _quadrilateral_morph_spec,
        "Square vs Rectangle",
        "Compare squares and rectangles by morphing a quadrilateral.",
        "A square has four equal sides; a rectangle has opposite sides equal.",
    ),
    (
        re.compile(
            r"\b(2d|2\s*d|2-dimensional)\b.*\bshapes?\b|"
            r"\bshapes?\b.*\b(2d|2\s*d|2-dimensional)\b|"
            r"\b(2d\s+shapes?|basic\s+shapes?|plane\s+figures?|polygons?)\b",
            re.I,
        ),
        lambda: _shape_lab_spec("picker"),
        "2D & 3D Shapes",
        "Name a shape, then explore only that shape in 2D and 3D.",
        "Pick one shape at a time — square, rectangle, triangle, pentagon or hexagon.",
    ),
    (
        re.compile(r"\b(symmetry|symmetric|mirror|line\s*of\s*symmetry|fold)\b", re.I),
        lambda: _viz(
            "symmetry",
            "Symmetry Mirror",
            "Toggle the mirror line. See reflective symmetry.",
            sliders=[{"id": "fold", "label": "Mirror position", "min": 0, "max": 100, "step": 5, "default": 50}],
            buttons=[{"id": "animate", "label": "▶ Fold & unfold", "action": "animate"}],
        ),
        "Symmetry",
        "A shape has symmetry if both halves match when folded.",
        "The fold line is the line of symmetry.",
    ),
    (
        re.compile(r"\b(pythagoras|hypotenuse|right\s*triangle\s*theorem)\b", re.I),
        lambda: _viz(
            "pythagoras",
            "Pythagoras Theorem",
            "Adjust sides a and b. See c² = a² + b² update live.",
            sliders=[
                {"id": "a", "label": "Side a", "min": 1, "max": 10, "step": 1, "default": 3},
                {"id": "b", "label": "Side b", "min": 1, "max": 10, "step": 1, "default": 4},
            ],
            calcs=[
                {"id": "csq", "label": "c²", "formula": "a * a + b * b", "unit": ""},
            ],
        ),
        "Pythagoras Theorem",
        "In a right triangle, a² + b² = c².",
        "The hypotenuse is the longest side.",
    ),
    (
        re.compile(r"\b(sine|cosine|tangent|trigonometry|trig|sohcahtoa)\b", re.I),
        lambda: _viz(
            "trig-basic",
            "Trigonometry Explorer",
            "Change the angle and see sin, cos and tan values.",
            sliders=[{"id": "angle", "label": "Angle (degrees)", "min": 0, "max": 90, "step": 1, "default": 30}],
            buttons=[{"id": "animate", "label": "▶ Sweep angle", "action": "animate"}],
        ),
        "Trigonometry",
        "Trig ratios relate angles to sides in a right triangle.",
        "SOH CAH TOA helps remember the ratios.",
    ),
    (
        re.compile(
            r"\b(perfect\s*(?:cube|square)|smallest\s+number\s+by\s+which|"
            r"multipl(?:y|ied)\s+to\s+(?:obtain|get|make)|"
            r"hcf|lcm|g\.?c\.?d\.?|highest\s*common|lowest\s*common|least\s*common|"
            r"factor\s*tree|prime\s*factors?|prime\s*factori[sz]ation|common\s*factor)\b",
            re.I,
        ),
        lambda: _viz(
            "factor-tree",
            "Prime Factors — Division Method",
            "Division ladder for prime factors, LCM / HCF, or completing a perfect cube / square.",
            sliders=[
                {"id": "x", "label": "Number A", "min": 2, "max": 20000, "step": 1, "default": 12},
                {"id": "y", "label": "Number B", "min": 2, "max": 20000, "step": 1, "default": 18},
            ],
            buttons=[{"id": "animate", "label": "▶ Show steps", "action": "animate"}],
        ),
        "Prime Factors, HCF and LCM",
        "Use the division method to factorise; for a perfect cube, every exponent must be a multiple of 3.",
        "Missing primes make the smallest multiplier that completes the cube or square.",
    ),
    (
        re.compile(r"\b(profit|loss|discount|simple\s*interest|principal|rate)\b", re.I),
        lambda: _viz(
            "concept-explorer",
            "Money Maths Explorer",
            "Adjust amount and rate. See profit, loss or interest update.",
            sliders=[
                {"id": "principal", "label": "Principal ₹", "min": 100, "max": 10000, "step": 100, "default": 1000},
                {"id": "rate", "label": "Rate %", "min": 1, "max": 20, "step": 1, "default": 5},
            ],
            calcs=[{"id": "interest", "label": "Simple interest", "formula": "principal * rate / 100", "unit": " ₹"}],
        ),
        "Commercial Mathematics",
        "Apply percentages to money problems.",
        "Interest = Principal × Rate ÷ 100.",
    ),
    (
        re.compile(r"\b(coordinate|plot|ordered\s*pair|cartesian|x\s*axis|y\s*axis)\b", re.I),
        lambda: _viz(
            "coordinate",
            "Coordinate Plane",
            "Drag the point and read its (x, y) coordinates.",
            draggables=[
                {"id": "P", "label": "P", "initialX": 2, "initialY": 3, "color": "#3B82F6"},
            ],
        ),
        "Coordinate Geometry",
        "Every point has an x (horizontal) and y (vertical) value.",
        "The axes cross at the origin (0, 0).",
    ),
    (
        re.compile(r"\b(perimeter|distance\s*around|fence)\b", re.I),
        lambda: _viz(
            "area-resizer",
            "Perimeter & Area",
            "Resize the rectangle. Watch perimeter and area change.",
            sliders=[
                {"id": "width", "label": "Length", "min": 1, "max": 15, "step": 1, "default": 6},
                {"id": "height", "label": "Width", "min": 1, "max": 15, "step": 1, "default": 4},
            ],
            calcs=[
                {"id": "perim", "label": "Perimeter", "formula": "2 * (width + height)", "unit": ""},
                {"id": "rectArea", "label": "Area", "formula": "width * height", "unit": ""},
            ],
        ),
        "Perimeter",
        "Perimeter is the total distance around a shape.",
        "Perimeter of rectangle = 2 × (length + width).",
    ),
]
