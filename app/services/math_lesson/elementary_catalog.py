"""
Class 1–10 mathematics — elementary & middle-school visualization rules.

Complements textbook_catalog.py (Class 6–10 advanced topics).
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.services.math_lesson.textbook_catalog import _lesson, _quadrilateral_morph_spec, _viz

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
        re.compile(r"\b(2d|2\s*d|shape|square|rectangle|pentagon|hexagon)\b", re.I),
        lambda: _viz(
            "shapes-basic",
            "2D Shapes Explorer",
            "Pick a shape. Count its sides and corners.",
            sliders=[{"id": "sides", "label": "Sides", "min": 3, "max": 8, "step": 1, "default": 4}],
            buttons=[{"id": "animate", "label": "▶ Draw shape", "action": "animate"}],
        ),
        "2D Shapes",
        "Shapes are named by their sides and corners.",
        "A triangle has 3 sides; a square has 4 equal sides.",
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
        re.compile(r"\b(hcf|lcm|highest\s*common|lowest\s*common|factor\s*tree)\b", re.I),
        lambda: _viz(
            "factor-tree",
            "Factor Tree / HCF & LCM",
            "Pick two numbers. Explore common factors.",
            sliders=[
                {"id": "x", "label": "Number A", "min": 2, "max": 48, "step": 1, "default": 12},
                {"id": "y", "label": "Number B", "min": 2, "max": 48, "step": 1, "default": 18},
            ],
        ),
        "HCF and LCM",
        "HCF is the largest shared factor; LCM is the smallest shared multiple.",
        "Break numbers into prime factors.",
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
