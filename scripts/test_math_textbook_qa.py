#!/usr/bin/env python3
"""
Offline audit: Class 9 Mathematics textbook queries → correct interactive viz + SymPy.

Usage:
  cd ai-tutor-backend && PYTHONPATH=. python scripts/test_math_textbook_qa.py
"""

from __future__ import annotations

import sys
from collections import defaultdict

from app.services.math_engine import try_solve
from app.services.math_lesson.fallbacks import build_fallback_math_lesson
from app.services.math_lesson.visual_catalog import match_math_visualization

CLASS = "CLASS_9"

# (chapter_tag, query, expected_viz_type, expect_sympy)
CASES: list[tuple[str, str, str, bool]] = [
    ("real_numbers", "What are real numbers? Explain with number line.", "number-line", False),
    ("real_numbers", "How do we represent sqrt(2) on the number line?", "number-line", False),
    ("real_numbers", "What are irrational numbers?", "number-line", False),
    ("polynomials", "Prove (a+b)^2 = a^2 + 2ab + b^2 using algebra tiles.", "algebra-tiles", False),
    ("polynomials", "Factorise x^2 + 5x + 6", "factor-rectangle", False),
    ("polynomials", "What is the remainder theorem?", "factor-rectangle", False),
    ("geometry", "What are the basic elements of geometry — point, line segment and ray?", "geometry-basics", False),
    ("lines_angles", "A transversal cuts two parallel lines. Explain corresponding angles.", "parallel-transversal", False),
    ("lines_angles", "What are alternate interior angles when a transversal crosses parallel lines?", "parallel-transversal", False),
    ("linear_equations", "Solve 3x - 5 = 16", "linear-graph", True),
    ("linear_equations", "Solve 2x + 7 = 19", "linear-graph", True),
    ("linear_equations", "A number when added to 7 gives 51. Find the number.", "linear-graph", True),
    ("linear_equations", "The cost of 5 pens is Rs 50. Find the cost of one pen.", "linear-graph", False),
    ("triangles", "In a triangle angle A = 50° and angle B = 60°. Find angle C.", "triangle-angle-sum", True),
    ("triangles", "Prove that the sum of angles in a triangle is 180 degrees.", "triangle-angle-sum", False),
    ("quadrilaterals", "What is a parallelogram? How is it different from a rectangle?", "quadrilateral-morph", False),
    ("areas", "Find the area of a triangle with base 10 cm and height 6 cm.", "area-resizer", True),
    ("areas", "Find the area of a parallelogram with base 8 cm and height 5 cm.", "area-resizer", True),
    ("statistics", "Find mean, median and mode of marks 12, 15, 18, 15, 20.", "statistics-lab", True),
    ("mensuration", "Find the volume of a cylinder of radius 7 cm and height 10 cm.", "mensuration-cylinder", True),
    ("mensuration", "Find the surface area and volume of a cube of side 5 cm.", "mensuration-cube", True),
    ("constructions", "How to construct a perpendicular bisector using compass and ruler?", "geometry-construction", False),
    ("probability", "What is the probability of getting a head when a coin is tossed?", "probability-coin", True),
    ("probability", "Probability of getting an even number when a die is thrown.", "probability-dice", True),
    ("probability", "A die is rolled 1000 times. What is experimental probability of getting 6?", "probability-dice", False),
    ("circles", "What is the area of a circle with radius 7 cm?", "circle", True),
    ("circles", "Explain circumference and diameter of a circle.", "circle", False),
    ("circles", "What is a tangent to a circle? Explain point of contact.", "circle-tangent", False),
]

WEAK_TYPES = frozenset({"concept-explorer", "generic", "shapes-basic", "probability", "bar-model"})


def main() -> int:
    viz_ok = viz_fail = sympy_ok = sympy_miss = 0
    by_chapter: dict[str, list[str]] = defaultdict(list)

    print(f"Class 9 Mathematics Textbook QA Audit ({len(CASES)} questions)\n")
    print(f"{'STATUS':6} {'VIZ':22} {'SYM':4} | Query")
    print("-" * 90)

    for chapter, query, expected_viz, expect_sympy in CASES:
        lesson = match_math_visualization(query, CLASS)
        vtype = lesson["visualization"]["visualizationType"]
        viz_match = vtype == expected_viz
        if viz_match:
            viz_ok += 1
            viz_status = "OK"
        else:
            viz_fail += 1
            viz_status = "FAIL"
            by_chapter[chapter].append(f"{query[:50]} → got {vtype}, expected {expected_viz}")

        sympy = try_solve(query, class_level=CLASS)
        sym_ok = bool(sympy and sympy.solved) if expect_sympy else True
        if expect_sympy:
            if sym_ok:
                sympy_ok += 1
            else:
                sympy_miss += 1

        sym_label = "yes" if sympy and sympy.solved else "no"
        weak = " [WEAK]" if vtype in WEAK_TYPES else ""
        print(f"{viz_status:6} {vtype:22}{weak:7} {sym_label:4} | {query[:55]}")

    print("\n" + "=" * 90)
    print(f"Visualization: {viz_ok}/{len(CASES)} correct ({100*viz_ok//len(CASES)}%)")
    print(f"SymPy (expected): {sympy_ok}/{sum(1 for c in CASES if c[3])} solved")
    if by_chapter:
        print("\nFailures by chapter:")
        for ch, fails in sorted(by_chapter.items()):
            print(f"  {ch}:")
            for f in fails:
                print(f"    - {f}")

    return 1 if viz_fail or sympy_miss else 0


if __name__ == "__main__":
    sys.exit(main())
