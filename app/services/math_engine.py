"""
SymPy-backed math engine for school mathematics (Classes 1–12).

Parses common student questions, computes verified steps, and injects results into
the LLM prompt so explanations use correct arithmetic and algebra.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from sympy import Eq, Rational, simplify, solve, sympify, symbols
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

_X = symbols("x")
_Y = symbols("y")

_INDIAN_NUM = r"(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)"
_DAYS_PER_YEAR = 365


@dataclass
class MathStep:
    label: str
    latex_lines: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class MathEngineResult:
    solved: bool
    kind: str
    class_band: str
    to_find: str
    given: list[str]
    concept: str
    formula_words: str
    formula_latex: str
    steps: list[MathStep]
    solution_latex: list[str]
    final_answer: str
    quick_check: str

    def to_prompt_block(self) -> str:
        if not self.solved:
            return ""
        lines = [
            "MATH ENGINE (SymPy-verified — mandatory):",
            "Use the exact numbers, formulas, and calculation lines below.",
            "Do NOT recalculate or change any verified value.",
            f"Problem type: {self.kind}",
            f"Class band: {self.class_band}",
            "",
            f"To find: {self.to_find}",
            "Given:",
            *[f"- {g}" for g in self.given],
            f"Concept: {self.concept}",
            "",
            "Formula (both lines required):",
            self.formula_words,
            self.formula_latex,
            "",
            "Solution (use these lines under **Solution**):",
            "Substituting the values:",
            *[line for line in self.solution_latex],
            "",
            f"Final answer (use exactly): {self.final_answer}",
            f"Quick check: {self.quick_check}",
        ]
        for step in self.steps:
            if step.label:
                lines.append(step.label)
            if step.note:
                lines.append(step.note)
            lines.extend(step.latex_lines)
        return "\n".join(lines)


def _class_band(class_level: str) -> str:
    m = re.search(r"(\d+)", class_level or "")
    if not m:
        return "6-8"
    n = int(m.group(1))
    if n <= 5:
        return "1-5"
    if n <= 8:
        return "6-8"
    if n <= 10:
        return "9-10"
    return "11-12"


def parse_number(text: str) -> float:
    return float((text or "").replace(",", "").strip())


def format_indian(n: int | float) -> str:
    """Format integers using Indian grouping (e.g. 384400 → 3,84,400)."""
    if isinstance(n, float):
        if n == int(n):
            n = int(n)
        else:
            return f"{n:g}"
    sign = "-" if n < 0 else ""
    n = abs(int(n))
    s = str(n)
    if len(s) <= 3:
        return sign + s
    last3 = s[-3:]
    rest = s[:-3]
    groups: list[str] = []
    while rest:
        groups.append(rest[-2:])
        rest = rest[:-2]
    groups.reverse()
    return sign + ",".join(groups + [last3])


def latex_number(n: int | float) -> str:
    """LaTeX-safe number with Indian-style comma grouping."""
    if isinstance(n, float) and n != int(n):
        return f"{n:g}"
    formatted = format_indian(int(n))
    return formatted.replace(",", "{,}")


def _extract_numbers_with_units(text: str) -> list[tuple[float, str]]:
    found: list[tuple[float, str]] = []
    for m in re.finditer(
        rf"{_INDIAN_NUM}\s*(km|m|cm|mm|years?|days?|hours?|minutes?|%)",
        text,
        re.I,
    ):
        found.append((parse_number(m.group(1)), m.group(2).lower()))
    return found


def _extract_km_from_context(context: str, *, keyword: str = "") -> float | None:
    blob = context or ""
    patterns = [
        rf"{keyword}.*?{_INDIAN_NUM}\s*km",
        rf"distance.*?{_INDIAN_NUM}\s*km",
        rf"{_INDIAN_NUM}\s*km.*?{keyword}",
    ]
    for pat in patterns:
        m = re.search(pat, blob, re.I | re.S)
        if m:
            return parse_number(m.group(1))
    return None


def _try_linear_equation(query: str, class_band: str) -> MathEngineResult | None:
    if not re.search(r"=\s*[\d.]", query):
        return None
    if not re.search(r"[a-zA-Z]", query):
        return None

    eq_part = re.search(r"([^?]*=\s*[\d.]+)", query)
    if not eq_part:
        return None
    expr = eq_part.group(1).strip().rstrip(".")
    expr = re.sub(r"^(?:solve|find)\s+(?:for\s+[a-zA-Z]\s*[:,-]?\s*)?", "", expr, flags=re.I).strip()
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)
    var_char = re.search(r"([a-zA-Z])", expr)
    if not var_char:
        return None
    var = symbols(var_char.group(1))
    try:
        left, right = expr.split("=", 1)
        equation = Eq(
            parse_expr(left.strip(), transformations=_TRANSFORMATIONS),
            parse_expr(right.strip(), transformations=_TRANSFORMATIONS),
        )
        solutions = solve(equation, var)
        if not solutions:
            return None
        sol = simplify(solutions[0])
    except Exception:
        return None

    formula_words = f"Solve the equation for {var}"
    formula_latex = f"$$ {left.strip()} = {right.strip()} $$"
    sol_display = latex_number(float(sol)) if sol.is_number else str(sol)
    solution = [f"$$ {var} = {sol_display} $$"]
    return MathEngineResult(
        solved=True,
        kind="linear_equation",
        class_band=class_band,
        to_find=f"Find the value of {var}.",
        given=[expr],
        concept="Use inverse operations to isolate the variable on one side.",
        formula_words=formula_words,
        formula_latex=formula_latex,
        steps=[],
        solution_latex=solution,
        final_answer=f"**{var} = {sol}**",
        quick_check=f"Substitute {var} = {sol} back into the equation to verify both sides match.",
    )


def _try_percentage(query: str, class_band: str) -> MathEngineResult | None:
    m = re.search(
        rf"(?:what\s+is\s+)?{_INDIAN_NUM}\s*%\s+of\s+{_INDIAN_NUM}",
        query,
        re.I,
    )
    if not m:
        m = re.search(
            rf"find\s+{_INDIAN_NUM}\s*%\s+of\s+{_INDIAN_NUM}",
            query,
            re.I,
        )
    if not m:
        return None
    pct = parse_number(m.group(1))
    base = parse_number(m.group(2))
    result = Rational(pct, 100) * base
    result_f = float(simplify(result))
    formula_words = "Result = (Percentage ÷ 100) × Base"
    formula_latex = r"$$\text{Result} = \frac{\text{Percentage}}{100} \times \text{Base}$$"
    solution = [
        f"$$= \\frac{{{latex_number(pct)}}}{{100}} \\times {latex_number(base)}$$",
        f"$$= {latex_number(result_f)}$$",
    ]
    return MathEngineResult(
        solved=True,
        kind="percentage",
        class_band=class_band,
        to_find=f"Find {format_indian(pct)}% of {format_indian(base)}.",
        given=[f"Percentage = {format_indian(pct)}%", f"Base = {format_indian(base)}"],
        concept="A percentage is a fraction out of 100; multiply the base by the percentage divided by 100.",
        formula_words=formula_words,
        formula_latex=formula_latex,
        steps=[],
        solution_latex=solution,
        final_answer=f"**{format_indian(result_f)}**",
        quick_check=f"Check: {format_indian(result_f)} is less than the base {format_indian(base)} when the percentage is below 100.",
    )


def _try_fraction_simplify(query: str, class_band: str) -> MathEngineResult | None:
    m = re.search(
        r"(?:simplify|reduce|write in lowest terms)\s+(\d+)\s*/\s*(\d+)",
        query,
        re.I,
    )
    if not m:
        m = re.search(r"(\d+)\s*/\s*(\d+)", query)
        if not m or "simplify" not in query.lower():
            return None
    num, den = int(m.group(1)), int(m.group(2))
    if den == 0:
        return None
    frac = Rational(num, den)
    simplified = simplify(frac)
    formula_words = "Simplified fraction = Numerator ÷ HCF ÷ Denominator ÷ HCF"
    formula_latex = r"$$\frac{a}{b} = \frac{a \div \text{HCF}}{b \div \text{HCF}}$$"
    solution = [
        f"$$= \\frac{{{simplified.p}}}{{{simplified.q}}}$$",
    ]
    return MathEngineResult(
        solved=True,
        kind="fraction_simplify",
        class_band=class_band,
        to_find=f"Simplify {num}/{den} to lowest terms.",
        given=[f"Fraction = {num}/{den}"],
        concept="Divide numerator and denominator by their highest common factor (HCF).",
        formula_words=formula_words,
        formula_latex=formula_latex,
        steps=[],
        solution_latex=solution,
        final_answer=f"**{simplified.p}/{simplified.q}**",
        quick_check=f"Multiply {simplified.p}/{simplified.q} back — it equals the original {num}/{den}.",
    )


def _try_arithmetic(query: str, class_band: str) -> MathEngineResult | None:
    m = re.search(
        rf"(?:what\s+is|calculate|evaluate|find)\s+(.+?)(?:\?|$)",
        query,
        re.I,
    )
    if not m:
        return None
    expr_raw = m.group(1).strip()
    if not re.search(r"[\+\-\×\÷*/]", expr_raw):
        return None
    if re.search(r"[a-zA-Z]", expr_raw):
        return None
    cleaned = (
        expr_raw.replace("×", "*")
        .replace("÷", "/")
        .replace(",", "")
    )
    try:
        value = simplify(sympify(cleaned))
        num = float(value)
    except Exception:
        return None

    formula_words = "Result = evaluate the expression using order of operations (BODMAS)"
    formula_latex = f"$$\\text{{Result}} = {cleaned.replace('*', ' \\times ').replace('/', ' \\div ')}$$"
    solution = [f"$$= {latex_number(num)}$$"]
    return MathEngineResult(
        solved=True,
        kind="arithmetic",
        class_band=class_band,
        to_find=f"Evaluate {expr_raw}.",
        given=[f"Expression = {expr_raw}"],
        concept="Follow BODMAS: brackets, orders, division/multiplication, addition/subtraction.",
        formula_words=formula_words,
        formula_latex=formula_latex,
        steps=[],
        solution_latex=solution,
        final_answer=f"**{format_indian(num) if num == int(num) else num}**",
        quick_check="Estimate with rounded numbers to see if the result is reasonable.",
    )


def _try_travel_distance_compare(
    query: str, class_band: str, chapter_context: str
) -> MathEngineResult | None:
    if not re.search(r"\bcan (?:you|we|i) reach\b", query, re.I):
        return None
    daily_m = re.search(rf"travel\s+{_INDIAN_NUM}\s*km\s+(?:every\s+)?day", query, re.I)
    if not daily_m:
        daily_m = re.search(rf"{_INDIAN_NUM}\s*km\s+(?:every\s+)?day", query, re.I)
    years_m = re.search(rf"(\d+)\s*years?", query, re.I)
    if not daily_m or not years_m:
        return None

    daily_km = parse_number(daily_m.group(1))
    years = int(years_m.group(1))
    target_km = _extract_km_from_context(chapter_context, keyword="moon")
    if target_km is None:
        target_km = _extract_km_from_context(chapter_context)
    if target_km is None:
        for val, unit in _extract_numbers_with_units(query):
            if unit.startswith("km"):
                target_km = val
                break
    if target_km is None:
        return None

    days = years * _DAYS_PER_YEAR
    total = daily_km * days
    can_reach = total >= target_km
    diff = abs(target_km - total)

    formula_words = "Total Distance = daily distance × number of days"
    formula_latex = r"$$\text{Total Distance} = \text{daily distance} \times \text{number of days}$$"
    solution = [
        f"$$= {years} \\times {_DAYS_PER_YEAR}$$",
        f"$$= {latex_number(days)}$$",
        f"$$= {latex_number(daily_km)} \\times {latex_number(days)}$$",
        f"$$= {latex_number(total)} \\text{{ km}}$$",
    ]
    if can_reach:
        final = f"**Yes**, you can reach — you travel {format_indian(total)} km, which is at least {format_indian(target_km)} km."
        quick = (
            f"Total {format_indian(total)} km ≥ target {format_indian(target_km)} km "
            f"(extra {format_indian(diff)} km)."
        )
    else:
        final = f"**No**, you cannot reach — you travel {format_indian(total)} km, but the distance is {format_indian(target_km)} km."
        quick = (
            f"Total {format_indian(total)} km < target {format_indian(target_km)} km "
            f"(short by {format_indian(diff)} km)."
        )

    return MathEngineResult(
        solved=True,
        kind="travel_distance_comparison",
        class_band=class_band,
        to_find="Check whether the total distance travelled in the given time is enough to reach the target.",
        given=[
            f"Daily travel = {format_indian(daily_km)} km",
            f"Time = {years} years ({_DAYS_PER_YEAR} days per year)",
            f"Target distance = {format_indian(target_km)} km",
        ],
        concept="Multiply daily distance by the total number of days, then compare with the target distance.",
        formula_words=formula_words,
        formula_latex=formula_latex,
        steps=[
            MathStep(
                label="Step 1: Find the number of days.",
                latex_lines=[f"$$= {years} \\times {_DAYS_PER_YEAR}$$", f"$$= {latex_number(days)}$$"],
            ),
            MathStep(
                label="Step 2: Find the total distance.",
                latex_lines=[
                    f"$$= {latex_number(daily_km)} \\times {latex_number(days)}$$",
                    f"$$= {latex_number(total)} \\text{{ km}}$$",
                ],
            ),
            MathStep(
                label="Step 3: Compare with the target distance.",
                note=(
                    f"Target = {format_indian(target_km)} km; "
                    f"Travelled = {format_indian(total)} km."
                ),
            ),
        ],
        solution_latex=solution,
        final_answer=final,
        quick_check=quick,
    )


def _try_rate_total(query: str, class_band: str) -> MathEngineResult | None:
    """Rate × time problems without reach comparison (e.g. buses, total people)."""
    if re.search(r"\bcan (?:you|we|i) reach\b", query, re.I):
        return None
    daily_m = re.search(rf"({_INDIAN_NUM})\s*(?:people|students|items|buses)?\s*(?:per|every|each)\s*(?:day|hour)", query, re.I)
    time_m = re.search(rf"(\d+)\s*(years?|days?|hours?)", query, re.I)
    total_m = re.search(rf"(?:total|how many)\s+.*?{_INDIAN_NUM}", query, re.I)
    if not daily_m:
        daily_m = re.search(rf"({_INDIAN_NUM})\s*(?:per|every|each)\s*(?:day|hour)", query, re.I)
    if not daily_m or not time_m:
        return None
    if not re.search(r"\b(how many|total|find|calculate)\b", query, re.I):
        return None

    rate = parse_number(daily_m.group(1))
    units = int(time_m.group(1))
    time_unit = time_m.group(2).lower()
    multiplier = units
    time_label = f"{units} {time_unit}"
    if time_unit.startswith("year"):
        multiplier = units * _DAYS_PER_YEAR
        time_label = f"{units} years × {_DAYS_PER_YEAR} days"

    total = rate * multiplier
    formula_words = "Total = rate per day × number of days"
    formula_latex = r"$$\text{Total} = \text{rate per day} \times \text{number of days}$$"
    solution = [
        f"$$= {latex_number(rate)} \\times {latex_number(multiplier)}$$",
        f"$$= {latex_number(total)}$$",
    ]
    return MathEngineResult(
        solved=True,
        kind="rate_multiplication",
        class_band=class_band,
        to_find="Find the total amount over the given time period.",
        given=[f"Rate = {format_indian(rate)} per day", f"Time = {time_label}"],
        concept="When a fixed amount repeats each day, multiply the daily amount by the number of days.",
        formula_words=formula_words,
        formula_latex=formula_latex,
        steps=[],
        solution_latex=solution,
        final_answer=f"**{format_indian(total)}**",
        quick_check="Divide the total by the number of days — you should get the daily rate back.",
    )


_SOLVERS: list[Callable[..., MathEngineResult | None]] = [
    _try_travel_distance_compare,
    _try_linear_equation,
    _try_percentage,
    _try_fraction_simplify,
    _try_arithmetic,
    _try_rate_total,
]


def try_solve(
    query: str,
    *,
    class_level: str = "",
    chapter_context: str = "",
) -> MathEngineResult | None:
    """
    Attempt to solve a school math question with SymPy.
    Returns None when the question is not recognised or not computable.
    """
    q = (query or "").strip()
    if not q:
        return None
    band = _class_band(class_level)
    for solver in _SOLVERS:
        try:
            if solver is _try_travel_distance_compare:
                result = solver(q, band, chapter_context)
            else:
                result = solver(q, band)
        except Exception:
            result = None
        if result and result.solved:
            return result
    return None
