"""
Convert mathematics notation (LaTeX, Unicode, caret) into natural spoken English for TTS.
"""

from __future__ import annotations

import re

# Unicode superscript digits
_SUPERSCRIPT_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT_TO_DIGIT = str.maketrans(_SUPERSCRIPT_CHARS, "0123456789")

# Common geometry / calculus symbols in plain text
_SYMBOL_WORDS: list[tuple[str, str]] = [
    ("∫", " integral "),
    ("∑", " sum "),
    ("∏", " product "),
    ("√", " square root of "),
    ("∞", " infinity "),
    ("π", " pi "),
    ("°", " degrees "),
    ("∠", " angle "),
    ("⊥", " perpendicular to "),
    ("∥", " parallel to "),
    ("≈", " approximately equal to "),
    ("≠", " not equal to "),
    ("≤", " less than or equal to "),
    ("≥", " greater than or equal to "),
    ("±", " plus or minus "),
    ("×", " times "),
    ("÷", " divided by "),
    ("Δ", " delta "),
    ("θ", " theta "),
    ("α", " alpha "),
    ("β", " beta "),
    ("γ", " gamma "),
]

_LATEX_COMMANDS: list[tuple[str, str]] = [
    (r"\\sqrt\[(\d+)\]\{([^}]*)\}", r"\2 to the power one over \1"),
    (r"\\sqrt\{([^}]*)\}", r"square root of \1"),
    (r"\\iint", "double integral"),
    (r"\\iiint", "triple integral"),
    (r"\\oint", "contour integral"),
    (r"\\int", "integral"),
    (r"\\sum", "sum"),
    (r"\\prod", "product"),
    (r"\\lim", "limit"),
    (r"\\infty", "infinity"),
    (r"\\pi", "pi"),
    (r"\\theta", "theta"),
    (r"\\alpha", "alpha"),
    (r"\\beta", "beta"),
    (r"\\gamma", "gamma"),
    (r"\\Delta", "delta"),
    (r"\\angle", "angle"),
    (r"\\triangle", "triangle"),
    (r"\\perp", "perpendicular to"),
    (r"\\parallel", "parallel to"),
    (r"\\circ", "degrees"),
    (r"\\degree", "degrees"),
    (r"\\pm", "plus or minus"),
    (r"\\mp", "minus or plus"),
    (r"\\leq", "less than or equal to"),
    (r"\\geq", "greater than or equal to"),
    (r"\\neq", "not equal to"),
    (r"\\approx", "approximately equal to"),
    (r"\\equiv", "equivalent to"),
    (r"\\cdot", " times "),
    (r"\\times", " times "),
    (r"\\div", " divided by "),
    (r"\\overline\{([^}]*)\}", r"\1 bar"),
    (r"\\bar\{([^}]*)\}", r"\1 bar"),
    (r"\\vec\{([^}]*)\}", r"vector \1"),
    (r"\\mathrm\{([^}]*)\}", r"\1"),
    (r"\\text\{([^}]*)\}", r"\1"),
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\,", " "),
    (r"\\;", " "),
    (r"\\!", ""),
]

_CARET_POWER_RE = re.compile(
    r"(\([^)]+\)|[A-Za-z][A-Za-z0-9]*|\d+(?:\.\d+)?)\s*\^\s*(\d+)\b"
)
_UNICODE_POWER_GLUED_RE = re.compile(
    r"([A-Za-z0-9]+)([" + re.escape(_SUPERSCRIPT_CHARS) + r"])"
)


def power_phrase(exponent: str) -> str:
    exp = exponent.strip()
    if exp == "2":
        return " squared"
    if exp == "3":
        return " cubed"
    if exp == "1":
        return ""
    if exp.isdigit():
        return f" to the power {exp}"
    return f" to the power {exp}"


def apply_power_to_base(base: str, exponent: str) -> str:
    base = base.strip()
    if not base:
        return power_phrase(exponent).strip()
    return f"{base}{power_phrase(exponent)}"


def _superscript_digits_to_number(chars: str) -> str:
    return chars.translate(_SUPERSCRIPT_TO_DIGIT)


def _latex_powers(inner: str) -> str:
    """Handle ^{2}, ^2, subscripts, and parenthesized bases."""

    def _repl(m: re.Match[str]) -> str:
        return apply_power_to_base(m.group(1), m.group(2))

    for pattern in (
        r"(\([^)]+\))\s*\^\{(\d+)\}",
        r"(\([^)]+\))\s*\^(\d+)",
        r"([A-Za-z0-9]+)\s*\^\{(\d+)\}",
        r"([A-Za-z0-9]+)\s*\^(\d+)",
    ):
        inner = re.sub(pattern, _repl, inner)

    inner = re.sub(r"\^\{([^}]+)\}", lambda m: f" to the power {m.group(1).strip()}", inner)
    inner = re.sub(r"\^([A-Za-z])", r" to the power \1", inner)
    inner = re.sub(r"_\{([^}]+)\}", r" sub \1", inner)
    inner = re.sub(r"_(\w)", r" sub \1", inner)
    return inner


def latex_to_speech(inner: str) -> str:
    """Convert a LaTeX math fragment to speakable English."""
    inner = (inner or "").strip().strip("$")
    if not inner:
        return ""

    for pattern, repl in _LATEX_COMMANDS:
        inner = re.sub(pattern, repl, inner)

    inner = re.sub(
        r"\\frac\{([^}]*)\}\{([^}]*)\}",
        lambda m: f"{m.group(1).strip()} over {m.group(2).strip()}",
        inner,
    )

    inner = _latex_powers(inner)

    # Remaining unknown commands
    inner = re.sub(r"\\[a-zA-Z]+", " ", inner)
    inner = re.sub(r"[{}]", " ", inner)
    inner = re.sub(r"\s+", " ", inner).strip()
    return inner


def plain_math_to_speech(text: str) -> str:
    """Convert Unicode / caret math in plain tutor text to spoken form."""
    if not text:
        return ""

    for sym, word in _SYMBOL_WORDS:
        text = text.replace(sym, word)

    def _caret_repl(m: re.Match[str]) -> str:
        return apply_power_to_base(m.group(1), m.group(2))

    text = _CARET_POWER_RE.sub(_caret_repl, text)

    def _unicode_glued(m: re.Match[str]) -> str:
        return apply_power_to_base(m.group(1), _superscript_digits_to_number(m.group(2)))

    text = _UNICODE_POWER_GLUED_RE.sub(_unicode_glued, text)

    # Phrases tutors use — ensure clear speech (no change needed but normalize)
    text = re.sub(r"\bdx\b", "d x", text, flags=re.I)
    text = re.sub(r"\bdy\b", "d y", text, flags=re.I)
    text = re.sub(r"\bdt\b", "d t", text, flags=re.I)

    return text


def _strip_orphan_dollar_signs(text: str) -> str:
    """Remove leftover $ from incomplete streaming chunks."""
    text = re.sub(r"\$\$+", " ", text)
    text = re.sub(r"\$", " ", text)
    return text


def _convert_bare_latex_commands(text: str) -> str:
    """LaTeX commands that appear outside $ delimiters (common in streamed tutor text)."""
    text = re.sub(
        r"\\frac\{([^}]*)\}\{([^}]*)\}",
        lambda m: f" {m.group(1).strip()} over {m.group(2).strip()} ",
        text,
    )
    for pattern, repl in _LATEX_COMMANDS:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[{}]", " ", text)
    return text


def math_to_speech(text: str) -> str:
    """Full math speech pass: LaTeX blocks then plain notation."""
    display_re = re.compile(r"\$\$[^$]+\$\$", re.DOTALL)
    inline_re = re.compile(r"\$([^$]+)\$")

    text = display_re.sub(lambda m: f" {latex_to_speech(m.group(0))} ", text)
    text = inline_re.sub(lambda m: f" {latex_to_speech(m.group(1))} ", text)
    text = _convert_bare_latex_commands(text)
    text = plain_math_to_speech(text)
    text = _strip_orphan_dollar_signs(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
