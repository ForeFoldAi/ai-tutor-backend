"""
Read chemical formulas aloud the classroom way: H₂O -> "H 2 O", Ca(OH)₂ -> "C A O H whole 2".

Edge TTS silently drops Unicode subscripts (H₂O came out as "HO") and slurs packed
symbols ("NaCl" as one word), so each element symbol is spelled letter by letter.
Mirrors chemistrySpeech in ai-tutor-backend-voice/src/audio/chemistry-speech.ts.
"""

from __future__ import annotations

import re

ELEMENTS = frozenset(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
    Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce
    Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
    Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl
    Mc Lv Ts Og
    """.split()
)

_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")

_LATEX_WRAP_RE = re.compile(r"\\(?:ce|text|mathrm|rm|textrm|mathbf)\s*\{([^{}]*)\}")
_LATEX_ARROW_RE = re.compile(r"\\(?:rightarrow|longrightarrow|to)\b")
_LATEX_EQ_RE = re.compile(r"\\(?:rightleftharpoons|leftrightharpoons|rightleftarrows)\b")

_FORMULA_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(\d+)?"
    r"((?=\(?[A-Z])(?:\((?=[A-Z])|[A-Z][a-z]?|\d+|[₀-₉]+|_\{?\d+\}?|\))+)"
    r"(\^\{?\d*[+-]\}?|[⁰¹²³⁴⁵⁶⁷⁸⁹]*[⁺⁻])?"
    r"(\((?:aq|s|l|g)\))?"
    r"(?![A-Za-z0-9])"
)

_STATES = {"(aq)": "aqueous", "(s)": "solid", "(l)": "liquid", "(g)": "gas"}


def _spell(core: str, *, charged: bool = False) -> str | None:
    """Spoken form of a formula body, or None when it isn't a formula."""
    s = core.translate(_SUB).replace("_", "").replace("{", "").replace("}", "")
    words: list[str] = []
    i = depth = elements = two_letter = caps = 0
    has_digit = False
    while i < len(s):
        c = s[i]
        if c == "(":
            depth += 1
            i += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return None
            if i + 1 < len(s) and s[i + 1].isdigit():
                words.append("whole")
            i += 1
        elif c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            if not words:
                return None
            words.append(str(int(s[i:j])))
            has_digit = True
            i = j
        else:
            pair = s[i : i + 2]
            if len(pair) == 2 and pair[1].islower() and pair in ELEMENTS:
                words.append(" ".join(pair.upper()))
                two_letter += 1
                i += 2
            elif c in ELEMENTS:
                words.append(c)
                i += 1
            else:
                return None
            elements += 1
            caps += 1
    if depth or not elements:
        return None
    # Without a digit only accept clear compounds (NaCl, HCl, NaOH) — never plain words
    # or acronyms like "No", "IS", "OK".
    if not has_digit and not charged and not (caps >= 2 and two_letter):
        return None
    return " ".join(words)


def _charge_words(charge: str) -> str:
    c = charge.translate(_SUP).replace("^", "").replace("{", "").replace("}", "")
    sign = "plus" if c.endswith("+") else "minus"
    n = c[:-1]
    return f"{n} {sign}" if n else sign


def chemistry_to_speech(text: str) -> str:
    if not text or not re.search(r"[A-Z]", text):
        return text
    t = _LATEX_WRAP_RE.sub(r"\1", text)
    converted = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal converted
        coef, core, charge, state = m.group(1), m.group(2), m.group(3), m.group(4)
        lead = trail = ""
        # Parentheses around the formula are prose, not part of it: "(H2O)", "(CO2 and".
        while core.startswith("(") and core.count("(") > core.count(")"):
            lead += "("
            core = core[1:]
        while core.endswith(")") and core.count(")") > core.count("("):
            trail = ")" + trail
            core = core[:-1]
        if core.startswith("(") and core.endswith(")") and not charge and not coef:
            inner = core[1:-1]
            if "(" not in inner and ")" not in inner:
                lead, core, trail = lead + "(", inner, ")" + trail
        spelled = _spell(core, charged=bool(charge)) if core else None
        if spelled is None:
            return m.group(0)
        out = spelled
        if charge:
            out += " " + _charge_words(charge)
        if state:
            out += " " + _STATES[state]
        if coef:
            out = f"{int(coef)} {out}"
        converted += 1
        return f"{lead}{out}{trail}"

    t = _FORMULA_RE.sub(_sub, t)
    if converted:
        t = _LATEX_EQ_RE.sub(" ⇌ ", t)
        t = _LATEX_ARROW_RE.sub(" → ", t)
        t = re.sub(r"\s*(?:⇌|⇄|<=>|<->)\s*", " is in equilibrium with ", t)
        t = re.sub(r"\s*(?:→|⟶|-->|->)\s*", " gives ", t)
        t = re.sub(r"\s*[↑↓]", "", t)
    return t
