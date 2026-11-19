"""Fixed light design palettes — aligned with frontend design-tokens.ts (Desmos-grade)."""

from __future__ import annotations

import re

PALETTE_IDS = ("primary-1to2", "primary-3to5", "middle-6to8", "technical-9to10")

# Signature colors shared across every lesson
_BLUE = "#2d70b3"
_RED = "#c74440"
_GREEN = "#388c46"
_PURPLE = "#6042a6"
_ORANGE = "#e08a2b"

PALETTES: dict[str, dict[str, str]] = {
    "primary-1to2": {
        "primary": _BLUE,
        "secondary": _GREEN,
        "accent": _ORANGE,
        "background": "#ffffff",
        "text": "#1a1a1f",
        "surface": "#ffffff",
        "muted": "#6b6c76",
        "gridLine": "#e8e8ec",
        "axisLine": "#c4c5cb",
        "shadow": "#1a1a1f",
    },
    "primary-3to5": {
        "primary": _BLUE,
        "secondary": _GREEN,
        "accent": _ORANGE,
        "background": "#fafafa",
        "text": "#1a1a1f",
        "surface": "#ffffff",
        "muted": "#6b6c76",
        "gridLine": "#e8e8ec",
        "axisLine": "#c4c5cb",
        "shadow": "#1a1a1f",
    },
    "middle-6to8": {
        "primary": _BLUE,
        "secondary": _GREEN,
        "accent": _ORANGE,
        "background": "#fafafa",
        "text": "#1a1a1f",
        "surface": "#ffffff",
        "muted": "#6b6c76",
        "gridLine": "#e8e8ec",
        "axisLine": "#c4c5cb",
        "shadow": "#1a1a1f",
    },
    "technical-9to10": {
        "primary": _BLUE,
        "secondary": _PURPLE,
        "accent": _ORANGE,
        "background": "#fafafa",
        "text": "#1a1a1f",
        "surface": "#ffffff",
        "muted": "#5c5d66",
        "gridLine": "#e8e8ec",
        "axisLine": "#c4c5cb",
        "shadow": "#1a1a1f",
    },
}

_CLASS_NUM_RE = re.compile(r"(?:class[_\s]*)?(\d{1,2})", re.I)


def parse_class_num(class_level: str) -> int:
    if not class_level:
        return 6
    m = _CLASS_NUM_RE.search(class_level.replace("_", " "))
    if m:
        return max(1, min(10, int(m.group(1))))
    return 6


def class_level_to_default_palette(class_level: str = "") -> str:
    n = parse_class_num(class_level)
    if n <= 2:
        return "primary-1to2"
    if n <= 5:
        return "primary-3to5"
    if n <= 8:
        return "middle-6to8"
    return "technical-9to10"


def default_colors(palette_id: str | None = None, class_level: str = "") -> dict[str, str]:
    pid = palette_id or class_level_to_default_palette(class_level)
    p = PALETTES.get(pid) or PALETTES["middle-6to8"]
    return {
        "primary": p["primary"],
        "secondary": p["secondary"],
        "accent": p["accent"],
        "background": p["background"],
        "text": p["text"],
    }


MATH_COLOR_TOKENS = PALETTES["middle-6to8"]
