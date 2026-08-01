"""Phase 2: PPTX theme packs for classroom decks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PptTheme:
    id: str
    label: str
    description: str
    # RGB tuples 0-255
    bg: tuple[int, int, int]
    surface: tuple[int, int, int]
    primary: tuple[int, int, int]
    accent: tuple[int, int, int]
    text: tuple[int, int, int]
    muted: tuple[int, int, int]
    on_primary: tuple[int, int, int]
    font_title: str
    font_body: str


THEMES: dict[str, PptTheme] = {
    "clean_academic": PptTheme(
        id="clean_academic",
        label="Clean Academic",
        description="Navy and white — clear for Class 8–10",
        bg=(248, 250, 252),
        surface=(255, 255, 255),
        primary=(30, 64, 175),
        accent=(14, 165, 233),
        text=(15, 23, 42),
        muted=(71, 85, 105),
        on_primary=(255, 255, 255),
        font_title="Calibri",
        font_body="Calibri",
    ),
    "bright_classroom": PptTheme(
        id="bright_classroom",
        label="Bright Classroom",
        description="Soft color cards — friendly for younger classes",
        bg=(255, 251, 235),
        surface=(255, 255, 255),
        primary=(217, 119, 6),
        accent=(234, 88, 12),
        text=(67, 20, 7),
        muted=(120, 53, 15),
        on_primary=(255, 255, 255),
        font_title="Calibri",
        font_body="Calibri",
    ),
    "stem_focus": PptTheme(
        id="stem_focus",
        label="STEM Focus",
        description="Teal on dark — Math / Science emphasis",
        bg=(15, 23, 42),
        surface=(30, 41, 59),
        primary=(45, 212, 191),
        accent=(56, 189, 248),
        text=(241, 245, 249),
        muted=(148, 163, 184),
        on_primary=(15, 23, 42),
        font_title="Calibri",
        font_body="Calibri",
    ),
    "soft_story": PptTheme(
        id="soft_story",
        label="Soft Story",
        description="Warm neutrals — Languages / Social Studies",
        bg=(250, 245, 240),
        surface=(255, 255, 255),
        primary=(120, 53, 15),
        accent=(180, 83, 9),
        text=(41, 37, 36),
        muted=(87, 83, 78),
        on_primary=(255, 255, 255),
        font_title="Georgia",
        font_body="Calibri",
    ),
}

DEFAULT_THEME_ID = "clean_academic"


def resolve_theme(theme_id: str | None) -> PptTheme:
    key = (theme_id or DEFAULT_THEME_ID).strip().lower()
    return THEMES.get(key) or THEMES[DEFAULT_THEME_ID]


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def list_themes() -> list[dict[str, str]]:
    return [
        {
            "id": t.id,
            "label": t.label,
            "description": t.description,
            "primary": _hex(t.primary),
            "accent": _hex(t.accent),
            "bg": _hex(t.bg),
        }
        for t in THEMES.values()
    ]
