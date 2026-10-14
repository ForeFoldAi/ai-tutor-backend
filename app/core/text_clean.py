"""Strip invisible/control characters that show up as □ in browsers (e.g. \\b from Word paste)."""

from __future__ import annotations

import re

# C0 controls except tab/LF/CR; DEL; zero-width / BOM
_INVISIBLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]")


def clean_display_label(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _INVISIBLE_RE.sub("", str(value)).strip()
    return cleaned


# ponytail: fails if \\b leak into titles again
assert clean_display_label("Unit 1 - Wit and Wisdom\b") == "Unit 1 - Wit and Wisdom"
assert clean_display_label(None) is None
