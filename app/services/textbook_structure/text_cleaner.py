"""Text cleaning, watermark removal, and running header/footer sanitization."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.services.textbook_structure.profiler import DocumentProfile

logger = logging.getLogger(__name__)


@dataclass
class TextCleaningStats:
    watermarks_removed: int = 0
    boilerplate_removed: int = 0
    page_numbers_removed: int = 0
    hyphens_repaired: int = 0
    original_char_count: int = 0
    cleaned_char_count: int = 0


class DocumentTextCleaner:
    """Sanitizes extracted PDF text by removing repeated watermarks, headers, footers, and scanning artifacts."""

    @classmethod
    def clean_text(
        cls,
        text: str,
        profile: DocumentProfile | None = None,
        extra_watermarks: list[str] | None = None,
        extra_boilerplate: list[str] | None = None,
    ) -> tuple[str, TextCleaningStats]:
        stats = TextCleaningStats(original_char_count=len(text))
        if not text:
            stats.cleaned_char_count = 0
            return "", stats

        cleaned = text

        # 1. Strip detected full-page watermarks
        watermarks = list(profile.detected_watermarks if profile else [])
        if extra_watermarks:
            watermarks.extend(extra_watermarks)

        # Standard known regional watermark phrases if they match
        watermarks.extend(["SCERT TELANGANA", "TELANGANA SCERT", "NCERT NOT TO BE REPUBLISHED"])
        watermarks = list(dict.fromkeys(watermarks))

        for wm in watermarks:
            if not wm:
                continue
            # Case-insensitive replacement
            pattern = re.compile(re.escape(wm), re.IGNORECASE)
            matches = len(pattern.findall(cleaned))
            if matches:
                stats.watermarks_removed += matches
                cleaned = pattern.sub("", cleaned)

        # 2. Strip repeated headers and footers
        boilerplate = []
        if profile:
            boilerplate.extend(profile.repeated_headers)
            boilerplate.extend(profile.repeated_footers)
        if extra_boilerplate:
            boilerplate.extend(extra_boilerplate)

        # Standard state distribution and copyright phrases
        boilerplate.extend([
            "Free distribution by T.S. Government",
            "Free distribution by Government",
            "Not for sale",
            "All rights reserved",
        ])
        boilerplate = list(dict.fromkeys(boilerplate))

        for bp in boilerplate:
            if not bp or len(bp) < 5:
                continue
            # Regex match with optional trailing year/number
            pattern = re.compile(rf"{re.escape(bp)}[^\n]*", re.IGNORECASE)
            matches = len(pattern.findall(cleaned))
            if matches:
                stats.boilerplate_removed += matches
                cleaned = pattern.sub("", cleaned)

        # 3. Strip isolated margin page numbers at text start/end
        # e.g. leading or trailing lines containing only 1-4 digits
        def _strip_isolated_number_lines(t: str) -> str:
            lines = t.split("\n")
            if not lines:
                return t
            # Check first 2 lines
            idx = 0
            while idx < min(2, len(lines)):
                if lines[idx].strip().isdigit() and len(lines[idx].strip()) <= 4:
                    stats.page_numbers_removed += 1
                    lines.pop(idx)
                else:
                    idx += 1
            # Check last 2 lines
            idx = len(lines) - 1
            while idx >= max(0, len(lines) - 2):
                if lines[idx].strip().isdigit() and len(lines[idx].strip()) <= 4:
                    stats.page_numbers_removed += 1
                    lines.pop(idx)
                else:
                    break
            return "\n".join(lines)

        cleaned = _strip_isolated_number_lines(cleaned)

        # 4. Repair hyphenated line wraps (e.g. 'com- \n puter' -> 'computer')
        hyphen_pattern = re.compile(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)")
        cleaned, num_hyphens = hyphen_pattern.subn(r"\1\2", cleaned)
        stats.hyphens_repaired = num_hyphens

        # 5. Collapse excessive whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        stats.cleaned_char_count = len(cleaned)

        return cleaned, stats
