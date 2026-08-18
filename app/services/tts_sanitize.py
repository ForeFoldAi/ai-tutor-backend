"""
Convert LLM / markdown text into natural speech-safe plain text for Edge-TTS.
"""

from __future__ import annotations

import re

# Fenced code blocks (optional language tag)
_CODE_FENCE_RE = re.compile(
    r"```[\w-]*\s*[\r\n]+.*?[\r\n]+```|```[\w-]*\s*.*?```",
    re.DOTALL | re.IGNORECASE,
)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")

# Markdown links: [label](url) and bare URLs
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL_RE = re.compile(
    r"https?://[^\s<>\"']+",
    re.IGNORECASE,
)

# HTML / XML tags
_TAG_RE = re.compile(r"<[^>]+>")

# Horizontal rules
_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)

# Headings at line start
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)

# Blockquote prefix
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)

# Bullet / ordered list line prefixes
_LIST_BULLET_RE = re.compile(r"^\s*[-*+•]\s+", re.MULTILINE)
_LIST_NUM_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)

# Bold / italic (order matters: ** before *)
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_ITALIC_UNDER_RE = re.compile(r"(?<!_)_([^_]+)_(?!_)")

# Stray markdown emphasis runs
_EMPH_RUN_RE = re.compile(r"\*{1,3}|_{1,3}")

# Table pipes (simple rows)
_TABLE_ROW_RE = re.compile(r"^\s*\|?.+\|.+\|?\s*$", re.MULTILINE)

# Textbook figure / page metadata — spoken only if the student asked (stripped here).
_FIG_LINE_RE = re.compile(
    r"^\s*(?:Fig\.?|Figure)\s*\d+(?:\.\d+)*\s*(?:[.:—–-]\s*)?.+$",
    re.I | re.MULTILINE,
)
_PAGE_LINE_RE = re.compile(r"^\s*Page\s+\d+\s*$", re.I | re.MULTILINE)
_FIG_INLINE_RE = re.compile(
    r"\b(?:Fig\.?|Figure)\s*\d+(?:\.\d+)*\b",
    re.I,
)
_PAGE_INLINE_RE = re.compile(r"\bpage\s+\d+\b", re.I)

# Template / JSON-ish noise (only when whole token is brackets)
_BRACKET_ONLY_RE = re.compile(r"^[\s\[\]{}]+$")

_LINK_PHRASE = "You can visit the provided link for more information."
_CODE_PHRASE = "There is a code example here."


def _strip_latex_for_speech(text: str) -> str:
    from app.services.math_speech import math_to_speech

    return math_to_speech(text)


def _replace_code_blocks(text: str) -> str:
    if "```" in text:
        text = _CODE_FENCE_RE.sub(_CODE_PHRASE, text)
    return text


def _replace_inline_code(text: str) -> str:
    return _INLINE_CODE_RE.sub(r"\1", text)


def _replace_links_and_urls(text: str) -> str:
    text = _MD_LINK_RE.sub(r"\1", text)
    if _URL_RE.search(text):
        text = _URL_RE.sub(_LINK_PHRASE, text)
    return text


def _strip_markdown_structure(text: str) -> str:
    text = _TAG_RE.sub(" ", text)
    text = _HR_RE.sub(" ", text)
    text = _HEADING_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _LIST_BULLET_RE.sub("", text)
    text = _LIST_NUM_RE.sub("", text)
    text = _TABLE_ROW_RE.sub(lambda m: m.group(0).replace("|", " ").strip(), text)
    return text


def _strip_emphasis(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _BOLD_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)
        text = _ITALIC_STAR_RE.sub(r"\1", text)
        text = _ITALIC_UNDER_RE.sub(r"\1", text)
    text = _EMPH_RUN_RE.sub("", text)
    return text


def _lists_to_sentences(text: str) -> str:
    """Turn line-broken list items into period-separated phrases."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return text
    if sum(1 for ln in lines if len(ln) < 120) >= 2:
        parts: list[str] = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            if ln[-1] not in ".?!":
                ln += "."
            parts.append(ln)
        return " ".join(parts)
    return text


def _normalize_whitespace_and_punctuation(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of punctuation (keep one)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"\.{2,}(?!\.)", ".", text)
    # Streaming LLM chunks often flush lone markdown symbols — drop them
    text = re.sub(r"[*#_`\\$]+", " ", text)
    # Remove isolated bracket-only fragments from JSON/templates
    text = re.sub(r"\s*[\[\]{}]\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_figure_page_metadata(text: str) -> str:
    t = _FIG_LINE_RE.sub(" ", text)
    t = _PAGE_LINE_RE.sub(" ", t)
    t = _FIG_INLINE_RE.sub(" ", t)
    t = _PAGE_INLINE_RE.sub(" ", t)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,+", ",", t)
    return t


def sanitize_for_tts(text: str) -> str:
    """
    Strip markdown / markup and return natural spoken plain text.

    Safe to call on streaming buffers or full sentences before Edge-TTS.
    """
    if not text or not text.strip():
        return ""

    t = text.strip()
    t = _replace_code_blocks(t)
    t = _replace_inline_code(t)
    t = _replace_links_and_urls(t)
    t = _strip_figure_page_metadata(t)
    t = _strip_markdown_structure(t)
    t = _strip_emphasis(t)
    t = _strip_latex_for_speech(t)
    t = _lists_to_sentences(t)
    t = _normalize_whitespace_and_punctuation(t)

    # Drop lines that are only symbols
    t = " ".join(
        part
        for part in re.split(r"\n+", t)
        if part.strip() and not _BRACKET_ONLY_RE.match(part.strip())
    )

    return t.strip()


def sanitize_chunk_for_tts(text: str) -> str:
    """Sanitize a single speakable chunk; return empty if nothing left to say."""
    cleaned = sanitize_for_tts(text)
    if not cleaned:
        return ""
    # Skip streaming fragments that are mostly punctuation / symbols
    letters = re.sub(r"[\W\s\d_]", "", cleaned)
    if len(letters) < 2:
        return ""
    if re.fullmatch(r"[\s\W]+", cleaned):
        return ""
    return cleaned
