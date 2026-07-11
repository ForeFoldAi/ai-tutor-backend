"""
Textbook heading parse + query scope (main section vs subsection).

Used at ingest (chunk metadata) and at retrieval time (select all subtopics vs one).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_NUMBERED_HEADING_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\s+(.+)$"
)
_CAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z\s\-/]{5,70}$")

_QUERY_PREFIX_RE = re.compile(
    r"^(?:what\s+is|what\s+are|define|explain|describe|tell\s+me\s+about|list)\s+",
    re.I,
)

_LETTERED_SUBTOPIC_RE = re.compile(
    r"^\s*([a-z])\)\s+(.+)$",
    re.I,
)
_FIG_REF_RE = re.compile(r"Fig\.?\s*(\d+(?:\.\d+)*)", re.I)

# In-text boxes — not the next main section boundary.
_PEDAGOGY_BOX_TITLE_RE = re.compile(
    r"^(think about|let'?s\s+(explore|remember)|in a nutshell|alert\b|no warning|warning\b|watch\b|"
    r"give it a thought|check your learning|don'?t\s+miss)\b",
    re.I,
)

# Captions/snippets that are sidebars, not the main teaching figure for a subtopic.
_SIDEBAR_FIGURE_RE = re.compile(
    r"think\s+about|let'?s\s+(explore|remember)|don'?t\s+miss|acclimatise|illustration\s+—\s+think",
    re.I,
)


def _normalize_apostrophes(text: str) -> str:
    return (text or "").replace("\u2019", "'").replace("\u2018", "'")


def is_pedagogy_box_text(text: str) -> bool:
    """True when text contains a textbook activity / pedagogy box heading."""
    for line in (text or "").splitlines():
        if _PEDAGOGY_BOX_TITLE_RE.match(_normalize_apostrophes(line.strip())):
            return True
    return False


def normalize_title(text: str) -> str:
    """Lowercase title for fuzzy match; strip leading section numbers."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    t = re.sub(r"^[\d.]+\s*", "", t)
    return t.strip(" .:;-")


def parse_heading_line(line: str) -> tuple[str, str, int] | None:
    """
    Parse one heading line.

    Returns (section_number, title, level) or None.
    level: 0 = chapter (``2 Title``), 1 = section (``2.3``), 2+ = subsection.
    """
    line = (line or "").strip()
    if not line or len(line) > 160 or line.endswith("."):
        return None
    m = _NUMBERED_HEADING_RE.match(line)
    if m:
        number = m.group(1)
        title = m.group(2).strip()
        # Skip bare large integers (usually printed page numbers, not section 2.3).
        if "." not in number:
            try:
                if int(number) >= 20:
                    return None
            except ValueError:
                pass
        level = number.count(".")
        return number, title, level
    if _CAPS_HEADING_RE.match(line) and len(line) >= 6:
        return "", line.title(), 1
    words = line.split()
    if 2 <= len(words) <= 12:
        cap = sum(1 for w in words if w and w[0].isupper())
        if cap / len(words) >= 0.65:
            return "", line, 1
    return None


def parse_section_hint(hint: str) -> tuple[str, str, int] | None:
    return parse_heading_line((hint or "").strip())


def query_topic_phrase(query: str) -> str:
    q = (query or "").strip()
    q = _QUERY_PREFIX_RE.sub("", q).strip()
    q = re.sub(r"^(?:the|a|an)\s+", "", q, flags=re.I).strip()
    q = re.sub(r"[?.!]+$", "", q).strip()
    return q


def title_match_score(query: str, title: str) -> float:
    """Higher = better match (0 if no match)."""
    qn = normalize_title(query_topic_phrase(query))
    tn = normalize_title(title)
    if not qn or not tn:
        return 0.0
    q_tokens = set(re.findall(r"[a-z0-9]+", qn))
    t_tokens = set(re.findall(r"[a-z0-9]+", tn))
    if qn == tn:
        return 100.0
    if qn in tn or tn in qn:
        # e.g. query "weather" must not strongly match chapter "Understanding the Weather"
        if len(q_tokens) == 1 and len(t_tokens) >= 2:
            return 52.0
        return 85.0
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
    if overlap >= 0.8 and len(q_tokens) >= 2:
        return 70.0 + overlap * 10
    if overlap >= 1.0 and len(q_tokens) == 1 and len(q_tokens & t_tokens) == 1:
        return 55.0
    return 0.0


@dataclass
class SubtopicInfo:
    """One lettered or titled block under a main section (from textbook text)."""

    title: str
    letter: str = ""
    figure_numbers: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)

    @property
    def normalized_title(self) -> str:
        return normalize_title(self.title)


@dataclass(frozen=True)
class HeadingInfo:
    raw_hint: str
    section_number: str
    title: str
    level: int
    page: int

    @property
    def normalized_title(self) -> str:
        return normalize_title(self.title or self.raw_hint)


@dataclass
class HeadingScope:
    """Resolved scope for a student question."""

    kind: str  # "main_section" | "subsection" | "general"
    matched: HeadingInfo | None = None
    child_headings: list[HeadingInfo] | None = None
    page_start: int = 0
    page_end: int = 10**9

    @property
    def is_main_section(self) -> bool:
        return self.kind == "main_section"

    @property
    def is_subsection(self) -> bool:
        return self.kind == "subsection"


def _page_from_doc(doc: Any) -> int:
    meta = getattr(doc, "metadata", None) or {}
    try:
        return int(meta.get("page", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _hint_from_doc(doc: Any) -> str:
    meta = getattr(doc, "metadata", None) or {}
    return (meta.get("section_hint") or meta.get("section_title") or "").strip()


def extract_headings_from_chunks(chunks: list[Any]) -> list[HeadingInfo]:
    seen: set[str] = set()
    out: list[HeadingInfo] = []

    def _add_hint(hint: str, page: int) -> None:
        if not hint or hint in seen:
            return
        parsed = parse_section_hint(hint)
        if not parsed:
            return
        seen.add(hint)
        number, title, level = parsed
        out.append(
            HeadingInfo(
                raw_hint=hint,
                section_number=number,
                title=title or hint,
                level=level,
                page=page,
            )
        )

    for doc in chunks:
        page = _page_from_doc(doc)
        hint = _hint_from_doc(doc)
        if hint:
            _add_hint(hint, page)
        for raw in (getattr(doc, "page_content", "") or "").strip().split("\n")[:10]:
            line = raw.strip()
            if not line:
                continue
            parsed = parse_heading_line(line)
            if parsed:
                _add_hint(line[:160], page)
    out.sort(key=lambda h: (h.page, h.section_number or "", h.raw_hint))
    return out


def _is_parent_heading(heading: HeadingInfo, all_headings: list[HeadingInfo]) -> bool:
    if heading.section_number:
        prefix = heading.section_number + "."
        return any(
            h.section_number.startswith(prefix) and h.section_number != heading.section_number
            for h in all_headings
            if h.section_number
        )
    # Unnumbered: children are later headings on following pages until next same-level title
    same_page_or_after = [h for h in all_headings if h.page > heading.page]
    for h in same_page_or_after:
        if h.level <= heading.level and h.normalized_title != heading.normalized_title:
            break
        if h.level > heading.level:
            return True
    return False


def _child_headings(parent: HeadingInfo, all_headings: list[HeadingInfo]) -> list[HeadingInfo]:
    if parent.section_number:
        prefix = parent.section_number + "."
        return [
            h
            for h in all_headings
            if h.section_number.startswith(prefix) and h.section_number != parent.section_number
        ]
    children: list[HeadingInfo] = []
    for h in all_headings:
        if h.page <= parent.page:
            continue
        if h.level <= parent.level and h.normalized_title != parent.normalized_title:
            break
        if h.level > parent.level:
            children.append(h)
    return children


def find_lettered_subtopic_match(query: str, chunks: list[Any]) -> SubtopicInfo | None:
    """Match query to an a) b) c) subheading line in chunk text (textbook subtopics)."""
    best: SubtopicInfo | None = None
    best_score = 0.0
    for doc in chunks:
        for line in (getattr(doc, "page_content", "") or "").splitlines():
            m = _LETTERED_SUBTOPIC_RE.match(line.strip())
            if not m:
                continue
            title = re.sub(r"\s+", " ", m.group(2).strip())[:80]
            score = title_match_score(query, title)
            if score > best_score:
                best_score = score
                page = _page_from_doc(doc)
                best = SubtopicInfo(
                    title=title,
                    letter=m.group(1).lower(),
                    pages=[page] if page else [],
                )
    return best if best_score >= 70.0 else None


def resolve_heading_scope(query: str, chunks: list[Any]) -> HeadingScope:
    headings = extract_headings_from_chunks(chunks)
    lettered = find_lettered_subtopic_match(query, chunks)
    if not headings:
        if lettered:
            return HeadingScope(
                kind="subsection",
                matched=HeadingInfo(
                    raw_hint=lettered.title,
                    section_number="",
                    title=lettered.title,
                    level=2,
                    page=lettered.pages[0] if lettered.pages else 0,
                ),
                page_start=lettered.pages[0] if lettered.pages else 0,
                page_end=(lettered.pages[0] + 3) if lettered.pages else 10**9,
            )
        return HeadingScope(kind="general")

    best: HeadingInfo | None = None
    best_score = 0.0
    for h in headings:
        score = title_match_score(query, h.title)
        if h.raw_hint and score < 80:
            score = max(score, title_match_score(query, h.raw_hint) * 0.95)
        if score > best_score:
            best_score = score
            best = h

    if lettered and title_match_score(query, lettered.title) >= max(best_score, 70.0):
        page = lettered.pages[0] if lettered.pages else (best.page if best else 0)
        return HeadingScope(
            kind="subsection",
            matched=HeadingInfo(
                raw_hint=lettered.title,
                section_number="",
                title=lettered.title,
                level=2,
                page=page,
            ),
            page_start=page,
            page_end=page + 4,
        )

    if not best or best_score < 55.0:
        if lettered:
            page = lettered.pages[0] if lettered.pages else 0
            return HeadingScope(
                kind="subsection",
                matched=HeadingInfo(
                    raw_hint=lettered.title,
                    section_number="",
                    title=lettered.title,
                    level=2,
                    page=page,
                ),
                page_start=page,
                page_end=page + 4,
            )
        return HeadingScope(kind="general")

    page_end = 10**9
    for h in headings:
        if h.page <= best.page or h.raw_hint == best.raw_hint:
            continue
        if h.level > best.level:
            continue
        if _PEDAGOGY_BOX_TITLE_RE.match(h.title.strip()):
            continue
        page_end = min(page_end, h.page)
        break

    children = _child_headings(best, headings)
    subtopic_infos = extract_subtopics_from_scope(
        chunks,
        HeadingScope(
            kind="main_section",
            matched=best,
            page_start=best.page,
            page_end=page_end,
        ),
    )
    lettered = [s.title for s in subtopic_infos]
    if subtopic_infos and not children:
        children = [
            HeadingInfo(
                raw_hint=s.title,
                section_number="",
                title=s.title,
                level=2,
                page=s.pages[0] if s.pages else best.page,
            )
            for s in subtopic_infos
        ]
    if children or _is_parent_heading(best, headings) or len(subtopic_infos) >= 2:
        return HeadingScope(
            kind="main_section",
            matched=best,
            child_headings=children,
            page_start=best.page,
            page_end=page_end,
        )

    return HeadingScope(
        kind="subsection",
        matched=best,
        page_start=best.page,
        page_end=page_end,
    )


def extract_subtopics_from_scope(
    chunks: list[Any],
    scope: HeadingScope,
) -> list[SubtopicInfo]:
    """
    Parse a) b) c) … subheadings and their Fig references from textbook chunks in scope.

    Scans all pages in the main section (page_start … page_end), not only chunks
    that repeat the parent heading line.
    """
    if scope.kind != "main_section" or not scope.matched:
        return []

    in_range = [
        d for d in chunks
        if scope.page_start <= _page_from_doc(d) < scope.page_end
    ]
    in_range.sort(key=lambda d: (_page_from_doc(d), (getattr(d, "metadata", None) or {}).get("section_number") or ""))

    by_letter: dict[str, SubtopicInfo] = {}
    current_letter: str = ""

    def _ensure(letter: str, title: str) -> SubtopicInfo:
        if letter not in by_letter:
            by_letter[letter] = SubtopicInfo(title=title, letter=letter)
        else:
            existing = by_letter[letter]
            if len(title) > len(existing.title):
                existing.title = title
        return by_letter[letter]

    for doc in in_range:
        page = _page_from_doc(doc)
        body = getattr(doc, "page_content", "") or ""
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m = _LETTERED_SUBTOPIC_RE.match(stripped)
            if m:
                letter = m.group(1).lower()
                title = re.sub(r"\s+", " ", m.group(2).strip())[:80]
                if len(title) >= 2:
                    current_letter = letter
                    _ensure(letter, title)
            elif current_letter:
                current = by_letter[current_letter]
                for fig in _FIG_REF_RE.findall(stripped):
                    norm = fig.strip()
                    if norm and norm not in current.figure_numbers:
                        current.figure_numbers.append(norm)
        if current_letter:
            current = by_letter[current_letter]
            if page not in current.pages:
                current.pages.append(page)

    return [by_letter[k] for k in sorted(by_letter)]


def extract_lettered_subtopics_from_chunks(
    chunks: list[Any],
    parent: HeadingInfo,
    *,
    page_end: int = 10**9,
) -> list[str]:
    """Backward-compatible title list from lettered subheadings in page range."""
    scope = HeadingScope(
        kind="main_section",
        matched=parent,
        page_start=parent.page,
        page_end=page_end,
    )
    return [s.title for s in extract_subtopics_from_scope(chunks, scope)]


def _number_prefix_match(chunk_number: str, parent_number: str) -> bool:
    if not parent_number:
        return False
    if chunk_number == parent_number:
        return True
    return chunk_number.startswith(parent_number + ".")


def chunk_belongs_to_scope(doc: Any, scope: HeadingScope) -> bool:
    if scope.kind == "general" or not scope.matched:
        return True

    meta = getattr(doc, "metadata", None) or {}
    hint = (meta.get("section_hint") or meta.get("section_title") or "").strip()
    chunk_number = (meta.get("section_number") or "").strip()
    parent_number = (meta.get("parent_section_number") or "").strip()

    parsed = parse_section_hint(hint) if hint else None
    if parsed:
        number, title, _level = parsed
        if not chunk_number and number:
            chunk_number = number

    matched = scope.matched

    if scope.kind == "subsection":
        if chunk_number and matched.section_number:
            return chunk_number == matched.section_number
        if hint and hint == matched.raw_hint:
            return True
        if parsed and title_match_score(matched.title, title) >= 85:
            return True
        return title_match_score(matched.title, hint) >= 70.0

    # main_section: parent + all children
    if chunk_number and matched.section_number:
        if _number_prefix_match(chunk_number, matched.section_number):
            return True
    if parent_number and matched.section_number and parent_number == matched.section_number:
        return True
    if hint:
        if hint == matched.raw_hint:
            return True
        if parsed:
            _num, title, _lvl = parsed
            if title_match_score(matched.title, title) >= 85:
                return True
        for child in scope.child_headings or []:
            if hint == child.raw_hint:
                return True
            if child.section_number and chunk_number.startswith(child.section_number):
                return True
            cp = parse_section_hint(hint)
            if cp and title_match_score(child.title, cp[1]) >= 85:
                return True
    page = _page_from_doc(doc)
    if scope.is_main_section and scope.page_start <= page < scope.page_end:
        return True
    if scope.is_subsection and scope.page_start <= page < scope.page_end:
        if hint == matched.raw_hint or (parsed and title_match_score(matched.title, parsed[1]) >= 85):
            return True
    if title_match_score(matched.title, hint) >= 70.0:
        return True
    body = (getattr(doc, "page_content", "") or "")[:400]
    if title_match_score(matched.title, body) >= 75.0:
        return True
    return False


def select_chunks_for_scope(
    chunks: list[Any],
    scope: HeadingScope,
    *,
    semantic_ranked: list[Any],
    max_chunks: int = 40,
) -> list[Any]:
    """Merge scope-filtered chapter chunks with semantic hits, dedupe, cap."""
    if scope.kind == "general":
        return semantic_ranked[:max_chunks]

    in_scope = [d for d in chunks if chunk_belongs_to_scope(d, scope)]
    if not in_scope:
        return semantic_ranked[:max_chunks]

    def sort_key(d: Any) -> tuple:
        meta = getattr(d, "metadata", None) or {}
        num = (meta.get("section_number") or "")
        return (_page_from_doc(d), num, (meta.get("section_hint") or ""))

    in_scope.sort(key=sort_key)

    seen_ids: set[int] = set()
    merged: list[Any] = []
    for doc in in_scope + semantic_ranked:
        oid = id(doc)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        merged.append(doc)
        if len(merged) >= max_chunks:
            break
    return merged


def enrich_chunks_with_section_metadata(chunks: list[Any]) -> None:
    """
    Tag each chunk with section_number / parent_section_number from the active heading.

    Call after splitting at ingest (PDF/DOCX). Mutates metadata in place.
    """
    current_number = ""
    current_title = ""
    current_level = -1

    def parent_of(number: str) -> str:
        if not number or "." not in number:
            return ""
        return number.rsplit(".", 1)[0]

    for doc in chunks:
        meta = dict(getattr(doc, "metadata", None) or {})
        hint = (meta.get("section_hint") or "").strip()
        if not hint:
            for raw in (getattr(doc, "page_content", "") or "").strip().split("\n")[:6]:
                parsed = parse_heading_line(raw.strip())
                if parsed:
                    hint = raw.strip()[:160]
                    meta["section_hint"] = hint
                    break

        parsed = parse_section_hint(hint) if hint else None
        if parsed:
            number, title, level = parsed
            current_number = number
            current_title = title
            current_level = level
            meta["section_number"] = number
            meta["section_title"] = title
            meta["parent_section_number"] = parent_of(number)
            meta["section_level"] = level
        elif current_number or current_title:
            meta.setdefault("section_number", current_number)
            meta.setdefault("section_title", current_title)
            meta.setdefault("parent_section_number", parent_of(current_number))
            meta.setdefault("section_level", current_level)

        doc.metadata = meta


def subtopics_for_main_section(scope: HeadingScope, chunks: list[Any]) -> list[str]:
    """Ordered subtopic titles under a main-section scope (from textbook lettered subheadings)."""
    return [s.title for s in subtopics_detail_for_main_section(scope, chunks)]


def subtopics_detail_for_main_section(
    scope: HeadingScope,
    chunks: list[Any],
) -> list[SubtopicInfo]:
    """Full subtopic metadata extracted from the textbook."""
    if scope.kind != "main_section" or not scope.matched:
        return []

    extracted = extract_subtopics_from_scope(chunks, scope)
    if len(extracted) >= 2:
        return extracted

    if scope.child_headings and len(scope.child_headings) >= 2:
        return [
            SubtopicInfo(title=h.title, pages=[h.page] if h.page else [])
            for h in scope.child_headings
        ]

    seen: set[str] = set()
    titles: list[str] = []
    for doc in chunks:
        if not chunk_belongs_to_scope(doc, scope):
            continue
        meta = getattr(doc, "metadata", None) or {}
        for hint in (
            (meta.get("section_hint") or "").strip(),
            (meta.get("subsection_title") or "").strip(),
        ):
            if not hint or hint in seen:
                continue
            parsed = parse_section_hint(hint)
            if not parsed:
                continue
            _num, title, level = parsed
            if level >= 2 and title_match_score(scope.matched.title, title) < 85:
                seen.add(hint)
                titles.append(title)
    if titles:
        return [SubtopicInfo(title=t) for t in titles]

    return []


_MAIN_SECTION_FORMAT_TEMPLATE = """\
The student asked about the MAIN section "{main_title}".

You MUST answer with ONE separate block per subtopic (use every subtopic from the textbook context).
Required subtopics (in this order — do not skip any that appear in the context): {subtopic_list}

MANDATORY FORMAT for EACH subtopic (repeat for every item above):
**Exact subtopic title from the list**  (e.g. **Precipitation**, not a different name)
One or two short sentences about that subtopic only (what it measures and which instrument is used).
You may use • bullets instead of sentences if you prefer.

RULES:
- Use the EXACT subtopic title from the textbook as the **bold** heading (do not rename subtopics).
- Do NOT paste figure captions, "Fig. 2.x" lines, or page numbers in your answer — figures are shown separately.
- Do NOT write one long paragraph mixing all subtopics together.
- Do NOT use emoji section headers (no 🌱 📚 🌍 📝 ❓).
- Plain bullet character • only (not numbered lists).
- After all subtopic blocks, add ONE short closing question (optional)."""


def is_sidebar_figure_caption(caption: str | None) -> bool:
    return bool(_SIDEBAR_FIGURE_RE.search(_normalize_apostrophes(caption or "")))


def figure_number_matches(figure_number: str | None, targets: list[str]) -> bool:
    if not figure_number or not targets:
        return False
    fn = figure_number.strip()
    return fn in targets or any(fn.startswith(t + ".") or t.startswith(fn) for t in targets)


def scope_instruction_for_prompt(scope: HeadingScope, *, chunks: list[Any] | None = None) -> str:
    if scope.kind == "general" or not scope.matched:
        return ""
    if scope.is_main_section:
        subtopics = subtopics_for_main_section(scope, chunks or [])
        if subtopics:
            subtopic_list = ", ".join(subtopics[:14])
        else:
            subtopic_list = (
                "(read the chapter context and list every instrument or sub-heading "
                f"under {scope.matched.title})"
            )
        return _MAIN_SECTION_FORMAT_TEMPLATE.format(
            main_title=scope.matched.title,
            subtopic_list=subtopic_list,
        )
    return (
        f'The student asked about the SUBTOPIC "{scope.matched.title}" only. '
        "Explain this specific topic in 2–4 sentences or 2–3 bullet points; "
        "do not summarize the whole chapter or parent section."
    )
