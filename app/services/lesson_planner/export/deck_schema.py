"""Normalize PPT outline into a structured classroom deck."""

from __future__ import annotations

import re
from typing import Any

LAYOUTS = frozenset(
    {
        "title",
        "section",
        "bullets",
        "two_column",
        "steps",
        "big_idea",
        "summary",
        "figure",
    }
)

_LAYOUT_HINTS = {
    "title": "title",
    "section": "section",
    "divider": "section",
    "bullet": "bullets",
    "bullets": "bullets",
    "content": "bullets",
    "two": "two_column",
    "two_column": "two_column",
    "two-column": "two_column",
    "columns": "two_column",
    "step": "steps",
    "steps": "steps",
    "big": "big_idea",
    "big_idea": "big_idea",
    "idea": "big_idea",
    "summary": "summary",
    "recap": "summary",
    "takeaway": "summary",
    "figure": "figure",
    "diagram": "figure",
    "image": "figure",
}

_GENERIC_TITLES = frozenset(
    {
        "title slide",
        "title",
        "welcome",
        "welcome slide",
        "introduction",
        "intro",
        "opening",
        "opening slide",
        "start",
        "lesson title",
        "presentation title",
        "today's lesson",
        "today",
    }
)

_MD_NOISE = re.compile(r"[*_`#]+")


def _strip_md(text: str) -> str:
    cleaned = _MD_NOISE.sub("", str(text or ""))
    return re.sub(r"\s+", " ", cleaned).strip(" -\t")


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _strip_md(text).lower())


def _is_title_echo(bullet: str, title: str) -> bool:
    """True when a bullet is just the slide title repeated (the empty-slide bug)."""
    b, t = _norm_key(bullet), _norm_key(title)
    if not b or not t:
        return False
    if b == t:
        return True
    # "Summary Checklist" as both title and only bullet
    if b in {"summarychecklist", "checklist"} and "summary" in t:
        return True
    if t.startswith(b) or b.startswith(t):
        return len(b) >= max(8, int(len(t) * 0.7))
    return False


def _bullets_from_notes(notes: str, *, limit: int = 4) -> list[str]:
    text = _strip_md(notes)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s+", text)
    out: list[str] = []
    for part in parts:
        line = _strip_md(part)
        if len(line.split()) < 3:
            continue
        words = line.split()
        if len(words) > 14:
            line = " ".join(words[:14]).rstrip(",.;:") + "…"
        out.append(line)
        if len(out) >= limit:
            break
    return out


def _repair_empty_content(title: str, layout: str, bullets: list[str], *, notes: str, callout: str, chapter: str) -> tuple[str, list[str]]:
    """Fill or reframe slides the LLM left blank (steps/summary/practice)."""
    real = [b for b in bullets if not _is_title_echo(b, title)]
    title_l = title.lower()

    if layout == "summary":
        real = [b for b in real if _norm_key(b) not in {"summarychecklist", "checklist", "summary"}]
        if len(real) >= 2:
            return layout, real
        from_notes = _bullets_from_notes(notes, limit=3)
        if len(from_notes) >= 2:
            return layout, from_notes
        if callout:
            return layout, [callout, f"Review {chapter} once more at home", "Ask one question you still have"]
        return layout, [
            f"Key ideas from {chapter}",
            "One tool or method we used today",
            "One question still open",
        ]

    if layout == "steps" or any(w in title_l for w in ("how a ", "how to ", "steps", "match the", "practice")):
        if len(real) >= 2:
            return ("steps" if layout == "steps" or "how" in title_l else layout), real
        from_notes = _bullets_from_notes(notes, limit=4)
        if len(from_notes) >= 2:
            return "steps", from_notes
        if "match" in title_l or "practice" in title_l:
            if real:
                extra = [
                    "Match each tool to what it measures",
                    "Check one answer with a partner",
                    "Correct any mismatch together",
                ]
                merged = []
                for line in [*real, *extra]:
                    if line not in merged:
                        merged.append(line)
                    if len(merged) >= 4:
                        break
                return "bullets", merged
            return "bullets", [
                "Read each instrument name carefully",
                "Match it to what it measures",
                "Check one answer with a partner",
                "Correct any mismatch together",
            ]
        if "thermometer" in title_l:
            return "steps", [
                "Liquid expands when the air is warmer",
                "The liquid rises inside the narrow tube",
                "Read the scale at eye level",
                "Note the temperature unit (°C or °F)",
            ]
        if "how" in title_l:
            return "steps", [
                "Start with what you observe",
                "Connect it to the rule or definition",
                "Check with a quick classroom example",
            ]
        if callout:
            return "bullets", [callout, f"Apply this idea to {chapter}", "Explain it in your own words"]
        # Still empty — caller should drop.
        return layout, real

    if len(real) < 1 and (notes or callout):
        recovered = _bullets_from_notes(notes, limit=3) or ([callout] if callout else [])
        return layout, recovered
    return layout, real


def _guess_layout(title: str, index: int, total: int) -> str:
    t = (title or "").lower()
    if index == 0 or any(w in t for w in ("welcome", "title slide", "today we", "let's learn")):
        return "title"
    if index == total - 1 or any(w in t for w in ("thank", "goodbye", "exit ticket")):
        return "summary"
    if any(w in t for w in ("objective", "goal", "what we will", "learning outcome")):
        return "bullets"
    if any(w in t for w in ("summary", "recap", "key takeaway", "remember", "checklist")):
        return "summary"
    if any(w in t for w in ("step", "how to", "procedure", "method", "match the")):
        return "steps"
    if any(w in t for w in ("example", "vs", "compare", "concept vs", "case study")):
        return "two_column"
    if any(w in t for w in ("big idea", "main idea", "remember this", "key idea", "big question")):
        return "big_idea"
    if any(w in t for w in ("part ", "section", "unit")):
        return "section"
    return "bullets"


def _norm_layout(raw: str | None, *, title: str, index: int, total: int) -> str:
    key = (raw or "").strip().lower().replace(" ", "_")
    if key in LAYOUTS:
        return key
    for token, layout in _LAYOUT_HINTS.items():
        if token in key:
            return layout
    return _guess_layout(title, index, total)


def _clip_bullets(bullets: list[str], *, limit: int = 5) -> list[str]:
    out: list[str] = []
    for b in bullets:
        text = _strip_md(b)
        if not text:
            continue
        words = text.split()
        if len(words) > 14:
            text = " ".join(words[:14]).rstrip(",.;:") + "…"
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _is_generic_title(title: str) -> bool:
    key = _strip_md(title).lower().rstrip("!")
    return key in _GENERIC_TITLES or key.startswith("title slide")


def _content_score(slide: dict[str, Any]) -> int:
    layout = slide.get("layout") or "bullets"
    bullets = slide.get("bullets") or []
    right = slide.get("right_bullets") or []
    callout = 1 if slide.get("callout") else 0
    if layout == "title":
        return 100
    if layout == "summary":
        return 40 + len(bullets) * 3 + callout
    if layout == "section" and not bullets:
        return 5  # weak divider
    if layout == "steps" and len(bullets) < 2:
        return 8
    if layout == "two_column":
        return 20 + len(bullets) * 4 + len(right) * 4 + callout
    if layout == "figure":
        return 22 + len(bullets) * 3
    return 12 + len(bullets) * 4 + callout


def _has_real_content(slide: dict[str, Any]) -> bool:
    layout = slide.get("layout") or ""
    title = _strip_md(str(slide.get("title") or ""))
    bullets = [b for b in (slide.get("bullets") or []) if _strip_md(b)]
    right = [b for b in (slide.get("right_bullets") or []) if _strip_md(b)]
    if layout == "title":
        return bool(title)
    if layout == "section":
        return bool(title)  # dividers can be title-only
    if layout == "steps":
        return len(bullets) >= 2
    if layout == "two_column":
        return bool(bullets or right)
    if layout == "summary":
        # Reject "✓ Summary Checklist" style duplicates of the title.
        real = [b for b in bullets if _strip_md(b).lower() not in {title.lower(), f"✓ {title}".lower()}]
        return len(real) >= 2
    if layout == "big_idea":
        return bool(bullets) and _strip_md(bullets[0]).lower() != title.lower()
    return len(bullets) >= 1


def polish_deck_slides(
    slides: list[dict[str, Any]],
    *,
    chapter: str | None = None,
    subject: str | None = None,
    max_slides: int | None = None,
) -> list[dict[str, Any]]:
    """Fix generic titles, drop empty slides, enforce slide budget."""
    chapter = _strip_md(chapter or "") or "Lesson"
    subject = _strip_md(subject or "")
    out: list[dict[str, Any]] = []

    for idx, raw in enumerate(slides):
        item = dict(raw)
        title = _strip_md(str(item.get("title") or f"Slide {idx + 1}"))
        layout = item.get("layout") or "bullets"
        bullets = list(item.get("bullets") or [])
        right = list(item.get("right_bullets") or [])

        # Opening slide: use chapter name, never "Title Slide".
        if idx == 0:
            title = chapter if (_is_generic_title(title) or layout == "title" or not title) else title
            if _is_generic_title(title):
                title = chapter
            layout = "title"
            bullets = []
            item["side_heading"] = item.get("side_heading") or subject or "Classroom Lesson"
            item["callout"] = ""
        elif layout == "title":
            # Misplaced title layouts mid-deck.
            if any(w in title.lower() for w in ("thank", "goodbye")):
                layout = "summary"
                title = "Thank You"
                if not bullets:
                    bullets = [
                        f"Today we learned about {chapter}",
                        "Share one new idea with a partner",
                        "See you next class!",
                    ]
            else:
                layout = "bullets" if bullets else "section"

        # Strip title-echo bullets ("How a Thermometer Works" as the only step).
        bullets = [b for b in bullets if not _is_title_echo(b, title)]
        right = [b for b in right if not _is_title_echo(b, title)]

        notes = str(item.get("speaker_notes") or "")
        callout = _strip_md(str(item.get("callout") or ""))
        if layout not in ("title", "section"):
            layout, bullets = _repair_empty_content(
                title,
                layout,
                bullets,
                notes=notes,
                callout=callout,
                chapter=chapter,
            )

        # Steps with a single real line → bullets card (one numbered card looks empty).
        if layout == "steps" and len(bullets) == 1:
            layout = "bullets"

        item["title"] = title
        item["layout"] = layout
        item["bullets"] = bullets
        item["right_bullets"] = right
        item["callout"] = callout.lstrip("*").strip()
        item["side_heading"] = _strip_md(str(item.get("side_heading") or ""))
        item["speaker_notes"] = _strip_md(notes)

        if not _has_real_content(item) and layout not in ("title", "section"):
            continue
        # Drop empty section dividers when they only restate a nearby idea with no value.
        if layout == "section" and not bullets and len(title.split()) <= 2:
            continue
        out.append(item)

    if not out:
        out = [
            {
                "number": 1,
                "title": chapter,
                "layout": "title",
                "side_heading": subject or "Classroom Lesson",
                "icon": "engage",
                "figure_file": "",
                "figure_caption": "",
                "bullets": [],
                "right_bullets": [],
                "callout": "",
                "speaker_notes": "",
            }
        ]

    # Ensure a single opening title.
    if out[0].get("layout") != "title":
        out.insert(
            0,
            {
                "number": 1,
                "title": chapter,
                "layout": "title",
                "side_heading": subject or "Classroom Lesson",
                "icon": "engage",
                "figure_file": "",
                "figure_caption": "",
                "bullets": [],
                "right_bullets": [],
                "callout": "",
                "speaker_notes": "",
            },
        )
    else:
        out[0]["title"] = chapter if _is_generic_title(out[0].get("title") or "") else out[0]["title"]
        if _is_generic_title(out[0].get("title") or "") or out[0].get("layout") == "title":
            out[0]["title"] = chapter

    # Enforce budget: keep title + strongest middle + closing summary.
    target = int(max_slides or 0)
    if target > 0 and len(out) > target:
        first = out[0]
        rest = out[1:]
        closing = None
        middle = []
        # Prefer an explicit thank-you / final summary as the closing slide.
        for s in rest:
            title_l = str(s.get("title") or "").lower()
            if "thank" in title_l or "goodbye" in title_l:
                if closing is not None:
                    middle.append(closing)
                closing = s
            elif closing is None and s.get("layout") == "summary":
                closing = s
            else:
                middle.append(s)
        if closing is None and middle:
            closing = middle.pop()
            closing = {
                **closing,
                "layout": "summary",
                "title": "What We Learned",
                "bullets": (closing.get("bullets") or [chapter])[:4],
            }
        slots = max(target - (2 if closing else 1), 0)
        middle_sorted = sorted(middle, key=_content_score, reverse=True)
        keep_middle = sorted(middle_sorted[:slots], key=lambda s: int(s.get("number") or 0))
        out = [first, *keep_middle]
        if closing and target >= 2:
            out.append(closing)
        out = out[:target]

    for i, s in enumerate(out):
        s["number"] = i + 1
    return out


def ppt_slides_from_markdown(
    md: str,
    *,
    chapter: str | None = None,
    subject: str | None = None,
    max_slides: int | None = None,
) -> list[dict[str, Any]]:
    """Parse ### Slide N: Title blocks into structured deck slides."""
    slides: list[dict[str, Any]] = []
    chunks = re.split(r"(?=^### Slide\s+\d+:)", md or "", flags=re.MULTILINE)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk.startswith("### Slide"):
            continue
        title_match = re.match(r"^### Slide\s+(\d+):\s*(.+)$", chunk, re.MULTILINE)
        number = int(title_match.group(1)) if title_match else len(slides) + 1
        title = title_match.group(2).strip() if title_match else "Slide"

        layout_raw = ""
        side_heading = ""
        callout = ""
        notes = ""
        icon = ""
        figure_file = ""
        figure_caption = ""
        bullets: list[str] = []
        right_bullets: list[str] = []
        in_content = False
        in_right = False

        for line in chunk.splitlines()[1:]:
            stripped = line.strip()
            if stripped.startswith("**Layout:**"):
                layout_raw = stripped.replace("**Layout:**", "", 1).strip()
                in_content = False
                in_right = False
                continue
            if stripped.startswith("**Side Heading:**"):
                side_heading = stripped.replace("**Side Heading:**", "", 1).strip()
                in_content = False
                in_right = False
                continue
            if stripped.startswith("**Icon:**"):
                icon = stripped.replace("**Icon:**", "", 1).strip()
                in_content = False
                in_right = False
                continue
            if stripped.startswith("**Figure:**"):
                figure_raw = stripped.replace("**Figure:**", "", 1).strip()
                if "—" in figure_raw:
                    left, right = figure_raw.split("—", 1)
                elif " - " in figure_raw:
                    left, right = figure_raw.split(" - ", 1)
                else:
                    left, right = figure_raw, ""
                left, right = left.strip(), right.strip()
                if left.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                    figure_file = left
                    figure_caption = right
                else:
                    figure_caption = figure_raw
                in_content = False
                in_right = False
                continue
            if stripped.startswith("**Callout:**") or stripped.startswith("**Key Takeaways:**"):
                callout = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
                in_content = False
                in_right = False
                continue
            if stripped.startswith("**Speaker Notes:**"):
                notes = stripped.replace("**Speaker Notes:**", "", 1).strip()
                in_content = False
                in_right = False
                continue
            # LLM often parks the real activity here and leaves Slide Content empty.
            if stripped.startswith("**Interactive Element:**") or stripped.startswith("**Activity:**"):
                activity = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
                if activity:
                    bullets.append(activity)
                in_content = False
                in_right = False
                continue
            if stripped.startswith("**Slide Content:**") or stripped.startswith("**Left:**"):
                in_content = True
                in_right = False
                continue
            if stripped.startswith("**Right:**") or stripped.startswith("**Example:**"):
                in_content = False
                in_right = True
                continue
            if stripped.startswith("**Visual Suggestions:**") or stripped.startswith("**Animation Suggestions:**"):
                in_content = False
                in_right = False
                continue
            if stripped.startswith("**") and not stripped.startswith("- "):
                in_content = False
                in_right = False
                continue
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if in_right:
                    right_bullets.append(item)
                elif in_content or not (in_content or in_right):
                    bullets.append(item)

        slides.append(
            {
                "number": number,
                "title": title,
                "layout": layout_raw,
                "side_heading": side_heading,
                "icon": icon,
                "figure_file": figure_file,
                "figure_caption": figure_caption,
                "bullets": bullets,
                "right_bullets": right_bullets,
                "callout": callout,
                "speaker_notes": notes,
            }
        )

    return normalize_deck_slides(
        slides,
        chapter=chapter,
        subject=subject,
        max_slides=max_slides,
    )


def normalize_deck_slides(
    raw_slides: list[dict[str, Any]] | None,
    *,
    chapter: str | None = None,
    subject: str | None = None,
    max_slides: int | None = None,
) -> list[dict[str, Any]]:
    """Normalize markdown/structured slides into renderer-ready deck slides."""
    from app.services.lesson_planner.export.figure_assets import normalize_icon

    src = [s for s in (raw_slides or []) if isinstance(s, dict)]
    total = max(len(src), 1)
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(src):
        title = _strip_md(str(item.get("title") or f"Slide {idx + 1}"))
        bullets = _clip_bullets(
            [str(b) for b in (item.get("bullets") or item.get("body") or []) if str(b).strip()]
        )
        # Never keep a bullet that only repeats the slide title.
        bullets = [b for b in bullets if not _is_title_echo(b, title)]
        right = _clip_bullets(
            [str(b) for b in (item.get("right_bullets") or item.get("right") or []) if str(b).strip()],
            limit=4,
        )
        right = [b for b in right if not _is_title_echo(b, title)]
        layout = _norm_layout(
            item.get("layout"),
            title=title,
            index=idx,
            total=total,
        )
        if layout == "two_column" and not right and len(bullets) >= 4:
            mid = len(bullets) // 2
            right = bullets[mid:]
            bullets = bullets[:mid]
        if layout == "steps" and len(bullets) >= 2:
            bullets = [f"{i + 1}. {b.lstrip('0123456789.-) ')}" for i, b in enumerate(bullets[:4])]
        elif layout == "steps" and len(bullets) < 2:
            # Leave unrepaired for polish to recover or drop — don't fake a 1-card step.
            pass
        out.append(
            {
                "number": int(item.get("number") or idx + 1),
                "title": title,
                "layout": layout,
                "side_heading": _strip_md(str(item.get("side_heading") or item.get("eyebrow") or ""))[:80],
                "icon": normalize_icon(str(item.get("icon") or "")),
                "figure_file": str(item.get("figure_file") or "").strip()[:120],
                "figure_caption": _strip_md(str(item.get("figure_caption") or ""))[:200],
                "bullets": bullets,
                "right_bullets": right,
                "callout": _strip_md(str(item.get("callout") or ""))[:160],
                "speaker_notes": _strip_md(str(item.get("speaker_notes") or item.get("notes") or "")),
            }
        )
    return polish_deck_slides(
        out,
        chapter=chapter,
        subject=subject,
        max_slides=max_slides,
    )


def deck_from_ppt_artifact(ppt: dict[str, Any] | None, *, chapter: str | None = None, subject: str | None = None) -> list[dict[str, Any]]:
    ppt = ppt or {}
    max_slides = ppt.get("slide_count_target")
    try:
        max_slides = int(max_slides) if max_slides is not None else None
    except (TypeError, ValueError):
        max_slides = None
    chapter = chapter or str(ppt.get("chapter_name") or "") or None
    subject = subject or str(ppt.get("subject") or "") or None

    if ppt.get("format") == "markdown" and ppt.get("markdown"):
        slides = ppt_slides_from_markdown(
            str(ppt["markdown"]),
            chapter=chapter,
            subject=subject,
            max_slides=max_slides,
        )
        if slides:
            return slides
    if isinstance(ppt.get("slides"), list) and ppt["slides"]:
        return normalize_deck_slides(
            ppt["slides"],
            chapter=chapter,
            subject=subject,
            max_slides=max_slides,
        )
    return []
