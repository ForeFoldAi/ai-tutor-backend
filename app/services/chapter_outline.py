"""Lesson outline of one chapter: its real readings / main sections, in book order.

Numbered books (1.1, 1.2 …) are handled by topic_progress's section-number path.
Everything else — English readers (story / poem / play), primary EVS, most State
Board books — has no numbers, and its layout labels are unreliable (running
watermarks like "Government's Gift for Students' Progress", "# QUESTIONS",
"READ AND FIND OUT"). The old fallback kept only titles containing words such as
"resources", so a Class 8 English unit came out empty and the tutor improvised a
chapter overview from whatever chunks the search returned (exercises, teacher
transcripts).

Here a compact page-tagged skeleton of heading-like lines is built from the
chapter's own indexed text, and one small LLM call picks the lesson topics from
it — every title must be copied from the skeleton and is checked against it, so
nothing can be invented. The result is cached in document storage (one call per
chapter, ever) and recomputed only when the chapter is re-indexed.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

OUTLINE_VERSION = 2
_CACHE_PREFIX = "_outlines"
_MAX_SKELETON_LINES = 260
_MAX_SKELETON_CHARS = 9000
# A failed LLM call is retried after this long, not on every turn.
_FAIL_RETRY_SEC = 600.0

_mem: dict[str, tuple[float, dict | None]] = {}
_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()

_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s*")
_SENTENCE_END_RE = re.compile(r"[.,;:]\s*$")
_PAGE_FURNITURE_RE = re.compile(
    r"^(?:reprint\s+\d{4}|page\s*\d+|\d{1,3}|www\.|https?://|©)", re.I
)


def _norm(s: str) -> str:
    s = (s or "").replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def _squash(s: str) -> str:
    """Compare titles ignoring case, spacing and punctuation ("Witthat" == "Wit that")."""
    return re.sub(r"[^a-z0-9]", "", _norm(s).lower())


def _page(doc: Any) -> int:
    try:
        return int((getattr(doc, "metadata", None) or {}).get("page", 0) or 0)
    except (TypeError, ValueError):
        return 0


def fingerprint(chunks: list[Any]) -> str:
    """Changes whenever the chapter is re-indexed (chunk count or page span)."""
    pages = [_page(d) for d in chunks] or [0]
    chars = sum(len(getattr(d, "page_content", "") or "") for d in chunks)
    return f"{len(chunks)}:{min(pages)}-{max(pages)}:{chars}"


_REPEAT_RE = re.compile(r"^(.+?)(?:\s+\1)+(?:\s+(.*))?$")


def _collapse_repeats(text: str) -> str:
    """Decorative titles are extracted several times: "Dust of Sno Dust of Sno Dust of Snow"."""
    m = _REPEAT_RE.match(text)
    if not m or len(m.group(1)) < 3:
        return text
    unit, rest = m.group(1).strip(), (m.group(2) or "").strip()
    if rest and _squash(rest).startswith(_squash(unit)):
        return rest
    return unit if not rest else text


def _candidate_lines(chunks: list[Any]) -> list[tuple[int, str]]:
    """(page, line) for short heading-like lines, page furniture removed."""
    raw: list[tuple[int, str, bool]] = []
    for doc in sorted(chunks, key=_page):
        page = _page(doc)
        for line in (getattr(doc, "page_content", "") or "").splitlines():
            is_md = bool(_MD_HEADING_RE.match(line))
            text = _collapse_repeats(_norm(_MD_HEADING_RE.sub("", line)))
            if not text or len(text) < 3 or len(text) > 90:
                continue
            if _PAGE_FURNITURE_RE.match(text):
                continue
            words = text.split()
            if not is_md:
                # Body sentences wrap into short lines too; keep title-shaped ones.
                if len(words) > 12:
                    continue
                # Sentences are body text — except short activity cues ("III Read and
                # enjoy the poem.") that tell the model a piece is an extra.
                if _SENTENCE_END_RE.search(text) and len(words) > 6:
                    continue
                if not (text[0].isupper() or text[0].isdigit()):
                    continue
            raw.append((page, text, is_md))

    # Running headers / watermarks repeat on many pages.
    pages_by_line: dict[str, set[int]] = defaultdict(set)
    for page, text, _ in raw:
        pages_by_line[_squash(text)].add(page)
    n_pages = len({p for p, _, _ in raw}) or 1
    repeated = {k for k, ps in pages_by_line.items() if len(ps) >= 3 and len(ps) >= 0.2 * n_pages}

    out: list[tuple[int, str]] = []
    last = ""
    for page, text, _ in raw:
        k = _squash(text)
        if not k or k in repeated or k == last:
            continue
        last = k
        out.append((page, text))
    return out


def build_skeleton(chunks: list[Any]) -> tuple[str, list[tuple[int, str]]]:
    lines = _candidate_lines(chunks)
    if len(lines) > _MAX_SKELETON_LINES:
        # Keep the shortest (most title-like) lines, in book order.
        keep = sorted(range(len(lines)), key=lambda i: len(lines[i][1].split()))[:_MAX_SKELETON_LINES]
        lines = [lines[i] for i in sorted(keep)]
    rendered: list[str] = []
    size = 0
    for page, text in lines:
        row = f"p{page}: {text}"
        if size + len(row) + 1 > _MAX_SKELETON_CHARS:
            break
        rendered.append(row)
        size += len(row) + 1
    return "\n".join(rendered), lines


_SYSTEM = """You find the LESSON STRUCTURE of one school textbook chapter.
You get candidate heading lines from the chapter, each tagged with its page (pN:).
Return ONLY JSON:
{"topics":[{"title":"...","page":N,"text_end_page":N,"subtopics":[{"title":"...","page":N}]}],"end_matter_page":N or null}

Rules:
- topics = what a student must learn, in book order: the readings (stories, poems, plays, letters, speeches) in language books; the main concept sections in other subjects.
- A poem, play or second story printed after the main reading is its own topic — include it.
- But a short piece inside an activity section (right after "Let us explore / write / speak / listen", "Read and enjoy") is an extra, not a topic.
- 1 to 10 topics. Prefer fewer, real topics over many fragments.
- Copy every title exactly from a candidate line (you may only fix broken spacing, e.g. "Witthat" -> "Wit that"). Never invent or rephrase.
- If the chapter's main reading is clearly the CHAPTER title itself but its title line is missing (printed as a picture), you may use the CHAPTER title (without its number) as the first topic, with the first page.
- EXCLUDE: activity prompts (Let us read / discuss / listen / speak / write, Do these, Try this, Think and answer, Before you read, Read and find out), exercises, questions, comprehension, vocabulary, grammar, glossary, keywords, about the author, project work, answers, transcripts, teacher notes, figure/table captions, character lists, speaker names, running headers, the chapter or unit title itself.
- subtopics: only clear parts of one topic (Scene I / Scene II, Part A / Part B, numbered sub-sections). Up to 6. Otherwise [].
- text_end_page: last page of the topic's own text (the reading / explanation), before its exercises or activities begin (e.g. the page before "Let us discuss", "Questions", "Exercises"). Use the topic's page if unsure.
- end_matter_page: first page from which only end material remains (transcripts, answers, teacher notes, appendix); null if none."""


def _llm_pick(skeleton: str, *, chapter_name: str, subject_name: str, class_level: str) -> dict | None:
    from app.config import INTERACTIVE_PASS2_TIMEOUT_SEC
    from app.services.llm_client import complete_sync

    user = (
        f"SUBJECT: {subject_name}\nCLASS: {class_level}\nCHAPTER: {chapter_name}\n\n"
        f"CANDIDATE LINES:\n{skeleton}"
    )
    raw = complete_sync(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        # Own route: defaults to the primary provider (LLM_OUTLINE_PROVIDER overrides).
        feature="outline",
        max_tokens=900,
        timeout=max(20.0, float(INTERACTIVE_PASS2_TIMEOUT_SEC)),
    )
    text = raw or ""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _match(title: str, lines: list[tuple[int, str]]) -> tuple[int, str] | None:
    """The candidate line this title was copied from (spacing-insensitive)."""
    key = _squash(title)
    if len(key) < 3:
        return None
    best: tuple[float, tuple[int, str]] | None = None
    for page, text in lines:
        k = _squash(text)
        if not k:
            continue
        if k == key:
            return page, text
        ratio = SequenceMatcher(None, key, k).ratio()
        # A title wrapped over two lines matches the start of the joined text.
        if (k.startswith(key) and len(key) >= 0.6 * len(k)) or (
            key.startswith(k) and len(k) >= 0.6 * len(key)
        ):
            ratio = max(ratio, 0.9)
        if ratio >= 0.88 and (best is None or ratio > best[0]):
            best = (ratio, (page, text))
    return best[1] if best else None


def _title_shaped(title: str) -> bool:
    """Headings, not body sentences picked by mistake."""
    words = title.split()
    if len(title) < 3 or len(words) > 12:
        return False
    # "Vitamins are organic substances." is a sentence; "Why is power sharing desirable?" is fine.
    if re.search(r"[.;,]\s*$", title):
        return False
    # Two sentences ("Vitamins are organic substances. They are micro nutrients").
    return not re.search(r"[a-z]\.\s+[A-Z]", title)


def validate_outline(
    data: dict | None, lines: list[tuple[int, str]], *, chapter_name: str = ""
) -> dict | None:
    """Keep only titles that really occur in the chapter text; drop the chapter title."""
    if not isinstance(data, dict):
        return None
    chapter_body = _norm(
        re.sub(r"^\s*(?:chapter|unit)?\s*[\dIVX]+\s*[:.\-–]?\s*", "", chapter_name or "", flags=re.I)
    )
    chapter_key = _squash(chapter_body)
    first_page = min((p for p, _ in lines), default=0)
    topics: list[dict] = []
    seen: set[str] = set()
    for t in data.get("topics") or []:
        if not isinstance(t, dict):
            continue
        title = _norm(str(t.get("title") or ""))
        if not _title_shaped(title):
            continue
        key = _squash(title)
        is_chapter_title = bool(chapter_key) and key == chapter_key
        # A single-reading chapter whose title is a picture: its title is the chapter name.
        hit = (first_page, chapter_body) if is_chapter_title else _match(title, lines)
        if not hit:
            continue
        if key in seen:
            continue
        seen.add(key)
        subs: list[dict] = []
        for s in t.get("subtopics") or []:
            if not isinstance(s, dict):
                continue
            st = _norm(str(s.get("title") or ""))
            sh = _match(st, lines)
            if sh and _squash(st) not in seen:
                seen.add(_squash(st))
                subs.append({"title": st, "page": sh[0]})
        try:
            text_end = int(t.get("text_end_page"))
        except (TypeError, ValueError):
            text_end = None
        topics.append({"title": title, "page": hit[0], "text_end_page": text_end, "subtopics": subs[:6]})
    # The chapter title is a topic only when nothing else opens the chapter
    # (a unit theme like "Wit and Wisdom" is a label over its readings, not a reading).
    if chapter_key and any(_squash(t["title"]) == chapter_key for t in topics):
        others = [t for t in topics if _squash(t["title"]) != chapter_key]
        if others and min(t["page"] for t in others) <= first_page + 1:
            topics = others
    if not topics:
        return None
    topics.sort(key=lambda t: t["page"])
    # text_end_page must sit inside the topic's own page range.
    for i, t in enumerate(topics):
        nxt = topics[i + 1]["page"] if i + 1 < len(topics) else None
        te = t.get("text_end_page")
        if te is None or te < t["page"] or (nxt is not None and te >= nxt):
            t["text_end_page"] = None
    end = data.get("end_matter_page")
    try:
        end_page = int(end) if end is not None else None
    except (TypeError, ValueError):
        end_page = None
    # End matter can't start before the last topic.
    if end_page is not None and end_page <= topics[-1]["page"]:
        end_page = None
    return {"topics": topics[:10], "end_matter_page": end_page}


def _cache_key(upload_id: int | str) -> str:
    return f"{_CACHE_PREFIX}/v{OUTLINE_VERSION}/{upload_id}.json"


def _load_cached(upload_id: int | str, fp: str) -> dict | None:
    try:
        from app.services.image_service.storage_backend import get_document_storage_backend

        raw = get_document_storage_backend().get_bytes(_cache_key(upload_id))
        if not raw:
            return None
        data = json.loads(raw.decode("utf-8"))
        if data.get("fingerprint") == fp:
            return data
    except Exception as exc:
        logger.debug("[OUTLINE] cache read %s failed: %s", upload_id, exc)
    return None


def _save_cached(upload_id: int | str, data: dict) -> None:
    try:
        from app.services.image_service.storage_backend import get_document_storage_backend

        get_document_storage_backend().save(
            _cache_key(upload_id), json.dumps(data, ensure_ascii=False).encode("utf-8")
        )
    except Exception as exc:
        logger.warning("[OUTLINE] cache write %s failed: %s", upload_id, exc)


def get_chapter_outline(
    upload_id: int | str,
    chunks: list[Any],
    *,
    chapter_name: str = "",
    subject_name: str = "",
    class_level: str = "",
    allow_llm: bool = True,
) -> dict | None:
    """Outline for one chapter, or None when it cannot be built (caller keeps its fallback)."""
    if not chunks:
        return None
    uid = str(upload_id)
    fp = fingerprint(chunks)
    mkey = f"{uid}:{fp}"
    hit = _mem.get(mkey)
    if hit is not None:
        at, data = hit
        if data is not None or time.monotonic() - at < _FAIL_RETRY_SEC:
            return data
    with _locks_guard:
        lock = _locks[uid]
    with lock:
        hit = _mem.get(mkey)
        if hit is not None and (hit[1] is not None or time.monotonic() - hit[0] < _FAIL_RETRY_SEC):
            return hit[1]
        cached = _load_cached(uid, fp)
        if cached is not None:
            _mem[mkey] = (time.monotonic(), cached)
            return cached
        if not allow_llm:
            return None
        skeleton, lines = build_skeleton(chunks)
        if not lines:
            _mem[mkey] = (time.monotonic(), None)
            return None
        try:
            picked = _llm_pick(
                skeleton, chapter_name=chapter_name, subject_name=subject_name, class_level=class_level
            )
        except Exception as exc:
            logger.warning("[OUTLINE] LLM failed for upload %s: %s", uid, exc)
            picked = None
        outline = validate_outline(picked, lines, chapter_name=chapter_name)
        if outline is None:
            _mem[mkey] = (time.monotonic(), None)
            return None
        outline.update(
            {"version": OUTLINE_VERSION, "fingerprint": fp, "upload_id": uid, "method": "llm_skeleton"}
        )
        _save_cached(uid, outline)
        _mem[mkey] = (time.monotonic(), outline)
        logger.info(
            "[OUTLINE] upload %s: %s",
            uid,
            [t["title"] for t in outline["topics"]],
        )
        return outline


def outline_topic_rows(outline: dict | None) -> list[dict[str, str]]:
    """topic_progress row shape ({key, title}) for the top-level topics."""
    from app.modules.student_learning.topic_progress import topic_key

    rows: list[dict[str, str]] = []
    for t in (outline or {}).get("topics") or []:
        title = (t.get("title") or "").strip()
        key = topic_key(title)
        if key:
            rows.append({"key": key, "title": title, "page": str(t.get("page", ""))})
    return rows


def lesson_docs(docs: list[Any], outline: dict | None) -> list[Any]:
    """Drop end matter (transcripts, answer keys) from an overview's context."""
    end = (outline or {}).get("end_matter_page")
    if not docs or end is None:
        return docs
    kept = [d for d in docs if _page(d) < int(end)]
    return kept or docs


_ACTIVITY_RE = re.compile(
    r"\b(?:let us|let's|before (?:we|you) read|work in (?:pairs|groups)|discuss|share (?:your|the) answers?|"
    r"classmates|given below|fill in|fill out|match the|answer the following|tick|circle the|"
    r"underline|complete the|rewrite|find out|look at the picture|observe the|write (?:a|the|down)|"
    r"read the (?:title|following)|you will (?:listen|find)|one example has been done|"
    r"note:|activity\b|exercise\b|project\b|https?://)",
    re.I,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\u201d\"])\s+(?=[A-Z0-9\u201c\"(#])|\s+#\s+")


_REPEATED_PHRASE_RE = re.compile(r"\b(.{4,40}?)(?:\s+\1){2,}")


def _lesson_prose(text: str) -> str:
    """Drop activity prompts and question lists; keep the reading / explanation itself."""
    keep: list[str] = []
    flat = _REPEATED_PHRASE_RE.sub(r"\1", re.sub(r"\s+", " ", text or ""))
    # Keep list markers on their item ("1. Riya's …" -> "1) Riya's …") so the
    # sentence split below doesn't detach them and let the item pass as prose.
    flat = re.sub(r"(?:(?<=\s)|^)(\d{1,2}|[IVX]{1,4}|[a-h])\.\s+(?=[A-Z\u2018\u201c\"'])", r"\1) ", flat)
    for sent in _SENT_SPLIT_RE.split(flat):
        sent = sent.strip(" #")
        if len(sent) < 25:
            continue
        if _ACTIVITY_RE.search(sent):
            continue
        # Fill-in-the-blank and multiple-choice items.
        if "___" in sent or re.search(r"\((?:ii|b)\)", sent):
            continue
        # Numbered prompts: "2. Why do you think kings …?", "II Work in pairs".
        if re.match(r"^(?:\d{1,2}|[IVX]{1,4}|[a-h])[.)]?\s", sent) and (sent.endswith("?") or len(sent) < 120):
            continue
        # Discussion questions to the student (dialogue questions are quoted).
        if sent.endswith("?") and not re.search(r"[\u201c\u201d\"]", sent):
            continue
        keep.append(sent)
    return " ".join(keep)


def _slices(text: str, n: int, size: int) -> str:
    """Beginning, middle and end of a topic — a story's plot is not in its first lines."""
    if len(text) <= n * size:
        return text
    step = (len(text) - size) / max(1, n - 1)
    parts = []
    for i in range(n):
        start = int(i * step)
        # Start on a word boundary.
        if start:
            sp = text.find(" ", start)
            start = sp + 1 if 0 <= sp < start + 40 else start
        parts.append(text[start : start + size].strip())
    return " … ".join(parts)


def topic_excerpts_block(
    chunks: list[Any], outline: dict | None, *, per_topic_chars: int = 1050, max_chars: int = 4800
) -> str:
    """The opening text of each topic, so a whole-chapter summary is grounded in every part."""
    topics = (outline or {}).get("topics") or []
    if not topics or not chunks:
        return ""
    ordered = sorted(chunks, key=_page)
    end = (outline or {}).get("end_matter_page")
    # Fewer topics get longer slices; the end slice carries a story's outcome.
    per_topic_chars = max(per_topic_chars, min(1800, max_chars // max(1, len(topics)) - 80))
    parts: list[str] = [
        "LESSON TEXT FOR EACH TOPIC (summarise each topic only from its own text, "
        "including who the main characters are, how it ends and its main message or idea):"
    ]
    used = len(parts[0])
    for i, t in enumerate(topics):
        start = int(t.get("page") or 0)
        stop = int(topics[i + 1]["page"]) if i + 1 < len(topics) else (int(end) if end is not None else 10**6)
        # Only the topic's own text, not the exercises that follow it.
        if t.get("text_end_page") is not None:
            stop = min(stop, int(t["text_end_page"]) + 1)
        body: list[str] = []
        for d in ordered:
            p = _page(d)
            if p < start or p >= max(stop, start + 1):
                continue
            text = _lesson_prose(getattr(d, "page_content", "") or "")
            if text:
                body.append(text)
        if not body:
            continue
        excerpt = _slices(" ".join(body), 3, per_topic_chars // 3)
        row = f"\n[{i + 1}. {t['title']} — from page {start}]\n{excerpt}"
        if used + len(row) > max_chars:
            break
        parts.append(row)
        used += len(row)
    return "\n".join(parts) if len(parts) > 1 else ""


def clear_memory_cache() -> None:
    """Test helper."""
    _mem.clear()


__all__ = [
    "build_skeleton",
    "clear_memory_cache",
    "fingerprint",
    "get_chapter_outline",
    "lesson_docs",
    "outline_topic_rows",
    "topic_excerpts_block",
    "validate_outline",
]
