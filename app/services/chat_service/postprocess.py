"""Answer post-processing: normalize, shrink, expand, context assembly."""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from app.config import CONTEXT_CHAR_BUDGET
from app.services.chat_service.answer_types import (
    _ANSWER_MIN_WORDS,
    _MATH_DIALOGUE_TYPES,
    _MATH_SHORT_TYPES,
    _is_mathematics_subject,
    _structure_tier,
    detect_answer_type,
    detect_question_type,
)
from app.services.chat_service.dialogue import _skip_answer_expansion
from app.services.chat_service.llm import _call_mistral_async
from app.services.llm_client import strip_emojis

logger = logging.getLogger(__name__)


_DIRECT_ANSWER_MAX_WORDS = 100


_DIRECT_ANSWER_TOKEN_LIMIT = 160


_MAIN_SECTION_TOKEN_LIMIT = 700


_STRUCTURED_HEADER_RE = re.compile(
    r"\*\*(Topic|In Simple Words|Key Points|Detailed Explanation|Important Terms|"
    r"Remember|Try This|Steps)\*\*",
    re.I,
)


_FIG_CAPTION_LINE_RE = re.compile(
    r"^\s*Fig\.?\s*\d+(?:\.\d+)*\s*(?:[.:—–-]\s*)?.+$",
    re.I,
)


_PAGE_LINE_RE = re.compile(r"^\s*Page\s+\d+\s*$", re.I)


def strip_embedded_figure_lines(text: str) -> str:
    """Drop Fig./Page lines the model pasted — figures render as image cards."""
    if not (text or "").strip():
        return text or ""
    kept: list[str] = []
    for line in text.splitlines():
        t = line.strip()
        if not t:
            if kept and kept[-1].strip():
                kept.append("")
            continue
        if _FIG_CAPTION_LINE_RE.match(t) or _PAGE_LINE_RE.match(t):
            continue
        if kept and kept[-1].strip() == t:
            continue
        kept.append(line)
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return strip_emojis(out)


def normalize_direct_answer_prose(text: str) -> str:
    """Fix lone **Topic** lines and mid-sentence line breaks in direct answers."""
    if not (text or "").strip():
        return text or ""
    out = text
    # **Weather**\n is → Weather is
    out = re.sub(
        r"^\s*\*\*([^*\n]+)\*\*\s*\n+(?=[a-z])",
        lambda m: f"{m.group(1).strip()} ",
        out,
        count=1,
        flags=re.M | re.I,
    )
    # "in the\n\ntroposphere" → "in the troposphere"
    out = re.sub(r"(\bthe)\s*\n+\s*(\w+)", r"\1 \2", out, flags=re.I)
    out = re.sub(r"(\w)\s*\n+\s*([,;.])", r"\1\2", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _format_chunk_for_context(doc) -> str:
    meta = getattr(doc, "metadata", None) or {}
    page_raw = meta.get("page")
    human_page = None
    if page_raw is not None:
        try:
            human_page = int(page_raw) + 1
        except (TypeError, ValueError):
            pass
    src = meta.get("source", "")
    short_src = os.path.basename(str(src)) if src else ""
    sec = (meta.get("section_hint") or "").strip()
    tag = f"[page {human_page}]" if human_page is not None else "[page ?]"
    if short_src:
        tag += f" [source: {short_src}]"
    if sec:
        tag += f" [section: {sec}]"
    body = (getattr(doc, "page_content", "") or "").strip()
    return f"{tag}\n{body}"


def _join_context_within_budget(docs: list, char_budget: int | None = None) -> str:
    budget = char_budget if char_budget is not None else CONTEXT_CHAR_BUDGET
    parts: list[str] = []
    used = 0
    for d in docs:
        block = _format_chunk_for_context(d)
        add = len(block) + (2 if parts else 0)
        if parts and used + add > budget:
            break
        if not parts and len(block) > budget:
            parts.append(block[:budget])
            break
        parts.append(block)
        used += add
    return "\n\n".join(parts)


def _answer_word_count(text: str) -> int:
    return len((text or "").split())


def _direct_answer_needs_shrink(answer: str) -> bool:
    wc = _answer_word_count(answer)
    if wc > _DIRECT_ANSWER_MAX_WORDS:
        return True
    return bool(_STRUCTURED_HEADER_RE.search(answer or ""))


def _mistral_token_limit_for_answer_type(
    answer_type: str,
    *,
    heading_scope: Any | None = None,
    subject_name: str = "",
    query: str = "",
) -> int | None:
    from app.config import MATH_CHAT_MAX_TOKENS
    from app.services.section_heading import HeadingScope

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        return _MAIN_SECTION_TOKEN_LIMIT
    # Math answers must include ```math-lesson``` JSON — 160 tokens cuts it off
    # and the incomplete fence wipes the reply ("Sorry — something went wrong").
    if _is_mathematics_subject(subject_name) and answer_type not in (
        _MATH_DIALOGUE_TYPES | _MATH_SHORT_TYPES
    ):
        return MATH_CHAT_MAX_TOKENS
    # Science/other calc problems also need room for formulas (160 tokens dies mid-`\[`).
    if detect_question_type(query) == "problem-solving" or answer_type == "stepwise":
        return _MAIN_SECTION_TOKEN_LIMIT
    if _structure_tier(answer_type) == "direct":
        return _DIRECT_ANSWER_TOKEN_LIMIT
    return None


async def _finalize_direct_answer(
    messages: list[dict[str, str]],
    answer: str,
    query: str,
    *,
    answer_type: str,
    heading_scope: Any | None = None,
    conversation_history: list[dict] | None = None,
    subject_name: str = "",
) -> str:
    """Shrink direct answers that leaked into long/structured form. Never expand them."""
    from app.services.image_service.image_intent_extractor import is_chapter_figure_list_ask
    from app.services.section_heading import HeadingScope

    def _finish(text: str) -> str:
        # Keep deterministic Fig. inventory lines for chapter list-asks.
        if is_chapter_figure_list_ask(query):
            return text.strip()
        return strip_embedded_figure_lines(text.strip())

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        return _finish(answer)
    if _is_mathematics_subject(subject_name) and answer_type not in (
        _MATH_DIALOGUE_TYPES | _MATH_SHORT_TYPES
    ):
        # Don't shrink math — required sections + math-lesson JSON look "too long".
        return _finish(answer)
    if _structure_tier(answer_type) != "direct":
        if not _skip_answer_expansion(query, conversation_history):
            answer = await _expand_short_answer(
                messages, answer, query, heading_scope=heading_scope
            )
        return _finish(answer)
    return _finish(
        normalize_direct_answer_prose(
            (await _shrink_overlong_direct_answer(messages, answer, query)).strip()
        )
    )


async def _shrink_overlong_direct_answer(
    messages: list[dict[str, str]],
    answer: str,
    query: str,
) -> str:
    """One retry when the model ignores direct-answer length / format rules."""
    if not _direct_answer_needs_shrink(answer):
        return answer
    logger.info(
        "Direct answer too long or structured (%d words); shrinking for query=%r",
        _answer_word_count(answer),
        query[:60],
    )
    shrink_msgs = [
        *messages,
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": (
                f"Rewrite that answer in MAX {_DIRECT_ANSWER_MAX_WORDS} words. "
                "Plain prose only — NO **Topic**, **Key Points**, **Detailed Explanation**, "
                "**Remember**, or **Try This** headers. "
                "Keep the definition, at most 3 bullets, and one short follow-up question."
            ),
        },
    ]
    try:
        shorter = await _call_mistral_async(
            shrink_msgs, max_tokens=_DIRECT_ANSWER_TOKEN_LIMIT
        )
        if shorter.strip() and _answer_word_count(shorter) <= _DIRECT_ANSWER_MAX_WORDS + 25:
            return shorter.strip()
    except Exception as exc:
        logger.warning("Direct-answer shrink failed: %s", exc)
    return answer


async def _expand_short_answer(
    messages: list[dict[str, str]],
    answer: str,
    query: str,
    *,
    heading_scope: Any | None = None,
) -> str:
    """One retry when the model stops too early despite length instructions."""
    from app.services.section_heading import HeadingScope

    if isinstance(heading_scope, HeadingScope) and heading_scope.is_main_section:
        return answer
    atype = detect_answer_type(query)
    if _structure_tier(atype) == "direct":
        return answer
    if atype in ("greeting", "one-word", "brief"):
        return answer
    min_w = _ANSWER_MIN_WORDS.get(atype, 120)
    if _answer_word_count(answer) >= int(min_w * 0.65):
        return answer
    logger.info(
        "Answer too short (%d words, need ~%d); requesting expansion for query=%r",
        _answer_word_count(answer),
        min_w,
        query[:60],
    )
    if _structure_tier(atype) == "full":
        expand_hint = (
            "Include ALL sections with **bold** headers (NO emoji): "
            "**Concept Overview**, **Detailed Explanation**, **Real-Life Example**, "
            "**Key Points to Remember** (3-5 bullets), **Quick Check**."
        )
    elif _structure_tier(atype) == "structured":
        expand_hint = (
            "Include ALL **bold** side headings: **Topic**, **In Simple Words**, **Key Points**, "
            "**Detailed Explanation**, **Example**, **Remember**, and **Try This**."
        )
    else:
        expand_hint = (
            "Continue in plain paragraphs only — NO emoji section headers, "
            "no 'Concept Overview' / 'Quick Check' labels."
        )
    extend_msgs = [
        *messages,
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": (
                f"Your answer was too short ({_answer_word_count(answer)} words). "
                f"Continue and expand until the full answer is at least {min_w} words. "
                f"{expand_hint}"
            ),
        },
    ]
    try:
        extra = await _call_mistral_async(extend_msgs)
        if extra.strip():
            return f"{answer.strip()}\n\n{extra.strip()}"
    except Exception as exc:
        logger.warning("Short-answer expansion failed: %s", exc)
    return answer

