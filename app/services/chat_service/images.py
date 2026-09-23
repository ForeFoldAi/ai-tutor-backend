"""Related textbook image fetch helpers for chat turns."""
from __future__ import annotations

import logging
import re
from typing import Any

from app.config import (
    ENABLE_LLM_IMAGE_SELECT,
    LLM_IMAGE_SELECT_CANDIDATES,
    TOP_RELATED_IMAGES,
)

logger = logging.getLogger(__name__)

# ponytail: relevance floor for fail-open when LLM select returns []; raise if noise returns
_STRONG_RELEVANCE = 80.0
_MAX_CITED_FIGURES = 8


def _log_image_stage(stage: str, images: list[dict]) -> None:
    """Structured per-stage image trace for debugging pipeline leaks."""
    try:
        summary = [
            f"{x.get('file_name') or x.get('image_url', '?')}:"
            f"{(x.get('caption') or '')[:60]}"
            for x in (images or [])
        ]
    except Exception:
        summary = ["<unserializable>"]
    logger.debug("[IMAGE-STAGE] stage=%s count=%d images=%s", stage, len(images or []), summary)


def _image_top_n_for_scope(scope, *, docs: list | None = None) -> int:
    from app.services.section_heading import HeadingScope, subtopics_for_main_section

    if ENABLE_LLM_IMAGE_SELECT:
        return max(TOP_RELATED_IMAGES, LLM_IMAGE_SELECT_CANDIDATES)

    if isinstance(scope, HeadingScope) and scope.is_main_section:
        n = len(subtopics_for_main_section(scope, docs or []))
        if n == 0:
            n = len(scope.child_headings or [])
        return min(12, max(TOP_RELATED_IMAGES, n + 1))
    return TOP_RELATED_IMAGES


def _fetch_related_images(
    scope,
    *,
    collection_name: str,
    chapter_ids: list[str],
    chapter_names: list[str],
    chapter: str,
    query: str,
    docs: list,
    conversation_history: list[dict] | None,
    top_n: int,
) -> list[dict]:
    from app.services.section_retrieval import related_images_for_heading_scope

    return related_images_for_heading_scope(
        scope,
        collection_name=collection_name,
        chapter_ids=chapter_ids,
        chapter_names=chapter_names,
        chapter_single=chapter,
        query=query,
        retrieved_docs=docs,
        conversation_history=conversation_history,
        fallback_top_n=top_n,
    )


def _docs_text(docs: list[Any] | None) -> str:
    parts: list[str] = []
    for d in docs or []:
        t = getattr(d, "page_content", None) or ""
        if t:
            parts.append(t)
    return "\n".join(parts)


def figure_hint_block_from_docs(docs: list[Any] | None) -> str:
    """Short prompt block so the tutor only names figures present in RAG."""
    from app.services.image_service.image_intent_extractor import figure_numbers_cited_in_text

    nums = figure_numbers_cited_in_text(_docs_text(docs))[:8]
    if not nums:
        return (
            "TEXTBOOK FIGURES: If the student asks to show Fig./Figure N, acknowledge it is "
            "being shown — never say you cannot see textbook figures. Do not invent figure numbers. "
            "If they ask to list chapter figures, say the figures are shown below — do not invent a list."
        )
    listed = ", ".join(f"Fig. {n}" for n in nums)
    return (
        f"TEXTBOOK FIGURES IN CONTEXT (app shows matching images automatically): {listed}. "
        "You may name only these. If the student asks to show one of them, acknowledge briefly — "
        "never say you cannot see it. Do not invent other figure numbers. "
        "If they ask to list chapter figures, say the figures are shown below."
    )


def figure_inventory_text(images: list[dict]) -> str:
    """Deterministic short list for chapter figure inventory asks."""
    if not images:
        return "I couldn't find textbook figures for this chapter right now."
    lines = ["Here are the textbook figures from this chapter:"]
    for im in images:
        num = str(im.get("figure_number") or "").strip()
        cap = (im.get("caption") or "").strip()
        if num and cap:
            lines.append(f"Fig. {num} — {cap}")
        elif num:
            lines.append(f"Fig. {num}")
        elif cap:
            lines.append(cap)
    lines.append("The matching images are shown below.")
    return "\n".join(lines)


def _match_cited(
    cited: list[str],
    candidates: list[dict],
) -> list[dict]:
    by_num: dict[str, dict] = {}
    for c in candidates:
        num = str(c.get("figure_number") or "").strip()
        if num and num not in by_num:
            by_num[num] = c
    return [by_num[n] for n in cited if n in by_num]


def _prefer_cited_figure_candidates(
    question: str,
    answer: str,
    candidates: list[dict],
    *,
    retrieved_docs: list[Any] | None = None,
) -> list[dict] | None:
    """Prefer Fig N cites on ranked candidates: question → RAG → answer (trusted)."""
    from app.services.image_service.image_intent_extractor import figure_numbers_cited_in_text

    if not candidates:
        return None

    q_cited = figure_numbers_cited_in_text(question)
    hit = _match_cited(q_cited, candidates)
    if hit:
        return hit

    rag_cited = figure_numbers_cited_in_text(_docs_text(retrieved_docs))
    hit = _match_cited(rag_cited, candidates)
    if hit:
        return hit

    a_cited = figure_numbers_cited_in_text(answer)
    if not a_cited:
        return None
    rag_set = set(rag_cited)
    trusted: list[str] = []
    by_num = {
        str(c.get("figure_number") or "").strip(): c
        for c in candidates
        if c.get("figure_number")
    }
    for n in a_cited:
        c = by_num.get(n)
        if not c:
            continue
        if n in rag_set or float(c.get("relevance") or 0) >= _STRONG_RELEVANCE:
            trusted.append(n)
    return _match_cited(trusted, candidates) or None


def _merge_cited_with_candidates(
    cited: list[dict],
    candidates: list[dict] | None,
    *,
    max_n: int,
) -> list[dict]:
    """Cited figures first; optionally fill remaining slots from ranked candidates."""
    if max_n <= 0:
        return []
    # When we only have exact DB hits, prefer not padding with unrelated ranked noise
    # beyond TOP_RELATED_IMAGES — but always keep every cited figure (up to max_n).
    if len(cited) >= max_n:
        return cited[:max_n]
    seen: set[str] = set()
    out: list[dict] = []
    for c in cited:
        key = str(c.get("url") or c.get("file_name") or id(c))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    fill_cap = max(max_n, TOP_RELATED_IMAGES)
    for c in candidates or []:
        if len(out) >= fill_cap:
            break
        key = str(c.get("url") or c.get("file_name") or id(c))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _db_figures_for_numbers(
    chapter_ids: list[str] | None,
    numbers: list[str],
) -> list[dict]:
    if not chapter_ids or not numbers:
        return []
    from app.services.image_service.textbook_image_retrieval import chapter_figures_by_number

    return chapter_figures_by_number(
        chapter_ids,
        numbers,
        max_n=min(_MAX_CITED_FIGURES, max(1, len(numbers))),
    )


def _strong_relevance_fallback(candidates: list[dict]) -> list[dict]:
    strong = [
        c for c in candidates if float(c.get("relevance") or 0) >= _STRONG_RELEVANCE
    ]
    if not strong:
        return []
    strong.sort(key=lambda c: float(c.get("relevance") or 0), reverse=True)
    return strong[: max(1, TOP_RELATED_IMAGES)]


def _any_candidate_fallback(candidates: list[dict]) -> list[dict]:
    """Soft fail-open when the student explicitly asked to see a figure."""
    if not candidates:
        return []
    ranked = sorted(
        candidates, key=lambda c: float(c.get("relevance") or 0), reverse=True
    )
    return ranked[: max(1, TOP_RELATED_IMAGES)]


def _explicit_show_image_ask(question: str) -> bool:
    from app.services.chapter_scope import is_image_follow_up_request
    from app.services.image_service.image_intent_extractor import (
        _IMAGE_REQUEST_RE,
        is_chapter_figure_list_ask,
    )

    q = question or ""
    if (
        is_image_follow_up_request(q)
        or _IMAGE_REQUEST_RE.search(q)
        or is_chapter_figure_list_ask(q)
    ):
        return True
    return bool(
        re.search(
            r"\b(?:show|see|display|open|bring|get|fetch)\b.{0,48}\b"
            r"(?:images?|pictures?|figures?|diagrams?|maps?|illustrations?)\b",
            q,
            re.I,
        )
    )


async def _select_related_images_for_answer(
    question: str,
    answer: str,
    candidates: list[dict],
    *,
    retrieved_docs: list[Any] | None = None,
    chapter_ids: list[str] | None = None,
) -> list[dict]:
    """Prefer exact Fig N (DB) / list-ask catalog; else candidates + LLM pick."""
    from app.services.image_service.image_intent_extractor import (
        figure_numbers_cited_in_text,
        is_chapter_figure_list_ask,
    )

    if is_chapter_figure_list_ask(question) and chapter_ids:
        from app.services.image_service.textbook_image_retrieval import (
            chapter_figures_catalog,
        )

        catalog = chapter_figures_catalog(chapter_ids)
        if catalog:
            _log_image_stage("chapter_figure_catalog", catalog)
            return catalog

    q_cited = figure_numbers_cited_in_text(question)
    a_cited = figure_numbers_cited_in_text(answer)

    if chapter_ids and q_cited:
        found = _db_figures_for_numbers(chapter_ids, q_cited)
        if found:
            merged = _merge_cited_with_candidates(
                found,
                candidates,
                max_n=max(TOP_RELATED_IMAGES, min(_MAX_CITED_FIGURES, len(found))),
            )
            _log_image_stage("question_fig_db", merged)
            return merged

    if chapter_ids and a_cited:
        found = _db_figures_for_numbers(chapter_ids, a_cited)
        if found:
            # Answer-cited figures are guaranteed when the chapter DB has them.
            merged = _merge_cited_with_candidates(
                found,
                candidates,
                max_n=max(TOP_RELATED_IMAGES, min(_MAX_CITED_FIGURES, len(found))),
            )
            _log_image_stage("answer_fig_db", merged)
            return merged

    cited = _prefer_cited_figure_candidates(
        question, answer, candidates, retrieved_docs=retrieved_docs
    )
    if cited:
        _log_image_stage("cited_figure_short_circuit", cited)
        return cited[: max(1, TOP_RELATED_IMAGES)]
    if not ENABLE_LLM_IMAGE_SELECT:
        if _explicit_show_image_ask(question) and candidates:
            picked = _strong_relevance_fallback(candidates) or _any_candidate_fallback(
                candidates
            )
            _log_image_stage("explicit_show_no_llm", picked)
            return picked
        return candidates
    from app.services.image_service.llm_image_select import select_images_for_qa

    selected = await select_images_for_qa(question, answer, candidates)
    if selected:
        return selected
    fallback = _strong_relevance_fallback(candidates)
    if fallback:
        _log_image_stage("llm_select_fail_open", fallback)
        return fallback
    if _explicit_show_image_ask(question):
        soft = _any_candidate_fallback(candidates)
        if soft:
            _log_image_stage("explicit_show_soft_fail_open", soft)
        return soft
    return []
