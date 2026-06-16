"""
Test three weather-chapter questions: scope, images, and LLM answer shape.

Run from ai-tutor-backend:
  .venv/bin/python scripts/test_three_weather_questions.py
"""

from __future__ import annotations

import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

UPLOAD_ID = "97ce23be-1c71-45f2-942c-34ea82cc205c"
COLLECTION = "CBSE_CLASS_9_Social"
CHAPTER = "Chapter 2 - Understanding the Weather"

QUESTIONS = [
    "what is weather",
    "what are the Weather Instruments",
    "what is Weather Stations",
]

# Expected answer-type routing (see chat_service.detect_answer_type)
EXPECTED_ANSWER_TYPES: dict[str, str] = {
    "what is weather": "short-answer",
    "what are the Weather Instruments": "short-answer",
    "what is Weather Stations": "short-answer",
}

# Heading scope expectations
EXPECTED_SCOPE: dict[str, str] = {
    "what are the Weather Instruments": "main_section",
}

# Minimum image expectations (figure_number); empty tuple = images optional
EXPECTED_FIGURES: dict[str, tuple[str, ...]] = {
    "what is weather": ("2.2",),
    "what are the Weather Instruments": ("2.6",),
    "what is Weather Stations": (),
}

# Subtopics expected under a main-section heading
EXPECTED_SUBTOPICS: dict[str, tuple[str, ...]] = {
    "what are the Weather Instruments": (
        "Temperature",
        "Precipitation",
        "Atmospheric pressure",
        "Wind",
        "Humidity",
    ),
}

SEP = "=" * 72


def _answer_shape(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return "empty"
    has_bold_blocks = a.count("**") >= 4
    has_bullets = "•" in a or "\n-" in a or "\n* " in a
    emoji_sections = sum(1 for e in ("🌱", "📚", "🌍", "📝", "❓") if e in a)
    lines = [ln.strip() for ln in a.splitlines() if ln.strip()]
    plain_heads = sum(
        1
        for ln in lines
        if ln
        in (
            "Temperature",
            "Precipitation",
            "Atmospheric pressure",
            "Wind",
            "Humidity",
        )
        or ln.startswith("**")
    )
    return (
        f"chars={len(a)} bold_blocks={has_bold_blocks} bullets={has_bullets} "
        f"emoji_sections={emoji_sections} heading_lines≈{plain_heads}"
    )


def _answer_format_ok(
    answer: str,
    answer_type: str,
    *,
    scope_kind: str = "general",
) -> tuple[bool, str]:
    """Validate new direct/compact format — no legacy five emoji sections."""
    from app.services.chat_service import _structure_tier

    a = (answer or "").strip()
    if not a:
        return False, "empty answer"
    tier = _structure_tier(answer_type)
    emoji_hits = sum(1 for e in ("🌱", "📚", "🌍", "📝", "❓") if e in a)
    legacy_labels = (
        "Concept Overview",
        "Detailed Explanation",
        "Key Points to Remember",
        "Quick Check",
    )
    if tier in ("direct", "compact") and emoji_hits > 0:
        return False, f"unexpected emoji sections ({emoji_hits}) for tier={tier}"
    if tier in ("direct", "compact") and any(lbl in a for lbl in legacy_labels):
        return False, "legacy section labels in direct/compact answer"
    if scope_kind == "main_section":
        if "•" not in a and "\n-" not in a:
            return False, "main_section answer missing bullet points"
        bold_hits = a.count("**") // 2
        if bold_hits < 3:
            return False, f"main_section answer needs bold subtopic headings (got {bold_hits})"
        return True, "ok"
    if answer_type == "short-answer" and len(a.split()) < 40:
        return False, f"short-answer too brief ({len(a.split())} words)"
    return True, "ok"


async def main() -> None:
    from app.services.chat_service import (
        _fetch_related_images,
        _structure_tier,
        chapter_aware_qa,
        detect_answer_type,
    )
    from app.services.conversation_context import resolve_conversation_context, should_retrieve_images
    from app.services.section_heading import subtopics_detail_for_main_section
    from app.services.section_retrieval import retrieve_for_tutor_query

    print(SEP)
    print("THREE-QUESTION WEATHER CHAPTER TEST")
    print(f"upload={UPLOAD_ID}")
    print(SEP)

    results: list[tuple[str, bool, str]] = []

    for q in QUESTIONS:
        print(f"\n{SEP}\nQ: {q!r}\n{SEP}")

        answer_type = detect_answer_type(q)
        tier = _structure_tier(answer_type)
        print(f"  answer_type: {answer_type}  structure_tier: {tier}")
        exp_type = EXPECTED_ANSWER_TYPES.get(q)
        if exp_type and answer_type != exp_type:
            print(f"  WARN: expected answer_type={exp_type!r}, got {answer_type!r}")

        conv = resolve_conversation_context(q, chapter=CHAPTER)
        print(f"  context: mode={conv.response_mode.value} visual={conv.visual_intent.value}")

        docs, scope, instr = retrieve_for_tutor_query(
            q,
            collection_name=COLLECTION,
            chapter_ids=[UPLOAD_ID],
        )
        matched = scope.matched.title if scope.matched else None
        print(f"  scope: {scope.kind} matched={matched!r} pages={scope.page_start}-{scope.page_end}")
        exp_scope = EXPECTED_SCOPE.get(q)
        if exp_scope and scope.kind != exp_scope:
            print(f"  SCOPE FAIL: expected {exp_scope!r}, got {scope.kind!r}")

        subs = subtopics_detail_for_main_section(scope, docs)
        if subs:
            print(f"  subtopics ({len(subs)}): {[s.title for s in subs]}")
        exp_subs = EXPECTED_SUBTOPICS.get(q)
        if exp_subs:
            got = [s.title for s in subs]
            if got != list(exp_subs):
                print(f"  SUBTOPIC FAIL: expected {list(exp_subs)}, got {got}")

        img_ok = should_retrieve_images(conv, chapter_ids=[UPLOAD_ID], heading_scope_kind=scope.kind)
        print(f"  images_allowed: {img_ok}")

        imgs: list[dict] = []
        if img_ok:
            try:
                imgs = await asyncio.to_thread(
                    _fetch_related_images,
                    scope,
                    collection_name=COLLECTION,
                    chapter_ids=[UPLOAD_ID],
                    chapter_names=[CHAPTER],
                    chapter=CHAPTER,
                    query=q,
                    docs=docs,
                    conversation_history=None,
                    top_n=8,
                )
            except Exception as exc:
                print(f"  images ERROR: {exc}")
        print(f"  images ({len(imgs)}):")
        fig_nums = []
        for i, im in enumerate(imgs, 1):
            cap = (im.get("caption") or "")[:70]
            fn = im.get("figure_number")
            if fn:
                fig_nums.append(str(fn))
            print(
                f"    [{i}] subtopic={im.get('subtopic')!r} fig={fn} "
                f"page={im.get('page')} rel={im.get('relevance')} "
                f"kind={im.get('content_kind', 'figure')} | {cap}"
            )

        expected_figs = EXPECTED_FIGURES.get(q, ())
        img_ok = True
        img_note = "ok"
        if expected_figs:
            if not any(f in fig_nums for f in expected_figs):
                img_ok = False
                img_note = f"missing expected fig(s) {expected_figs}, got {fig_nums}"
                print(f"  IMAGE FAIL: {img_note}")
            else:
                print(f"  IMAGE PASS: found {expected_figs}")
        if q in EXPECTED_FIGURES and q == "what are the Weather Instruments":
            bad = [f for f in fig_nums if f.startswith("2.4.") or f == "2.5"]
            if bad:
                img_ok = False
                img_note = f"wrong figures must not appear: {bad}"
                print(f"  IMAGE FAIL: {img_note}")
            caps = [(im.get("subtopic"), im.get("caption") or "") for im in imgs]
            short_caps = [c for _, c in caps if len(c.strip()) < 15 or c.strip().isdigit()]
            if short_caps:
                img_ok = False
                img_note = f"captions too short or numeric: {short_caps[:3]}"
                print(f"  IMAGE FAIL: {img_note}")

        print("\n  --- LLM answer (may take ~15-30s) ---")
        llm_ok = False
        llm_note = "not run"
        try:
            answer, qa_imgs = await chapter_aware_qa(
                q,
                collection_name=COLLECTION,
                chapter_ids=[UPLOAD_ID],
                class_level="CLASS_9",
                board="CBSE",
                subject_name="Social",
                chapter=CHAPTER,
                chapter_names=[CHAPTER],
            )
            print(f"  shape: {_answer_shape(answer)}")
            print(f"  qa_images: {len(qa_imgs)}")
            fmt_ok, fmt_note = _answer_format_ok(answer, answer_type, scope_kind=scope.kind)
            llm_ok = fmt_ok
            llm_note = fmt_note
            print(f"  format_check: {'PASS' if fmt_ok else 'FAIL'} ({fmt_note})")
            preview = answer[:500].replace("\n", "\\n")
            if len(answer) > 500:
                preview += "..."
            print(f"  preview: {preview}")
        except Exception as exc:
            llm_ok = False
            llm_note = str(exc)
            print(f"  LLM ERROR: {exc}")

        scope_ok = not exp_scope or scope.kind == exp_scope
        subs_ok = True
        if exp_subs := EXPECTED_SUBTOPICS.get(q):
            subs_ok = [s.title for s in subs] == list(exp_subs)
        case_ok = img_ok and llm_ok and scope_ok and subs_ok
        results.append((q, case_ok, f"images={img_note}; llm={llm_note}"))
        print(f"  CASE: {'PASS' if case_ok else 'FAIL'}")

    print(f"\n{SEP}\nSUMMARY\n{SEP}")
    passed = sum(1 for _, ok, _ in results if ok)
    for q, ok, note in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {q!r}  ({note})")
    print(f"\n{passed}/{len(results)} passed")
    print(f"{SEP}\nDONE\n{SEP}")
    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
