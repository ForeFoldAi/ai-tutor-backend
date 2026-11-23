"""Regression tests for the Suneel QA accuracy fixes (scope gate, fallbacks, figures)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.services import chapter_scope as cs
from app.services.chat_service import llm as chat_llm
from app.services.chat_service import qa


def _doc(text: str, page: int = 1):
    return SimpleNamespace(page_content=text, metadata={"page": page})


# --- scope gate: plurals, typos, filler words, title topics ---------------------------


def test_plural_query_term_matches_singular_textbook_word():
    assert cs._term_hits_in_text({"mughals"}, "The Mughal empire expanded under Akbar.") == 1


def test_typo_term_fuzzy_matches_chapter_word():
    assert cs._term_hits_in_text({"squre"}, "The square of a number is the number times itself.") == 1


def test_real_english_word_is_not_fuzzy_matched():
    assert cs._term_hits_in_text({"cube"}, "a tube of paint") == 0
    assert cs._term_hits_in_text({"telephone"}, "The Mughal empire expanded under Akbar.") == 0


def test_chapter_title_topic_counts_as_in_scope():
    assert cs._any_term_in_chapter_titles({"shivaji", "marathas"}, ["Chapter 3 - The Rise of the Marathas"])
    assert not cs._any_term_in_chapter_titles({"telephone"}, ["Chapter 3 - The Rise of the Marathas"])


def test_title_topic_question_is_full_coverage_without_cross_chapter_search(monkeypatch):
    monkeypatch.setattr(cs, "_best_other_chapter", lambda *a, **k: pytest.fail("should not search"))
    a = cs.assess_chapter_coverage(
        "who was shivaji and why are the marathas important?",
        docs=[_doc("Shivaji built forts.")],
        collection_name="x",
        chapter_ids=["34"],
        chapter_names=["Chapter 3 - The Rise of the Marathas"],
        board="CBSE",
        class_level="CLASS_8",
        subject_name="Social",
    )
    assert a.level == cs.ChapterCoverageLevel.FULL


def test_scope_phrase_is_stripped_for_llm_verdict():
    assert cs._strip_scope_phrase("who were the mughals in this chapter?") == "who were the mughals?"


def test_no_llm_verdict_answers_instead_of_refusing(monkeypatch):
    async def _none(*a, **k):
        return None

    monkeypatch.setattr(cs, "_llm_confirms_topic_mismatch", _none)
    monkeypatch.setattr(
        cs,
        "assess_chapter_coverage",
        lambda *a, **k: cs.ChapterCoverageAssessment(
            level=cs.ChapterCoverageLevel.NONE,
            topic_label="q",
            current_chapter_label="Chapter 2 - Reshaping India’s Political Map",
        ),
    )
    early, _, _, guidance = asyncio.run(
        cs.resolve_chapter_awareness_turn(
            "who were the mughals",
            docs=[],
            conversation_history=[],
            collection_name="x",
            chapter_ids=["33"],
            chapter_names=["Chapter 2 - Reshaping India’s Political Map"],
            board="CBSE",
            class_level="CLASS_8",
            subject_name="Social",
        )
    )
    assert early is None
    assert guidance.startswith("CHAPTER COVERAGE: RELATED")


# --- fallbacks ------------------------------------------------------------------------


def test_chunk_fallback_strips_page_headers_and_figure_labels():
    out = qa._best_chunk_fallback(
        "political map",
        [_doc(
            "22 Exploring Society: India and Beyond | Grade 8 Part 1 Fig. 2.2. The period in this "
            "chapter and the next is often called the latter part of the medieval period of Indian history."
        )],
    )
    assert "|" not in out and "Grade 8" not in out and "Fig." not in out
    assert out.startswith("Here's what your textbook says:")


def test_stream_retries_once_when_first_attempt_is_empty(monkeypatch):
    calls = {"n": 0}

    async def fake_stream(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return
        for t in ("Hello", " there"):
            yield t

    monkeypatch.setattr(chat_llm.llm_client, "stream", fake_stream)
    monkeypatch.setattr(chat_llm, "_STREAM_RETRY_DELAY_SEC", 0)

    async def run():
        return [t async for t in chat_llm._stream_mistral_async([{"role": "user", "content": "x"}])]

    assert "".join(asyncio.run(run())) == "Hello there"
    assert calls["n"] == 2


def test_stream_error_after_first_token_is_not_retried(monkeypatch):
    calls = {"n": 0}

    async def fake_stream(*a, **k):
        calls["n"] += 1
        yield "partial"
        raise RuntimeError("boom")

    monkeypatch.setattr(chat_llm.llm_client, "stream", fake_stream)

    async def run():
        return [t async for t in chat_llm._stream_mistral_async([{"role": "user", "content": "x"}])]

    with pytest.raises(RuntimeError):
        asyncio.run(run())
    assert calls["n"] == 1


def test_completion_retries_on_error_then_succeeds(monkeypatch):
    calls = {"n": 0}

    async def fake_complete(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429")
        return "ok answer"

    monkeypatch.setattr(chat_llm.llm_client, "complete", fake_complete)
    monkeypatch.setattr(chat_llm, "_STREAM_RETRY_DELAY_SEC", 0)
    assert asyncio.run(chat_llm._call_mistral_async([{"role": "user", "content": "x"}])) == "ok answer"


# --- figure requests ------------------------------------------------------------------


def test_that_figure_resolves_to_figure_named_in_last_reply():
    hist = [
        {"role": "user", "content": "show me a map"},
        {"role": "assistant", "content": "Look at Fig. 2.3 on your screen."},
    ]
    nums, rows = asyncio.run(
        qa._resolve_figure_request("what is that figure trying to show?", hist, ["33"], [], None)
    )
    assert nums == ["2.3"] and rows == []


def test_generic_diagram_request_uses_chapter_figures(monkeypatch):
    from app.services.image_service import textbook_image_retrieval as tir

    rows = [{"figure_number": None, "page": 1, "caption": "Probe and ponder"}]
    monkeypatch.setattr(tir, "chapter_figures_for_generic_request", lambda *a, **k: rows)
    msgs = [{"role": "system", "content": "base"}]
    nums, got = asyncio.run(
        qa._resolve_figure_request(
            "can you show a diagram or figure from this science chapter?", [], ["35"], [], msgs
        )
    )
    assert got == rows and nums == []
    assert "ALREADY DISPLAYED" in msgs[0]["content"]
    assert "Never say you cannot show" in msgs[0]["content"]


# --- round 2: collection resolution, failure messages, bridge repeats, image follow-ups ----


def test_collection_resolved_from_chapter_upload(monkeypatch):
    from app.services import chapter_collection as cc

    cc._CACHE.clear()
    monkeypatch.setattr(cc, "_upload_collection", lambda ids: "CBSE_CLASS_8_Social_-_Part-I")
    assert (
        cc.resolve_chat_collection("CBSE", "CLASS_8", "Social", ["33"])
        == "CBSE_CLASS_8_Social_-_Part-I"
    )


def test_collection_falls_back_to_request_on_db_error(monkeypatch):
    from app.services import chapter_collection as cc

    cc._CACHE.clear()

    def boom(ids):
        raise RuntimeError("db down")

    monkeypatch.setattr(cc, "_upload_collection", boom)
    assert cc.resolve_chat_collection("CBSE", "CLASS_8", "English", ["26"]) == "CBSE_CLASS_8_English"


def test_repeated_bridge_lead_is_removed_across_tokens():
    d = qa._PhraseDropper(qa._BRIDGE_REPEAT_RE)
    toks = ["It is 144.", " This i", "sn't in your ch", "apter, but it uses squares."]
    out = "".join(d.feed(t) for t in toks) + d.flush()
    assert "isn't in your chapter" not in out
    assert out == "It is 144. It uses squares."


def test_image_request_with_own_topic_is_not_swapped():
    assert cs._image_request_topic_terms("can you show the political map figure?") == {"political", "map"}
    assert cs._image_request_topic_terms("show me a diagram") == set()


def test_failed_llm_gives_honest_busy_message_not_chapter_refusal():
    from app.core.student_messages import ANSWER_NOT_IN_CHAPTER, CHAT_LLM_BUSY

    assert "chapter" not in CHAT_LLM_BUSY.lower()
    assert CHAT_LLM_BUSY != ANSWER_NOT_IN_CHAPTER
    assert "lost my train" not in qa.VOICE_RETRY_LINE


# --- round 3: maths correctness (2-question understanding eval) ---------------------------


@pytest.mark.parametrize(
    "wrong,right",
    [
        ("So **2⁵ = 64** candies.", "So **2⁵ = 32** candies."),
        ("2 to the power 5 is 64.", "2 to the power 5 is 32."),
        ("2 × 2 × 2 × 2 × 2 = 64", "2 × 2 × 2 × 2 × 2 = 32"),
        ("3 x 3 x 3 = 9", "3 x 3 x 3 = 27"),
    ],
)
def test_wrong_arithmetic_claims_are_corrected(wrong, right):
    from app.services.arithmetic_check import fix_arithmetic_claims

    assert fix_arithmetic_claims(wrong) == right


@pytest.mark.parametrize("ok", ["12 × 12 = 144", "Solve 2x = 10", "1,000 × 2 = 2,000", "2.5 × 2 = 5"])
def test_correct_or_non_integer_claims_untouched(ok):
    from app.services.arithmetic_check import fix_arithmetic_claims

    assert fix_arithmetic_claims(ok) == ok


def test_cut_off_answer_trimmed_to_last_sentence_not_abbreviation():
    from app.services.arithmetic_check import trim_incomplete_tail

    out = trim_incomplete_tail(
        "Political maps show the borders of kingdoms and who ruled each area at a given time.\n\n"
        "- They show how dynasties like the Tughlaqs and Lodis grew and shrank.\n- They show dynasties.\n\n"
        "**Quick check**: If you look at Fig. 2.30 (a Vij"
    )
    assert out.endswith("They show dynasties.")


def test_mean_as_verb_is_not_statistics():
    from app.services.math_engine import try_solve

    r = try_solve("what does 2 to the power 5 actually mean?", class_level="CLASS_8")
    assert r is not None and r.kind == "arithmetic"
    assert "**32**" in r.final_answer
    assert "NOT 2 × 5" in r.concept
    assert try_solve("find the mean of 2, 4, 6", class_level="CLASS_8").kind == "statistics"


def test_confused_follow_up_reuses_verified_math_question():
    from app.services.chat_service.prompts import _math_followup_query
    from app.services.math_engine import try_solve

    def solve(q):
        return try_solve(q, class_level="CLASS_8")

    hist = [
        {"role": "user", "content": "what does 2 to the power 5 actually mean?"},
        {"role": "assistant", "content": "..."},
    ]
    assert _math_followup_query(
        "i still don't really get it. can you explain it more simply?", hist, solve
    ) == "what does 2 to the power 5 actually mean?"
