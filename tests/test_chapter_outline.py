"""Lesson outlines for unnumbered chapters (readers, EVS, State Board books).

Regression: "explain this chapter" on Poorvi Class 8 Unit 1 listed a listening
activity and a teacher transcript as the chapter's topics, because the topic list
only understood 1.1-style numbers and a fixed set of Social Science words.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.services import chapter_outline as co


def _doc(page: int, text: str) -> Document:
    return Document(page_content=text, metadata={"page": page, "textbook_upload_id": "752"})


POORVI = [
    _doc(0, "Poorvi\n# The Wit that Won Hearts\nLet us do these activities before we read.\nReprint 2026-27"),
    _doc(1, "Poorvi\nLet us read\nThe Vijayanagara Empire was renowned for its glory.\nReprint 2026-27"),
    _doc(3, "Poorvi\nThe king remembered scolding the queen for yawning while he recited his poem.\nReprint 2026-27"),
    _doc(6, "Poorvi\nLet us discuss\n1. Why does the queen believe that only Tenali Rama can help?\nReprint 2026-27"),
    _doc(16, "Poorvi\n# A Concrete Example\nMy next-door neighbour, Mrs. Jones, has got a garden full of stones.\nReprint 2026-27"),
    _doc(26, "Poorvi\n# Wisdom Paves the Way\nScene I: The Road to Ujjain\nFour young men walk to Ujjain.\nReprint 2026-27"),
    _doc(32, "Poorvi\nScene II: The King's Court\nThe merchant accuses the young men of theft.\nReprint 2026-27"),
    _doc(45, "Poorvi\n# TRANSCRIPTS\nThe Wit that Won Hearts\nA renowned scholar, proud of his abilities, visited the court.\nReprint 2026-27"),
]

LLM_REPLY = {
    "topics": [
        {"title": "The Wit that Won Hearts", "page": 0, "text_end_page": 5, "subtopics": []},
        {"title": "A Concrete Example", "page": 16, "text_end_page": 18, "subtopics": []},
        {
            "title": "Wisdom Paves the Way",
            "page": 26,
            "text_end_page": 35,
            "subtopics": [
                {"title": "Scene I: The Road to Ujjain", "page": 26},
                {"title": "Scene II: The King's Court", "page": 32},
            ],
        },
    ],
    "end_matter_page": 45,
}


@pytest.fixture(autouse=True)
def _fresh_cache():
    co.clear_memory_cache()
    yield
    co.clear_memory_cache()


def test_skeleton_drops_running_headers_and_keeps_titles():
    skeleton, lines = co.build_skeleton(POORVI)
    titles = [t for _, t in lines]
    assert "The Wit that Won Hearts" in titles
    assert "Scene II: The King's Court" in titles
    assert "Poorvi" not in titles  # running header on every page
    assert not any(t.startswith("Reprint") for t in titles)
    assert "p16: A Concrete Example" in skeleton


def test_collapse_repeated_decorative_titles():
    assert co._collapse_repeats("Dust of Sno Dust of Sno Dust of Sno Dust of Snow") == "Dust of Snow"
    assert co._collapse_repeats("Fire and Ice Fire and Ice Fire and Ice") == "Fire and Ice"
    assert co._collapse_repeats("The Wit that Won Hearts") == "The Wit that Won Hearts"


def test_validate_keeps_only_titles_from_the_text():
    _, lines = co.build_skeleton(POORVI)
    reply = {
        "topics": [
            {"title": "The Wit that Won Hearts", "page": 0},
            {"title": "The Rock Garden of Chandigarh", "page": 20},  # not in the chapter text
            {"title": "A Concrete Example", "page": 16},
        ],
        "end_matter_page": 45,
    }
    out = co.validate_outline(reply, lines, chapter_name="1: Wit and Wisdom")
    assert [t["title"] for t in out["topics"]] == ["The Wit that Won Hearts", "A Concrete Example"]
    assert out["end_matter_page"] == 45


def test_validate_tolerates_broken_spacing_and_rejects_sentences():
    lines = [(0, "The Witthat Won Hearts"), (19, "Vitamins are organic substances. They are micro nutrients")]
    reply = {
        "topics": [
            {"title": "The Wit that Won Hearts", "page": 0},
            {"title": "Vitamins are organic substances. They are micro nutrients", "page": 19},
        ]
    }
    out = co.validate_outline(reply, lines)
    assert [t["title"] for t in out["topics"]] == ["The Wit that Won Hearts"]


def test_chapter_title_allowed_only_for_single_reading_chapters():
    # "A Letter to God": the story title is a picture; its poems follow later.
    lines = [(0, "BEFORE YOU READ"), (13, "Dust of Snow"), (14, "Fire and Ice")]
    reply = {"topics": [{"title": "A Letter to God", "page": 0}, {"title": "Dust of Snow", "page": 13}]}
    out = co.validate_outline(reply, lines, chapter_name="1. A Letter to God")
    assert [t["title"] for t in out["topics"]] == ["A Letter to God", "Dust of Snow"]

    # A unit theme above its own readings is a label, not a topic.
    lines = [(0, "WIT AND WISDOM"), (0, "The Wit that Won Hearts"), (16, "A Concrete Example")]
    reply = {"topics": [{"title": "Wit and Wisdom", "page": 0}, {"title": "The Wit that Won Hearts", "page": 0}]}
    out = co.validate_outline(reply, lines, chapter_name="1: Wit and Wisdom")
    assert [t["title"] for t in out["topics"]] == ["The Wit that Won Hearts"]


def test_get_outline_calls_llm_once_and_caches():
    saved: dict = {}
    with patch.object(co, "_llm_pick", return_value=LLM_REPLY) as llm, \
         patch.object(co, "_load_cached", return_value=None), \
         patch.object(co, "_save_cached", side_effect=lambda uid, d: saved.update({uid: d})):
        a = co.get_chapter_outline(752, POORVI, chapter_name="1: Wit and Wisdom")
        b = co.get_chapter_outline(752, POORVI, chapter_name="1: Wit and Wisdom")
    assert llm.call_count == 1
    assert a is b
    assert [t["title"] for t in a["topics"]] == [
        "The Wit that Won Hearts",
        "A Concrete Example",
        "Wisdom Paves the Way",
    ]
    assert [s["title"] for s in a["topics"][2]["subtopics"]] == [
        "Scene I: The Road to Ujjain",
        "Scene II: The King's Court",
    ]
    assert "752" in saved and saved["752"]["fingerprint"] == co.fingerprint(POORVI)


def test_llm_failure_returns_none_and_is_not_retried_every_turn():
    with patch.object(co, "_llm_pick", side_effect=RuntimeError("429")) as llm, \
         patch.object(co, "_load_cached", return_value=None), \
         patch.object(co, "_save_cached") as save:
        assert co.get_chapter_outline(752, POORVI) is None
        assert co.get_chapter_outline(752, POORVI) is None
    assert llm.call_count == 1
    save.assert_not_called()


def test_reindex_changes_fingerprint():
    assert co.fingerprint(POORVI) != co.fingerprint(POORVI[:3])


def test_lesson_docs_drop_transcripts():
    kept = co.lesson_docs(POORVI, {"topics": [], "end_matter_page": 45})
    assert all(d.metadata["page"] < 45 for d in kept)
    assert len(kept) == len(POORVI) - 1


def test_excerpts_come_from_each_topic_not_activities_or_transcripts():
    outline = co.validate_outline(LLM_REPLY, co.build_skeleton(POORVI)[1])
    block = co.topic_excerpts_block(POORVI, outline)
    assert "yawning" in block  # the story's own text
    assert "Mrs. Jones" in block  # the poem
    assert "merchant" in block or "Ujjain" in block  # the play
    assert "renowned scholar" not in block  # teacher transcript (end matter)
    assert "Why does the queen believe" not in block  # exercise question
    assert "before we read" not in block.lower()


def test_numbered_chapters_keep_their_section_list():
    from app.modules.student_learning import topic_progress as tp

    chunks = [
        Document(page_content="1.1 Introduction to Real Numbers\nText", metadata={"page": 1}),
        Document(page_content="1.2 The Fundamental Theorem of Arithmetic\nText", metadata={"page": 2}),
        Document(page_content="1.3 Revisiting Irrational Numbers\nText", metadata={"page": 4}),
    ]
    with patch.object(tp, "_load_chapter_chunks", return_value=chunks), \
         patch("app.services.chapter_outline.get_chapter_outline") as outline:
        rows = tp.list_chapter_topics(chapter_id=652, subject_name="Mathematics", chapter_name="1. Real Numbers")
    outline.assert_not_called()
    assert len([r for r in rows if r.get("section_number")]) >= 2


def test_unnumbered_chapters_use_the_outline():
    from app.modules.student_learning import topic_progress as tp

    outline = co.validate_outline(LLM_REPLY, co.build_skeleton(POORVI)[1])
    with patch.object(tp, "_load_chapter_chunks", return_value=POORVI), \
         patch("app.services.chapter_outline.get_chapter_outline", return_value=outline):
        rows = tp.list_chapter_topics(
            chapter_id=752, subject_name="English -  Poorvi", chapter_name="1: Wit and Wisdom"
        )
    assert [r["title"] for r in rows] == [
        "The Wit that Won Hearts",
        "A Concrete Example",
        "Wisdom Paves the Way",
    ]
    assert all(set(r) == {"key", "title"} for r in rows)
