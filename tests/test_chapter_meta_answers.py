"""Whole-chapter / topic-list answers — voice + text quality without breaking narrow asks."""

from __future__ import annotations

from types import SimpleNamespace

from app.config import VOICE_PROMPT_CHAPTER_WORDS, VOICE_PROMPT_MAX_WORDS
from app.modules.student_learning.topic_progress import (
    format_chapter_topics_list_answer,
    is_chapter_topics_list_query,
    is_topics_left_query,
    topics_from_retrieved_docs,
)
from app.services.chapter_scope import is_current_lesson_query
from app.services.science_experiment.service import should_use_interactive_science_experiment
from app.services.section_retrieval import query_breadth
from app.services.voice_chapter_vocab import is_chapter_overview_request
from app.services.voice_prompts import voice_word_limit
from app.services.chat_service.dialogue import _chapter_meta_prompt_override


def test_topics_present_detected_as_list_not_left():
    q = "What topics are present in this chapter?"
    assert is_chapter_topics_list_query(q)
    assert not is_topics_left_query(q)
    assert is_current_lesson_query(q)
    assert query_breadth(q) == "chapter"


def test_topics_remaining_stays_left_query():
    q = "what topics are remaining in this chapter"
    assert is_topics_left_query(q)
    assert not is_chapter_topics_list_query(q)


def test_explain_whole_chapter_is_overview():
    assert is_chapter_overview_request("Explain the whole chapter")
    assert is_chapter_overview_request("explain this chapter from start to end")
    assert query_breadth("Explain the whole chapter") == "chapter"


def test_voice_word_cap_rises_for_chapter_meta_only():
    ctx = " ".join(["reflection refraction mirrors lenses prism"] * 40)
    assert voice_word_limit(query="what is refraction?", context=ctx) == VOICE_PROMPT_MAX_WORDS
    assert (
        voice_word_limit(query="Explain the whole chapter", context=ctx)
        == VOICE_PROMPT_CHAPTER_WORDS
    )
    assert (
        voice_word_limit(query="What topics are present in this chapter?", context=ctx)
        == VOICE_PROMPT_CHAPTER_WORDS
    )


def test_format_topics_list_text_and_spoken():
    topics = [
        {"key": "a", "title": "Reflection of light", "section_number": "9.1"},
        {"key": "b", "title": "Spherical mirrors", "section_number": "9.2"},
    ]
    text = format_chapter_topics_list_answer(topics, chapter_label="Light", spoken=False)
    assert "1. 9.1 Reflection of light" in text
    assert "2. 9.2 Spherical mirrors" in text
    assert "Which topic would you like to learn first?" in text
    assert "```" not in text

    spoken = format_chapter_topics_list_answer(topics, chapter_label="Light", spoken=True)
    assert "Topic 1: 9.1 Reflection of light." in spoken
    assert "Topic 2: 9.2 Spherical mirrors." in spoken
    assert "Which topic should we start with?" in spoken
    assert "\n" not in spoken.strip()


def test_topics_from_docs_parses_numbered_headings():
    docs = [
        SimpleNamespace(
            page_content=(
                "9.1 Reflection of light\n"
                "Light bounces off surfaces.\n"
                "9.2 Spherical mirrors\n"
                "Concave and convex.\n"
                "9.10 Dispersion of white light by a glass prism\n"
            ),
            metadata={"page": 1},
        )
    ]
    topics = topics_from_retrieved_docs(docs, chapter_name="Light")
    titles = [t["title"] for t in topics]
    assert "Reflection of light" in titles
    assert "Spherical mirrors" in titles
    assert "Dispersion of white light by a glass prism" in titles
    # Numeric order: 9.2 before 9.10
    nums = [t.get("section_number") for t in topics]
    assert nums.index("9.2") < nums.index("9.10")


def test_science_experiment_skipped_for_chapter_meta():
    assert not should_use_interactive_science_experiment(
        subject_name="Science",
        answer_type="short-answer",
        query="What topics are present in this chapter?",
        class_level="CLASS_10",
    )
    assert not should_use_interactive_science_experiment(
        subject_name="Science",
        answer_type="paragraph",
        query="Explain the whole chapter",
        class_level="CLASS_10",
    )


def test_chapter_meta_prompt_override_forbids_quiz_menu():
    ov = _chapter_meta_prompt_override("Explain the whole chapter")
    assert ov is not None
    assert "EVERY" in ov["instruction"] or "every" in ov["instruction"].lower()
    assert "quiz" in ov["instruction"].lower()

    lst = _chapter_meta_prompt_override("What topics are present in this chapter?")
    assert lst is not None
    assert "list" in lst["instruction"].lower() or "EVERY" in lst["instruction"]
