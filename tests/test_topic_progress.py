"""Topic-coverage progress — match + topics-left answers."""

from __future__ import annotations

from app.modules.student_learning.topic_progress import (
    _is_usable_topic_title,
    _recompute_pct,
    format_topics_left_answer,
    is_topics_left_query,
    match_topic_keys_for_query,
    topic_key,
)


def test_topics_left_query_detection():
    assert is_topics_left_query("How many topics left?")
    assert is_topics_left_query("what topics are remaining in this chapter")
    assert not is_topics_left_query("what is the monsoon")


def test_match_topic_from_question():
    topics = [
        {"key": topic_key("The Monsoons"), "title": "The Monsoons"},
        {"key": topic_key("Climate Change"), "title": "Climate Change"},
        {"key": topic_key("Factors Determining the Climate"), "title": "Factors Determining the Climate"},
    ]
    keys = match_topic_keys_for_query("Explain the monsoons", topics)
    assert topic_key("The Monsoons") in keys


def test_progress_pct_bounds():
    assert _recompute_pct(10, ["a"], completed=False) == 10
    assert _recompute_pct(4, ["a", "b"], completed=False) == 50
    assert _recompute_pct(4, ["a", "b", "c", "d"], completed=True) == 100


def test_noise_topic_filter():
    assert not _is_usable_topic_title("LET'S REMEMBER")
    assert not _is_usable_topic_title("LET’S REMEMBER")  # curly apostrophe
    assert not _is_usable_topic_title("DON’T MISS OUT")
    assert not _is_usable_topic_title("Chapter")
    assert not _is_usable_topic_title("Mahabaleshwar, Mount Abu, Shimla, Nainital")
    # Cyclone / disaster pedagogy signal boxes — not chapter topics
    assert not _is_usable_topic_title("No Warning")
    assert not _is_usable_topic_title("Warning (Take Action)")
    assert not _is_usable_topic_title("Watch (Be Updated)")
    assert _is_usable_topic_title("The Monsoons")
    assert _is_usable_topic_title("Climate Change")
    assert not _is_usable_topic_title("The Big")  # too vague / incomplete


def test_topics_left_answer_lists_remaining():
    ans = format_topics_left_answer(
        {
            "topics_total": 3,
            "progress_pct": 33,
            "covered": [{"key": "a", "title": "Rain"}],
            "remaining": [
                {"key": "b", "title": "Wind"},
                {"key": "c", "title": "Clouds"},
            ],
        },
        chapter_label="Weather",
    )
    assert "Topics left (2)" in ans
    assert "1. Wind" in ans
    assert "Clouds" in ans
