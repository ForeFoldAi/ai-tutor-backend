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
    assert not _is_usable_topic_title("by changing the direction")


def test_unnumbered_chapter_keeps_ideas_and_drops_map_labels():
    from app.modules.student_learning.topic_progress import is_unnumbered_teaching_heading

    chapter = "Chapter 1 - Natural Resources and Their Use"
    assert is_unnumbered_teaching_heading("Categories of Natural Resources", chapter_name=chapter)
    assert is_unnumbered_teaching_heading("Distribution of Natural Resources", chapter_name=chapter)
    assert is_unnumbered_teaching_heading("Resources: Stewardship", chapter_name=chapter)
    assert is_unnumbered_teaching_heading(
        "Responsible and Wise Use of Natural", chapter_name=chapter
    )
    assert not is_unnumbered_teaching_heading("and Their Use", chapter_name=chapter)
    assert not is_unnumbered_teaching_heading("Natural Resources", chapter_name=chapter)
    assert not is_unnumbered_teaching_heading("Mumbai High", chapter_name=chapter)
    assert not is_unnumbered_teaching_heading("Iron Ore", chapter_name=chapter)
    assert not is_unnumbered_teaching_heading("Fatehgarh Sahib", chapter_name=chapter)
    assert not is_unnumbered_teaching_heading("SBS Nagar", chapter_name=chapter)
    assert not is_unnumbered_teaching_heading("Taran Tarn", chapter_name=chapter)
    assert not is_unnumbered_teaching_heading("Yarlung Tsangpe", chapter_name=chapter)
    assert not is_unnumbered_teaching_heading(
        "Tom Rivett-Carnac in ‘The Future We Choose’", chapter_name=chapter
    )
    assert not is_unnumbered_teaching_heading(
        "A Note on History's Darker Periods", chapter_name=chapter
    )


def test_topics_missing_from_text_ignores_topics_already_named():
    from app.modules.student_learning.topic_progress import topics_missing_from_text

    topics = [
        {"title": "Electromagnets", "section_number": "4.1.1"},
        {"title": "How Does a Battery Generate Electricity?", "section_number": "4.3"},
    ]
    said = "4.1.1 Electromagnets are temporary magnets made with a coil and an iron core."
    missing = topics_missing_from_text(topics, said)
    assert [t["title"] for t in missing] == ["How Does a Battery Generate Electricity?"]
    assert topics_missing_from_text(topics, said + " How Does a Battery Generate Electricity?") == []
    assert not _is_usable_topic_title("Exercises")
    assert not _is_usable_topic_title("Questions")
    assert not _is_usable_topic_title("Summary")


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
