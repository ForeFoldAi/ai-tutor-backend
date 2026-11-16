"""Activity N cites show the textbook diagram on screen (like Fig N)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.services.chat_service.images import (
    _select_related_images_for_answer,
    requested_activity_prompt_block,
)
from app.services.image_service.image_intent_extractor import (
    activity_numbers_cited_in_text,
    normalize_spoken_activity_refs,
)


def test_spoken_activity_numbers_resolve_to_textbook_form():
    assert activity_numbers_cited_in_text("Let's look at Activity 9.10") == ["9.10"]
    assert activity_numbers_cited_in_text("show activity 9 point 10") == ["9.10"]
    assert activity_numbers_cited_in_text("open activity nine point one zero") == ["9.10"]
    assert activity_numbers_cited_in_text("Activity 9-10 please") == ["9.10"]
    assert activity_numbers_cited_in_text("do Activity 3.1 now") == ["3.1"]


def test_spoken_activity_normalization_leaves_other_text_alone():
    assert normalize_spoken_activity_refs("activity time tomorrow") == "activity time tomorrow"
    assert normalize_spoken_activity_refs("What is refraction?") == "What is refraction?"


def test_activity_not_confused_with_figure():
    assert activity_numbers_cited_in_text("Look at Fig. 9.10") == []
    assert activity_numbers_cited_in_text("Activity 9.10 and Fig. 9.11") == ["9.10"]


def test_prompt_block_carries_on_screen_note():
    act = {
        "activity_number": "9.10",
        "caption": "Activity 9.10. Trace light through a glass slab",
        "page": 168,
        "visual_description": "Pins E F G H around a rectangular glass slab.",
    }
    block = requested_activity_prompt_block(["9.10"], [act])
    assert "textbook page 168" in block
    assert "ALREADY DISPLAYED" in block
    assert "use exactly: 168" in block
    assert "Activity 9.10" in block


def test_prompt_block_flags_missing_activity():
    block = requested_activity_prompt_block(["9.10"], [])
    assert "Activity 9.10 is not in this chapter" in block
    assert "ALREADY DISPLAYED" not in block


def test_answer_activity_cite_loads_from_db():
    answer = (
        "Let's look at Activity 9.10. You place pins E and F. Look through the opposite edge."
    )
    db_hit = [
        {
            "activity_number": "9.10",
            "caption": "Activity 9.10. Glass slab refraction",
            "url": "/auth/catalog/textbook-images/1/figures/activity_9_10.jpg",
            "relevance": 100.0,
        }
    ]
    with patch(
        "app.services.image_service.textbook_image_retrieval.chapter_activities_by_number",
        return_value=db_hit,
    ) as mock_db:
        out = asyncio.run(
            _select_related_images_for_answer(
                "explain refraction through a glass slab",
                answer,
                [{"figure_number": "9.3", "caption": "other", "relevance": 40}],
                chapter_ids=["42"],
            )
        )
    assert out[0]["activity_number"] == "9.10"
    mock_db.assert_called_once()
    assert mock_db.call_args.args[1] == ["9.10"]


def test_question_activity_db_beats_answer_figure():
    with (
        patch(
            "app.services.image_service.textbook_image_retrieval.chapter_activities_by_number",
            side_effect=lambda _ids, nums, max_n=8: [
                {
                    "activity_number": n,
                    "caption": f"Activity {n}",
                    "url": f"/u/a/{n}",
                    "relevance": 100,
                }
                for n in nums
            ],
        ),
        patch(
            "app.services.image_service.textbook_image_retrieval.chapter_figures_by_number",
            side_effect=lambda _ids, nums, max_n=8: [
                {
                    "figure_number": n,
                    "caption": f"Fig {n}",
                    "url": f"/u/f/{n}",
                    "relevance": 100,
                }
                for n in nums
            ],
        ),
    ):
        out = asyncio.run(
            _select_related_images_for_answer(
                "Show Activity 9.10",
                "Look at Fig. 9.5 instead.",
                [],
                chapter_ids=["1"],
            )
        )
    assert [x.get("activity_number") for x in out] == ["9.10"]
