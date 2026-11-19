"""Guaranteed Fig N from answer cites / exact asks / chapter list-asks."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.services.chat_service.images import (
    _prefer_cited_figure_candidates,
    _select_related_images_for_answer,
    figure_inventory_text,
)
from app.services.image_service.image_intent_extractor import (
    figure_numbers_cited_in_text,
    is_chapter_figure_list_ask,
)


def test_figure_list_ask_detector():
    assert is_chapter_figure_list_ask("list the figures from this chapter")
    assert is_chapter_figure_list_ask("how many figures")
    assert is_chapter_figure_list_ask("get the images from the textbook")
    assert not is_chapter_figure_list_ask("what is weather?")
    assert not is_chapter_figure_list_ask("where is Vijayanagara")


def test_bring_figure_number_parsed():
    assert figure_numbers_cited_in_text("Bring the figure 2.11") == ["2.11"]
    assert figure_numbers_cited_in_text("get Fig. 2.12 please") == ["2.12"]


def test_vijayanagara_answer_cite_loads_from_db():
    """Answer cites Fig. 2.12 even when ranked candidates are empty/unrelated."""
    answer = (
        "Vijayanagara was the capital city of the powerful Vijayanagara Empire. "
        "According to the chapter’s map (Fig. 2.12), its ruins lie near Hampi."
    )
    db_hit = [
        {
            "figure_number": "2.12",
            "caption": "Vijayanagara empire map",
            "url": "/auth/catalog/textbook-images/1/figures/fig_2_12.jpg",
            "relevance": 100.0,
        }
    ]
    with patch(
        "app.services.image_service.textbook_image_retrieval.chapter_figures_by_number",
        return_value=db_hit,
    ) as mock_db:
        out = asyncio.run(
            _select_related_images_for_answer(
                "where is Vijayanagara",
                answer,
                [{"figure_number": "2.3", "caption": "other", "relevance": 40}],
                chapter_ids=["42"],
            )
        )
    assert out[0]["figure_number"] == "2.12"
    assert any(x["figure_number"] == "2.12" for x in out)
    mock_db.assert_called_once()
    assert mock_db.call_args.args[1] == ["2.12"]


def test_vijayanagara_does_not_need_llm_select():
    answer = "See the chapter map (Fig. 2.12) near Hampi."
    db_hit = [{"figure_number": "2.12", "caption": "map", "url": "/u/2.12", "relevance": 100}]
    with (
        patch(
            "app.services.image_service.textbook_image_retrieval.chapter_figures_by_number",
            return_value=db_hit,
        ),
        patch(
            "app.services.image_service.llm_image_select.select_images_for_qa",
        ) as mock_llm,
    ):
        out = asyncio.run(
            _select_related_images_for_answer(
                "where is Vijayanagara",
                answer,
                [],
                chapter_ids=["1"],
            )
        )
    assert out[0]["figure_number"] == "2.12"
    mock_llm.assert_not_called()


def test_question_fig_db_beats_answer():
    with patch(
        "app.services.image_service.textbook_image_retrieval.chapter_figures_by_number",
        side_effect=lambda _ids, nums, max_n=8: [
            {"figure_number": n, "caption": n, "url": f"/u/{n}", "relevance": 100}
            for n in nums
        ],
    ):
        out = asyncio.run(
            _select_related_images_for_answer(
                "Bring the figure 2.11",
                "Look at Fig. 2.16 instead.",
                [],
                chapter_ids=["1"],
            )
        )
    assert [x["figure_number"] for x in out] == ["2.11"]


def test_list_ask_uses_catalog():
    catalog = [
        {"figure_number": "2.1", "caption": "a", "url": "/u/1", "relevance": 100},
        {"figure_number": "2.2", "caption": "b", "url": "/u/2", "relevance": 100},
    ]
    with (
        patch(
            "app.services.image_service.textbook_image_retrieval.chapter_figures_catalog",
            return_value=catalog,
        ),
        patch(
            "app.services.image_service.llm_image_select.select_images_for_qa",
        ) as mock_llm,
    ):
        out = asyncio.run(
            _select_related_images_for_answer(
                "list the figures from this chapter",
                "Here are some figures.",
                [{"figure_number": "9.9", "caption": "noise", "relevance": 99}],
                chapter_ids=["1"],
            )
        )
    assert [x["figure_number"] for x in out] == ["2.1", "2.2"]
    mock_llm.assert_not_called()
    text = figure_inventory_text(out)
    assert "Fig. 2.1" in text and "Fig. 2.2" in text


def test_no_fig_cite_skips_db_lookup():
    """Normal teaching question with no Fig N must not hit DB-by-number path."""
    with (
        patch(
            "app.services.image_service.textbook_image_retrieval.chapter_figures_by_number",
        ) as mock_db,
        patch(
            "app.services.image_service.textbook_image_retrieval.chapter_figures_catalog",
        ) as mock_cat,
        patch(
            "app.services.chat_service.images.ENABLE_LLM_IMAGE_SELECT",
            False,
        ),
    ):
        cands = [{"figure_number": "2.1", "caption": "weather", "relevance": 55}]
        out = asyncio.run(
            _select_related_images_for_answer(
                "what is weather?",
                "Weather is the day-to-day state of the atmosphere.",
                cands,
                chapter_ids=["1"],
            )
        )
    mock_db.assert_not_called()
    mock_cat.assert_not_called()
    assert out == cands


def test_list_ask_skips_figure_line_strip():
    import asyncio as _asyncio

    from app.services.chat_service.postprocess import _finalize_direct_answer

    inventory = "Here are the textbook figures from this chapter:\nFig. 2.1 — Weather map\n"
    with patch(
        "app.services.chat_service.postprocess._structure_tier",
        return_value="direct",
    ), patch(
        "app.services.chat_service.postprocess._direct_answer_needs_shrink",
        return_value=False,
    ):
        out = _asyncio.run(
            _finalize_direct_answer(
                [],
                inventory,
                "list the figures from this chapter",
                answer_type="direct",
            )
        )
    assert "Fig. 2.1 — Weather map" in out


def test_answer_cite_missing_db_falls_through():
    with patch(
        "app.services.image_service.textbook_image_retrieval.chapter_figures_by_number",
        return_value=[],
    ):
        out = _prefer_cited_figure_candidates(
            "where is Vijayanagara",
            "According to the chapter map (Fig. 2.12).",
            [{"figure_number": "2.3", "caption": "other", "relevance": 40}],
        )
    assert out is None


def test_multi_cite_in_answer():
    with patch(
        "app.services.image_service.textbook_image_retrieval.chapter_figures_by_number",
        side_effect=lambda _ids, nums, max_n=8: [
            {"figure_number": n, "caption": n, "url": f"/u/{n}", "relevance": 100}
            for n in nums
            if n in {"2.12", "2.16"}
        ],
    ):
        out = asyncio.run(
            _select_related_images_for_answer(
                "tell me about the maps",
                "See Fig. 2.12 and Fig. 2.16.",
                [],
                chapter_ids=["1"],
            )
        )
    assert [x["figure_number"] for x in out] == ["2.12", "2.16"]
