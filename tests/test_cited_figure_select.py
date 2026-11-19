"""Exact Fig N citation short-circuits LLM image select."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.services.chat_service.images import (
    _prefer_cited_figure_candidates,
    _select_related_images_for_answer,
    figure_hint_block_from_docs,
)
from app.services.image_service.image_intent_extractor import figure_numbers_cited_in_text


def test_figure_numbers_cited_in_text():
    assert figure_numbers_cited_in_text("Look at Fig. 2.3 and Figure 2.16") == ["2.3", "2.16"]
    assert figure_numbers_cited_in_text("no figures here") == []


def test_prefer_question_over_answer():
    cands = [
        {"figure_number": "2.16", "caption": "regional", "relevance": 90},
        {"figure_number": "3.4", "caption": "pathogen", "relevance": 50},
    ]
    out = _prefer_cited_figure_candidates(
        "show figure 3.4",
        "Look at Fig. 2.16 instead.",
        cands,
    )
    assert [x["figure_number"] for x in out] == ["3.4"]


def test_prefer_rag_over_answer_hallucination():
    docs = [
        SimpleNamespace(page_content="Fig 2.3 shows a political map snapshot for a period."),
        SimpleNamespace(page_content="Fig 2.12 and Fig 2.16 are other political map snapshots."),
    ]
    cands = [
        {"figure_number": "2.16", "caption": "regional powers", "relevance": 95},
        {"figure_number": "2.3", "caption": "political map", "relevance": 70},
    ]
    out = _prefer_cited_figure_candidates(
        "What was India's political map boundaries around 1947?",
        "Look at Fig. 2.16 — regional powers.",
        cands,
        retrieved_docs=docs,
    )
    # RAG cite order wins over answer's 2.16-first preference
    assert [x["figure_number"] for x in out] == ["2.3", "2.16"]


def test_select_short_circuits_without_llm():
    cands = [
        {"figure_number": "2.16", "caption": "wrong", "relevance": 99},
        {"figure_number": "3.4", "caption": "pathogen spread", "relevance": 40},
    ]
    out = asyncio.run(
        _select_related_images_for_answer(
            "can you show me figure 3.4",
            "I don't see Figure 3.4.",
            cands,
        )
    )
    assert [x["figure_number"] for x in out] == ["3.4"]


def test_figure_hint_lists_rag_figs():
    docs = [SimpleNamespace(page_content="See Fig. 3.4 for pathogen spread.")]
    block = figure_hint_block_from_docs(docs)
    assert "Fig. 3.4" in block
    assert "cannot see" in block.lower() or "never say you cannot" in block.lower()
