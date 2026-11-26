"""Whole-chapter walkthroughs taught in parts ("explain this chapter" + Continue).

The single-shot overview prompt was ~7.4k tokens and Groq rejected it (per-key TPM cap),
so long chapters are taught a few topics at a time with a smaller raw-page budget.
"""
from __future__ import annotations

from langchain_core.documents import Document

from app.services import chapter_outline as co
from app.services.chat_service.dialogue import _chapter_meta_prompt_override
from app.services.chat_service.qa import _overview_doc_budget
from app.modules.student_learning.topic_progress import chapter_toc_context_block

_OUTLINE = {
    "topics": [
        {"title": "The Wit that Won Hearts", "page": 1},
        {"title": "A Concrete Example", "page": 16},
        {"title": "A Sea of Foliage", "page": 25},
        {"title": "Wisdom Paves the Way", "page": 26},
    ],
    "end_matter_page": 40,
}

_STORY = (
    "The King of the land was a wise ruler who loved poetry and music very much. "
    "One day the Queen yawned while he read his poem aloud to the whole court. "
    "The clever courtier found a gentle way to make the King see his mistake. "
)


def _doc(page: int, text: str = _STORY) -> Document:
    return Document(page_content=text, metadata={"page": page})


def test_excerpts_only_cover_the_requested_part():
    chunks = [_doc(p) for p in (1, 16, 25, 26)]
    block = co.topic_excerpts_block(chunks, _OUTLINE, only=range(2, 4))
    assert "A Sea of Foliage" in block and "Wisdom Paves the Way" in block
    assert "The Wit that Won Hearts" not in block and "A Concrete Example" not in block


def test_part_page_range():
    assert co.part_page_range(_OUTLINE, 0, 2) == (1, 25)
    assert co.part_page_range(_OUTLINE, 2, 4) == (25, 40)


def test_toc_numbers_continue_in_later_parts():
    block = chapter_toc_context_block(_OUTLINE["topics"][2:4], start=3)
    assert "3. A Sea of Foliage" in block and "4. Wisdom Paves the Way" in block


def test_part_prompt_offers_continue_until_last_part():
    first = _chapter_meta_prompt_override("explain this chapter", (1, 2))
    assert "PART 1 OF 2" in first["instruction"]
    assert "Continue for part 2" in first["interaction_policy"]
    last = _chapter_meta_prompt_override("explain this chapter", (2, 2))
    assert "Continue" not in last["interaction_policy"]
    # Unsplit chapters keep the whole-chapter wording.
    whole = _chapter_meta_prompt_override("explain this chapter")
    assert "WHOLE chapter" in whole["instruction"]


def test_overview_raw_page_budget_is_capped():
    assert _overview_doc_budget({}, 14000) == 14000
    assert _overview_doc_budget({"overview": True, "excerpts": True}, 14000) <= 3000
    assert _overview_doc_budget({"overview": True, "excerpts": False}, 14000) <= 6000
