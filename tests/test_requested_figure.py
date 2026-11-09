"""A figure the student names ("get me figure 2.12") is grounded on its real caption and page."""
from app.services.chat_service.images import requested_figure_prompt_block
from app.services.image_service.image_intent_extractor import (
    figure_numbers_cited_in_text,
    normalize_spoken_figure_refs,
)


def test_spoken_figure_numbers_resolve_to_textbook_form():
    assert figure_numbers_cited_in_text("Can you get me the figure 2.12?") == ["2.12"]
    assert figure_numbers_cited_in_text("show figure 2 point 12") == ["2.12"]
    assert figure_numbers_cited_in_text("show figure two point one two") == ["2.12"]
    assert figure_numbers_cited_in_text("open fig 2-12 please") == ["2.12"]
    assert figure_numbers_cited_in_text("Figure 2, 12 please") == ["2.12"]


def test_spoken_normalization_leaves_other_text_alone():
    assert normalize_spoken_figure_refs("figure 2, one more thing") == "figure 2, one more thing"
    assert normalize_spoken_figure_refs("What is a map?") == "What is a map?"


def test_prompt_block_carries_real_page_and_on_screen_note():
    fig = {
        "figure_number": "2.12",
        "caption": "Kingdoms in the Deccan and the Vijayanagara Empire",
        "page": 13,
        "visual_description": "A map of the Deccan kingdoms.",
    }
    block = requested_figure_prompt_block(["2.12"], [fig])
    assert "textbook page 13" in block
    assert "ALREADY DISPLAYED" in block
    assert "use exactly: 13" in block


def test_prompt_block_flags_missing_figure():
    block = requested_figure_prompt_block(["9.9"], [])
    assert "Fig. 9.9 is not in this chapter" in block
    assert "ALREADY DISPLAYED" not in block
