"""Generic subtopic figure scoring (no subject-specific tables)."""

from types import SimpleNamespace

from app.services.section_heading import SubtopicInfo
from app.services.subtopic_figure_match import (
    is_panel_subfigure,
    score_image_for_subtopic,
)


def test_panel_subfigure_detection():
    assert is_panel_subfigure("2.4.1")
    assert not is_panel_subfigure("2.6")


def test_activity_figure_scores_low_for_subtopic():
    im = SimpleNamespace(
        caption="y in Madhya e recorded in",
        figure_number="2.5",
        page_index=5,
        page_text_snippet="",
        figure_context="",
        is_decorative=False,
        educational_salience=0.5,
        upload=SimpleNamespace(
            file_path="uploads/CBSE/CLASS_9/Social/c75c7c82_gees102.pdf"
        ),
    )
    sub = SubtopicInfo(title="Temperature", letter="a", figure_numbers=["2.5"], pages=[4, 5])
    assert score_image_for_subtopic(im, sub) < 0
