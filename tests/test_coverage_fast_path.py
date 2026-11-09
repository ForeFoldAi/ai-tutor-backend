"""A chapter that already holds every question term skips the cross-chapter search."""

from types import SimpleNamespace
from unittest.mock import patch

from app.main import app  # noqa: F401 — app import order avoids circular imports
from app.services import chapter_scope as cs


def _doc(text: str):
    return SimpleNamespace(page_content=text, metadata={"textbook_upload_id": "7"})


def _assess(docs):
    return cs.assess_chapter_coverage(
        "What is horizontal power sharing?",
        docs=docs,
        collection_name="CBSE_10_Social",
        chapter_ids=["7"],
        chapter_names=["Power-sharing"],
        board="CBSE",
        class_level="10",
        subject_name="Social",
    )


def test_strong_match_skips_cross_chapter_search():
    docs = [_doc("Horizontal power sharing divides power among organs of government.")]
    with patch.object(cs, "_best_other_chapter") as other, patch.object(
        cs, "_terms_match_chapter_titles", return_value=False
    ):
        result = _assess(docs)
    other.assert_not_called()
    assert result.level == cs.ChapterCoverageLevel.FULL


def test_weak_match_still_checks_other_chapters():
    docs = [_doc("Federalism and the levels of government.")]
    with patch.object(cs, "_best_other_chapter", return_value=("", 0, 0)) as other, patch.object(
        cs, "_terms_match_chapter_titles", return_value=False
    ), patch.object(cs, "_subject_upload_labels", return_value={}):
        _assess(docs)
    other.assert_called_once()
