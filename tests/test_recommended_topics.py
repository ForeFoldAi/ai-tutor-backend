"""Recommended topics pick remaining / next chapters (no DB)."""

from __future__ import annotations

from types import SimpleNamespace

from app.modules.catalog.models import BoardEnum, ClassEnum
from app.modules.student_learning.schemas import LearningChapterOut, LearningSubjectOut
from app.modules.student_learning.service import _build_recommended_topics


def test_recommends_next_not_started_chapter() -> None:
    subjects = [
        LearningSubjectOut(
            id=1,
            board=BoardEnum.CBSE,
            class_level=ClassEnum.CLASS_6,
            subject_name="Science",
            chapters=[
                LearningChapterOut(
                    id=10,
                    chapter="Climates of India",
                    file_name="climates.pdf",
                    status="not_started",
                )
            ],
            total_chapters=1,
        )
    ]
    recs = _build_recommended_topics(subjects, {})
    assert len(recs) == 1
    assert recs[0].title == "Climates of India"
    assert "Science" in recs[0].reason


def test_prefers_in_progress_chapter_when_no_topics() -> None:
    subjects = [
        LearningSubjectOut(
            id=2,
            board=BoardEnum.CBSE,
            class_level=ClassEnum.CLASS_6,
            subject_name="Mathematics",
            chapters=[
                LearningChapterOut(
                    id=20,
                    chapter="Fractions",
                    file_name="fractions.pdf",
                    status="in_progress",
                    progress=40,
                ),
                LearningChapterOut(
                    id=21,
                    chapter="Decimals",
                    file_name="decimals.pdf",
                    status="not_started",
                ),
            ],
            total_chapters=2,
            status="in_progress",
            progress=20,
        )
    ]
    # Empty progress → list_chapter_topics may return [] → falls back to chapter title
    progress = {20: SimpleNamespace(covered_topics=[])}
    recs = _build_recommended_topics(subjects, progress)  # type: ignore[arg-type]
    assert recs
    assert recs[0].chapter_id == 20


def test_interleaves_across_multiple_subjects() -> None:
    s1 = LearningSubjectOut(
        id=1,
        board=BoardEnum.CBSE,
        class_level=ClassEnum.CLASS_6,
        subject_name="Science",
        chapters=[
            LearningChapterOut(id=10, chapter="Plants", file_name="p.pdf", status="not_started"),
            LearningChapterOut(id=11, chapter="Animals", file_name="a.pdf", status="not_started"),
        ],
        total_chapters=2,
    )
    s2 = LearningSubjectOut(
        id=2,
        board=BoardEnum.CBSE,
        class_level=ClassEnum.CLASS_6,
        subject_name="Mathematics",
        chapters=[
            LearningChapterOut(id=20, chapter="Algebra", file_name="alg.pdf", status="not_started"),
        ],
        total_chapters=1,
    )
    recs = _build_recommended_topics([s1, s2], {}, limit=2)
    assert len(recs) == 2
    subjects_in_recs = [r.subject_name for r in recs]
    assert "Science" in subjects_in_recs
    assert "Mathematics" in subjects_in_recs
