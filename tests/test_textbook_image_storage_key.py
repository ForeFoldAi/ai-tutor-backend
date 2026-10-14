"""Tests for curriculum-scoped textbook image storage keys."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.image_service.textbook_image_extraction import (
    image_storage_key,
    upload_asset_prefix,
)


def test_upload_asset_prefix_uses_curriculum_grade_subject_chapter():
    upload = SimpleNamespace(
        id=16,
        board=SimpleNamespace(value="CBSE"),
        class_level=SimpleNamespace(value="CLASS_9"),
        subject_name="Social",
        chapter="Chapter 2 - Understanding the Weather",
        content_label=None,
    )
    assert (
        upload_asset_prefix(upload)
        == "CBSE/CLASS_9/Social/Chapter_2_-_Understanding_the_Weather"
    )


def test_image_storage_key_semantic_and_legacy():
    upload = SimpleNamespace(
        id=16,
        board="CBSE",
        class_level="CLASS_9",
        subject_name="Social",
        chapter="Chapter 2 - Understanding the Weather",
        content_label=None,
    )
    assert image_storage_key(upload, "figures/fig_2_6.jpg") == (
        "CBSE/CLASS_9/Social/Chapter_2_-_Understanding_the_Weather/figures/fig_2_6.jpg"
    )
    assert image_storage_key(16, "figures/fig_2_6.jpg", legacy=True) == "16/figures/fig_2_6.jpg"
    assert image_storage_key(upload, "figures/fig_2_6.jpg", legacy=True) == "16/figures/fig_2_6.jpg"
