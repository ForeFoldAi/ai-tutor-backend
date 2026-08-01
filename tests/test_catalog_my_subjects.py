"""Catalog my-subjects class_level query param."""

from __future__ import annotations

from app.modules.auth.constants import Role
from app.modules.catalog.models import ClassEnum
from app.modules.catalog.router import _parse_class_level, _user_may_access_class


def test_parse_class_level():
    assert _parse_class_level("6") == ClassEnum.CLASS_6
    assert _parse_class_level("CLASS_8") == ClassEnum.CLASS_8
    assert _parse_class_level("") is None


def test_user_may_access_class_for_tutor():
    class _User:
        role = Role.TUTOR
        teaching_classes = [{"grade": "6", "sections": ["A"]}, {"grade": "8", "sections": ["B"]}]

    assert _user_may_access_class(_User(), ClassEnum.CLASS_6) is True
    assert _user_may_access_class(_User(), ClassEnum.CLASS_7) is False
