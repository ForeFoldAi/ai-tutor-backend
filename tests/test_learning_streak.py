"""Streak decay + restart rules (no DB)."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

from app.modules.student_learning.service import _apply_streak_decay


def test_streak_alive_if_studied_yesterday() -> None:
    today = date(2026, 7, 22)
    streak = SimpleNamespace(current_streak=5, last_study_date=today - timedelta(days=1))
    assert _apply_streak_decay(streak, today=today) == 5
    assert streak.current_streak == 5


def test_streak_alive_if_studied_today() -> None:
    today = date(2026, 7, 22)
    streak = SimpleNamespace(current_streak=3, last_study_date=today)
    assert _apply_streak_decay(streak, today=today) == 3


def test_streak_resets_to_zero_after_missed_day() -> None:
    today = date(2026, 7, 22)
    streak = SimpleNamespace(current_streak=7, last_study_date=today - timedelta(days=2))
    assert _apply_streak_decay(streak, today=today) == 0
    assert streak.current_streak == 0


def test_streak_none_is_zero() -> None:
    assert _apply_streak_decay(None, today=date(2026, 7, 22)) == 0
