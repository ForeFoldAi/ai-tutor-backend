"""Scope keys for LIA partitioning."""

from __future__ import annotations


def build_scope_key(
    *,
    board: str = "",
    class_level: str = "",
    subject_name: str = "",
    chapter_ids: list[str] | None = None,
) -> str:
    ch = ",".join(sorted(chapter_ids or []))
    return f"{board}|{class_level}|{subject_name}|{ch}".strip("|")[:512]
