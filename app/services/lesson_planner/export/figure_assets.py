"""Resolve textbook figure files for PPTX embedding."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_ICONS = frozenset({"engage", "example", "try", "check", "diagram", "formula", "remember"})


def normalize_icon(raw: str | None) -> str:
    key = (raw or "").strip().lower().replace(" ", "_")
    if key in _ICONS:
        return key
    for token in _ICONS:
        if token in key:
            return token
    return ""


def attach_figures_to_slides(
    slides: list[dict[str, Any]],
    figures: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Match slide figure refs to retrieved textbook figures; resolve disk paths."""
    figs = [f for f in (figures or []) if isinstance(f, dict)]
    if not figs:
        return slides

    used: set[int] = set()
    out: list[dict[str, Any]] = []
    for slide in slides:
        item = dict(slide)
        match = _match_figure(item, figs, used)
        if match is None and item.get("layout") == "figure":
            # First unused figure for dedicated figure slides.
            for i, fig in enumerate(figs):
                if i not in used:
                    match = fig
                    used.add(i)
                    break
        if match:
            path = resolve_figure_disk_path(match)
            item["figure_caption"] = str(
                item.get("figure_caption") or match.get("caption") or ""
            ).strip()[:200]
            item["figure_file"] = str(match.get("file_name") or item.get("figure_file") or "")
            if path:
                item["figure_path"] = path
        out.append(item)
    return out


def _match_figure(
    slide: dict[str, Any],
    figures: list[dict[str, Any]],
    used: set[int],
) -> dict[str, Any] | None:
    needle_file = str(slide.get("figure_file") or "").strip().lower()
    needle_cap = str(slide.get("figure_caption") or "").strip().lower()
    if not needle_file and not needle_cap:
        return None
    for i, fig in enumerate(figures):
        if i in used:
            continue
        fname = str(fig.get("file_name") or "").lower()
        caption = str(fig.get("caption") or "").lower()
        if needle_file and needle_file in fname:
            used.add(i)
            return fig
        if needle_cap and (needle_cap in caption or caption in needle_cap):
            used.add(i)
            return fig
        # Soft match: any significant word overlap with caption.
        if needle_cap:
            words = [w for w in needle_cap.replace(",", " ").split() if len(w) > 3]
            if words and sum(1 for w in words if w in caption) >= min(2, len(words)):
                used.add(i)
                return fig
    return None


def resolve_figure_disk_path(fig: dict[str, Any]) -> str | None:
    file_name = str(fig.get("file_name") or "").strip()
    upload_raw = fig.get("textbook_upload_id")
    if not file_name or upload_raw is None:
        return None
    try:
        upload_id = int(upload_raw)
    except (TypeError, ValueError):
        return None
    try:
        from app.services.image_service.textbook_image_extraction import image_disk_path

        path = image_disk_path(upload_id, file_name)
        if os.path.isfile(path):
            return path
    except Exception as exc:
        logger.debug("figure path resolve failed: %s", exc)
    return None
