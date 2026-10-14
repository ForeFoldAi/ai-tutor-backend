from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    entity: str
    id: str
    title: str
    subtitle: str | None = None
    href: str


class GlobalSearchResponse(BaseModel):
    q: str
    hits: list[SearchHit] = Field(default_factory=list)


class EntitySearchResponse(BaseModel):
    """Tab search: all matching rows for one entity (no page window)."""

    entity: str
    q: str
    total: int
    items: list[dict[str, Any]] = Field(default_factory=list)
