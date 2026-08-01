from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.dependencies import require_active_user
from app.modules.search.schemas import EntitySearchResponse, GlobalSearchResponse
from app.modules.search.service import entity_search, global_search
from app.modules.users.models import User

router = APIRouter(prefix="/search", tags=["search"])

CurrentUser = Annotated[User, Depends(require_active_user)]


@router.get("", response_model=GlobalSearchResponse)
def search_global(
    db: Annotated[Session, Depends(get_db)],
    current_user: CurrentUser,
    q: Annotated[str | None, Query(description="Global search query")] = None,
):
    """Dashboard global search — role-scoped, multi-entity."""
    return global_search(db, current_user, q)


@router.get("/{entity}", response_model=EntitySearchResponse)
def search_entity(
    entity: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: CurrentUser,
    q: Annotated[str, Query(min_length=1, description="Entity search query")],
):
    """Tab search — one entity, all matching rows (no page window)."""
    return entity_search(db, current_user, entity, q)
