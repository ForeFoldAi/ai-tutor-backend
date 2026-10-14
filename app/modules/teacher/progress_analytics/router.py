"""Tutor progress analytics routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.teacher.progress_analytics.schemas import ProgressAnalyticsResponse
from app.modules.teacher.progress_analytics.service import build_progress_analytics
from app.modules.users.models import User

router = APIRouter(prefix="/auth/tutor/progress", tags=["tutor-progress"])

TutorUser = Annotated[User, Depends(require_roles(Role.TUTOR))]


@router.get("/analytics", response_model=ProgressAnalyticsResponse)
def tutor_progress_analytics(
    db: Annotated[Session, Depends(get_db)],
    current_user: TutorUser,
):
    result = build_progress_analytics(db, current_user)
    db.commit()
    return result
