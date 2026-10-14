"""Tutor home dashboard routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.teacher.dashboard.schemas import TutorDashboardSummaryResponse
from app.modules.teacher.dashboard.service import build_tutor_dashboard
from app.modules.users.models import User

router = APIRouter(prefix="/auth/tutor/dashboard", tags=["tutor-dashboard"])

TutorUser = Annotated[User, Depends(require_roles(Role.TUTOR))]


@router.get("", response_model=TutorDashboardSummaryResponse)
@router.get("/", response_model=TutorDashboardSummaryResponse, include_in_schema=False)
@router.get("/summary", response_model=TutorDashboardSummaryResponse)
def tutor_dashboard_summary(
    db: Annotated[Session, Depends(get_db)],
    current_user: TutorUser,
):
    result = build_tutor_dashboard(db, current_user)
    db.commit()
    return result
