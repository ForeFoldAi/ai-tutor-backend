from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.school_admin.dashboard.schemas import DashboardSummaryResponse
from app.modules.school_admin.dashboard.service import build_dashboard_summary
from app.modules.users.models import User

router = APIRouter(prefix="/auth/admin/dashboard", tags=["school-admin-dashboard"])

SchoolAdminUser = Annotated[User, Depends(require_roles(Role.SCHOOL_ADMIN))]


@router.get("/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    return build_dashboard_summary(db, current_user)
