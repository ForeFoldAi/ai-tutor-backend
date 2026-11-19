"""Student home dashboard routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.student_dashboard.schemas import StudentDashboardResponse
from app.modules.student_dashboard.service import build_student_dashboard
from app.modules.users.models import User

router = APIRouter(prefix="/auth/student/dashboard", tags=["student-dashboard"])

StudentUser = Annotated[User, Depends(require_roles(Role.STUDENT))]


@router.get("", response_model=StudentDashboardResponse)
@router.get("/", response_model=StudentDashboardResponse, include_in_schema=False)
@router.get("/summary", response_model=StudentDashboardResponse)
def student_dashboard_summary(
    db: Annotated[Session, Depends(get_db)],
    current_user: StudentUser,
):
    result = build_student_dashboard(db, current_user)
    db.commit()
    return result
