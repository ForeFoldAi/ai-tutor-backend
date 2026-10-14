from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.teacher.assignments import service as svc
from app.modules.teacher.assignments.schemas import (
    StudentAssignmentDetail,
    StudentAssignmentListResponse,
)
from app.modules.users.models import User

router = APIRouter(prefix="/api/student/assignments", tags=["student-assignments"])

StudentUser = Annotated[User, Depends(require_roles(Role.STUDENT))]
DbSession = Annotated[Session, Depends(get_db)]


class SubmitBody(BaseModel):
    answers: dict[str, Any] = Field(default_factory=dict)


@router.get("", response_model=StudentAssignmentListResponse)
def list_assignments(db: DbSession, current_user: StudentUser):
    return svc.list_student_assignments(db, current_user)


@router.get("/{assignment_id}", response_model=StudentAssignmentDetail)
def get_assignment(assignment_id: int, db: DbSession, current_user: StudentUser):
    return svc.get_student_assignment_detail(db, current_user, assignment_id)


@router.post("/{assignment_id}/start", response_model=StudentAssignmentDetail)
def start_assignment(assignment_id: int, db: DbSession, current_user: StudentUser):
    return svc.start_assignment(db, current_user, assignment_id)


@router.post("/{assignment_id}/submit", response_model=StudentAssignmentDetail)
def submit_assignment(
    assignment_id: int,
    payload: SubmitBody,
    db: DbSession,
    current_user: StudentUser,
):
    return svc.submit_assignment(db, current_user, assignment_id, payload.answers)
