from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.teacher.assignments import service as svc
from app.modules.teacher.assignments.schemas import (
    AssignmentResultsResponse,
    CreateAssignmentRequest,
    CreateAssignmentsResponse,
    PatchAssignmentRequest,
    PreviewMatchRequest,
    PreviewMatchResponse,
    TutorAssignmentListItem,
    TutorAssignmentListResponse,
)
from app.modules.users.models import User

router = APIRouter(prefix="/api/tutor/assignments", tags=["tutor-assignments"])

TutorUser = Annotated[User, Depends(require_roles(Role.TUTOR))]
DbSession = Annotated[Session, Depends(get_db)]


@router.post("/preview-match", response_model=PreviewMatchResponse)
def preview_match(payload: PreviewMatchRequest, db: DbSession, current_user: TutorUser):
    return svc.preview_match(
        db,
        current_user,
        grade=payload.grade,
        section=payload.section,
        curriculum=payload.curriculum,
        subject=payload.subject,
    )


@router.post("", response_model=CreateAssignmentsResponse)
def create_assignments(payload: CreateAssignmentRequest, db: DbSession, current_user: TutorUser):
    return svc.create_assignments(db, current_user, payload)


@router.get("", response_model=TutorAssignmentListResponse)
def list_assignments(db: DbSession, current_user: TutorUser):
    return svc.list_tutor_assignments(db, current_user)


@router.get("/{assignment_id}/results", response_model=AssignmentResultsResponse)
def assignment_results(assignment_id: int, db: DbSession, current_user: TutorUser):
    return svc.get_assignment_results(db, current_user, assignment_id)


@router.patch("/{assignment_id}", response_model=TutorAssignmentListItem)
def patch_assignment(
    assignment_id: int,
    payload: PatchAssignmentRequest,
    db: DbSession,
    current_user: TutorUser,
):
    return svc.patch_assignment_deadline(db, current_user, assignment_id, payload)
