from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.auth.public_ids import to_user_response
from app.modules.auth.schemas import (
    CreateStudentRequest,
    MessageResponse,
    UpdateStudentRequest,
    UserResponse,
)
from app.modules.auth.service import admin_create_user, delete_student_user, update_student
from app.modules.users.models import User

# Legacy paths kept for org-admin onboard/edit.
router = APIRouter(prefix="/auth/admin", tags=["school-admin-students-legacy"])

SchoolAdminUser = Annotated[User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))]
StudentCreator = Annotated[
    User,
    Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN, Role.TUTOR)),
]


@router.post("/create-student", response_model=UserResponse, status_code=201)
def create_student_route(
    payload: CreateStudentRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: StudentCreator,
):
    user = admin_create_user(
        db,
        current_user,
        Role.STUDENT,
        payload,
        teaching_board=payload.teaching_board,
        teaching_classes=payload.teaching_classes,
    )
    db.commit()
    db.refresh(user)
    return to_user_response(db, user)


@router.patch("/users/{user_id}/student", response_model=UserResponse)
def patch_student_user(
    user_id: int,
    payload: UpdateStudentRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    user = update_student(db, current_user, user_id, payload)
    db.commit()
    db.refresh(user)
    return to_user_response(db, user)


@router.delete("/users/{user_id}/student", response_model=MessageResponse)
def remove_student_user(
    user_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    delete_student_user(db, current_user, user_id)
    db.commit()
    return MessageResponse(message="Student removed.")
