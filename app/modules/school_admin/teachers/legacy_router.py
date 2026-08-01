from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.auth.public_ids import to_user_response
from app.modules.auth.schemas import (
    CreateTutorRequest,
    MessageResponse,
    UpdateTutorRequest,
    UserResponse,
)
from app.modules.auth.service import admin_create_user, delete_tutor_user, update_tutor
from app.modules.users.models import User

router = APIRouter(prefix="/auth/admin", tags=["school-admin-teachers"])

SchoolAdminUser = Annotated[User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))]


@router.post("/create-tutor", response_model=UserResponse, status_code=201)
def create_tutor_route(
    payload: CreateTutorRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    user = admin_create_user(
        db,
        current_user,
        Role.TUTOR,
        payload,
        teaching_board=payload.teaching_board,
        teaching_classes=payload.teaching_classes,
    )
    db.commit()
    db.refresh(user)
    return to_user_response(db, user)


@router.patch("/users/{user_id}/tutor", response_model=UserResponse)
def patch_tutor_user(
    user_id: int,
    payload: UpdateTutorRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    user = update_tutor(db, current_user, user_id, payload)
    db.commit()
    db.refresh(user)
    return to_user_response(db, user)


@router.delete("/users/{user_id}/tutor", response_model=MessageResponse)
def remove_tutor_user(
    user_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    delete_tutor_user(db, current_user, user_id)
    db.commit()
    return MessageResponse(message="Tutor removed.")
