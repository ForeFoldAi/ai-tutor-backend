from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.auth.public_ids import to_school_detail
from app.modules.auth.schemas import SchoolDetailResponse, SchoolProfileUpdateRequest
from app.modules.auth.service import get_school_for_school_admin, update_school_profile_for_admin
from app.modules.users.models import User

router = APIRouter(prefix="/auth/admin", tags=["school-admin-settings"])

SchoolAdminUser = Annotated[User, Depends(require_roles(Role.SCHOOL_ADMIN))]


@router.get("/school", response_model=SchoolDetailResponse)
def get_my_school(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    school = get_school_for_school_admin(db, current_user)
    return to_school_detail(school)


@router.patch("/school", response_model=SchoolDetailResponse)
def patch_my_school(
    payload: SchoolProfileUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    school = update_school_profile_for_admin(db, current_user, payload)
    db.commit()
    db.refresh(school)
    return to_school_detail(school)
