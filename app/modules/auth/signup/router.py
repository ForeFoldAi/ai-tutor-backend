"""Public signup API routes."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.schools.models import School
from app.modules.users.models import User
from app.services.mail import send_signup_pending_admin_email
from app.modules.auth.service import issue_verify_email_token
from app.modules.auth.signup.constants import signup_options_payload
from app.modules.auth.signup.schemas import (
    SchoolSignupRequest,
    SignupOptionsResponse,
    SignupResponse,
    StudentSignupRequest,
    TutorSignupRequest,
)
from app.modules.auth.signup.service import (
    signup_catalog,
    signup_school,
    signup_student,
    signup_success_message,
    signup_tutor,
)

router = APIRouter(prefix="/auth/signup", tags=["signup"])


@router.get("/options", response_model=SignupOptionsResponse)
def signup_options_route():
    return SignupOptionsResponse(**signup_options_payload())


@router.get("/catalog")
def signup_catalog_route(
    curriculum: Annotated[str, Query(min_length=1, max_length=50)],
    grade: Annotated[str, Query(min_length=1, max_length=20)],
    db: Annotated[Session, Depends(get_db)],
):
    """Subjects + publishers available for a curriculum and grade (public, read-only)."""
    try:
        return signup_catalog(db, curriculum, grade)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


def _commit_signup(db: Session, user, login_user_id: str, background: BackgroundTasks) -> SignupResponse:
    verify_token = issue_verify_email_token(user)
    db.commit()
    db.refresh(user)
    school = db.get(School, user.school_id) if user.school_id else None
    background.add_task(
        send_signup_pending_admin_email,
        full_name=user.full_name,
        username=login_user_id.lower(),
        email=user.email,
        phone=user.phone,
        role_label=user.role.value.replace("_", " ").title(),
        school_name=school.name if school else None,
    )
    return SignupResponse(
        id=user.id,
        user_id=login_user_id.lower(),
        role=user.role.value,
        message=signup_success_message(user, verify_token),
    )


@router.post("/student", response_model=SignupResponse, status_code=201)
def signup_student_route(
    payload: StudentSignupRequest,
    db: Annotated[Session, Depends(get_db)],
    background: BackgroundTasks,
):
    user = signup_student(db, payload)
    return _commit_signup(db, user, payload.user_id, background)


@router.post("/tutor", response_model=SignupResponse, status_code=201)
def signup_tutor_route(
    payload: TutorSignupRequest,
    db: Annotated[Session, Depends(get_db)],
    background: BackgroundTasks,
):
    user = signup_tutor(db, payload)
    return _commit_signup(db, user, payload.user_id, background)


@router.post("/school", response_model=SignupResponse, status_code=201)
def signup_school_route(
    payload: SchoolSignupRequest,
    db: Annotated[Session, Depends(get_db)],
    background: BackgroundTasks,
):
    user = signup_school(db, payload)
    return _commit_signup(db, user, payload.user_id, background)


# ─── Master admin: same payloads as public signup, but pre-approved ───────────

admin_router = APIRouter(prefix="/auth/admin/signup", tags=["signup-admin"])

MasterAdminUser = Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))]


def _commit_admin_signup(db: Session, user: User, login_user_id: str, actor: User) -> SignupResponse:
    """Approve immediately (no pending-approval notice).

    Deliberately does NOT set ``created_by``: a student with ``created_by`` set is treated as
    tutor-owned (no favourite subjects, Session/Assignments visible), not as an individual student.
    """
    user.approved_at = datetime.now(UTC)
    db.commit()
    db.refresh(user)
    return SignupResponse(
        id=user.id,
        user_id=login_user_id.lower(),
        role=user.role.value,
        message=f"{user.role.value} account created and approved.",
    )


@admin_router.post("/student", response_model=SignupResponse, status_code=201)
def admin_signup_student_route(
    payload: StudentSignupRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: MasterAdminUser,
):
    user = signup_student(db, payload)
    return _commit_admin_signup(db, user, payload.user_id, current_user)


@admin_router.post("/tutor", response_model=SignupResponse, status_code=201)
def admin_signup_tutor_route(
    payload: TutorSignupRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: MasterAdminUser,
):
    user = signup_tutor(db, payload)
    return _commit_admin_signup(db, user, payload.user_id, current_user)


@admin_router.post("/school", response_model=SignupResponse, status_code=201)
def admin_signup_school_route(
    payload: SchoolSignupRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: MasterAdminUser,
):
    user = signup_school(db, payload)
    return _commit_admin_signup(db, user, payload.user_id, current_user)
