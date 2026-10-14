"""Public signup API routes."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.schools.models import School
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
