"""Public signup API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
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
    signup_school,
    signup_student,
    signup_success_message,
    signup_tutor,
)

router = APIRouter(prefix="/auth/signup", tags=["signup"])


@router.get("/options", response_model=SignupOptionsResponse)
def signup_options_route():
    return SignupOptionsResponse(**signup_options_payload())


def _commit_signup(db: Session, user, login_user_id: str) -> SignupResponse:
    verify_token = issue_verify_email_token(user)
    db.commit()
    db.refresh(user)
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
):
    user = signup_student(db, payload)
    return _commit_signup(db, user, payload.user_id)


@router.post("/tutor", response_model=SignupResponse, status_code=201)
def signup_tutor_route(
    payload: TutorSignupRequest,
    db: Annotated[Session, Depends(get_db)],
):
    user = signup_tutor(db, payload)
    return _commit_signup(db, user, payload.user_id)


@router.post("/school", response_model=SignupResponse, status_code=201)
def signup_school_route(
    payload: SchoolSignupRequest,
    db: Annotated[Session, Depends(get_db)],
):
    user = signup_school(db, payload)
    return _commit_signup(db, user, payload.user_id)
