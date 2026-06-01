import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import get_settings
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import get_current_user, require_roles
from app.modules.auth.schemas import (
    AdminCreateUserRequest,
    CreateStudentRequest,
    CreateTutorRequest,
    ForgotPasswordRequest,
    LoginRequest,
    LogoutRequest,
    MeProfileUpdateRequest,
    MessageResponse,
    OrganizationDetailResponse,
    OrganizationSignupRequest,
    OrganizationUpdateRequest,
    RefreshRequest,
    ResetPasswordRequest,
    SchoolSummaryResponse,
    SchoolUpdateRequest,
    StudentSignupRequest,
    TokenResponse,
    UpdateStudentRequest,
    UpdateTutorRequest,
    UserResponse,
    UserStatusPatchRequest,
)
from app.modules.auth.service import (
    admin_create_user,
    build_auth_tokens,
    create_school_admin,
    delete_school,
    delete_student_user,
    delete_tutor_user,
    get_organization_for_org_admin,
    issue_reset_password_token,
    issue_verify_email_token,
    list_schools_with_stats,
    list_tutor_assigned_students,
    list_users,
    login,
    logout,
    refresh_access_token,
    reset_password,
    set_user_status,
    signup_organization,
    signup_student,
    update_me_profile,
    update_organization_for_org_admin,
    update_school,
    update_student,
    update_tutor,
    verify_email_token,
)
from app.modules.users.models import User

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


class CreateSchoolAdminRequest(AdminCreateUserRequest):
    """Create a new school + first admin, or add another admin to ``school_id``."""

    school_id: uuid.UUID | None = None
    school_name: str | None = Field(default=None, max_length=255)
    branch: str | None = Field(default=None, max_length=255)
    board: str | None = Field(default=None, max_length=100)

    @field_validator("branch", "board", mode="before")
    @classmethod
    def _empty_optional_str(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s if s else None

    @field_validator("school_name", mode="before")
    @classmethod
    def _strip_school_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @model_validator(mode="after")
    def _school_name_when_new(self) -> CreateSchoolAdminRequest:
        if self.school_id is None:
            sn = self.school_name or ""
            if len(sn) < 2:
                raise ValueError("school_name is required (min 2 characters) when school_id is not provided.")
        return self


@router.post("/signup/student", response_model=MessageResponse, status_code=201)
def signup_student_route(payload: StudentSignupRequest, db: Annotated[Session, Depends(get_db)]):
    user = signup_student(db, payload)
    verify_token = issue_verify_email_token(user)
    db.commit()
    return MessageResponse(
        message=f"Student signup successful. Verification token generated (hook): {verify_token[:16]}..."
    )


@router.post("/signup/organization", response_model=MessageResponse, status_code=201)
def signup_organization_route(payload: OrganizationSignupRequest, db: Annotated[Session, Depends(get_db)]):
    user = signup_organization(db, payload)
    verify_token = issue_verify_email_token(user)
    db.commit()
    return MessageResponse(
        message=f"Organization signup successful. Verification token generated (hook): {verify_token[:16]}..."
    )


@router.post("/login", response_model=TokenResponse)
def login_route(payload: LoginRequest, request: Request, db: Annotated[Session, Depends(get_db)]):
    _user, access_token, refresh_token = login(db, payload, request)
    db.commit()
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_exp_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh_route(payload: RefreshRequest, db: Annotated[Session, Depends(get_db)]):
    _user, access_token, refresh_token = refresh_access_token(db, payload.refresh_token)
    db.commit()
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_exp_minutes * 60,
    )


@router.post("/logout", response_model=MessageResponse)
def logout_route(
    payload: LogoutRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    logout(db, current_user, payload.refresh_token, payload.all_devices)
    db.commit()
    return MessageResponse(message="Logged out successfully.")


@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(payload: ForgotPasswordRequest, db: Annotated[Session, Depends(get_db)]):
    # no user existence leakage
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if user:
        token = issue_reset_password_token(user)
        _ = token  # hook for email sender integration
    return MessageResponse(message="If the account exists, reset instructions have been issued.")


@router.post("/reset-password", response_model=MessageResponse)
def reset_password_route(payload: ResetPasswordRequest, db: Annotated[Session, Depends(get_db)]):
    reset_password(db, payload.token, payload.new_password)
    db.commit()
    return MessageResponse(message="Password reset successful.")


@router.get("/verify-email", response_model=MessageResponse)
def verify_email(token: str, db: Annotated[Session, Depends(get_db)]):
    verify_email_token(db, token)
    db.commit()
    return MessageResponse(message="Email verified successfully.")


@router.get("/me", response_model=UserResponse)
def me(current_user: Annotated[User, Depends(get_current_user)]):
    return UserResponse.model_validate(current_user)


@router.patch("/me/profile", response_model=UserResponse)
def patch_me_profile(
    payload: MeProfileUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    user = update_me_profile(db, current_user, payload)
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.get("/admin/organization", response_model=OrganizationDetailResponse)
def get_my_organization(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN))],
):
    org = get_organization_for_org_admin(db, current_user)
    return OrganizationDetailResponse.model_validate(org)


@router.patch("/admin/organization", response_model=OrganizationDetailResponse)
def patch_my_organization(
    payload: OrganizationUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN))],
):
    org = update_organization_for_org_admin(db, current_user, payload)
    db.commit()
    db.refresh(org)
    return OrganizationDetailResponse.model_validate(org)


@router.post("/admin/create-school-admin", response_model=UserResponse, status_code=201)
def create_school_admin_route(
    payload: CreateSchoolAdminRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.MASTER_ADMIN))],
):
    user = create_school_admin(
        db,
        current_user,
        payload,
        payload.school_name,
        payload.board,
        branch=payload.branch,
        school_id=payload.school_id,
    )
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.post("/admin/create-tutor", response_model=UserResponse, status_code=201)
def create_tutor_route(
    payload: CreateTutorRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
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
    return UserResponse.model_validate(user)


@router.post("/admin/create-student", response_model=UserResponse, status_code=201)
def create_student_route(
    payload: CreateStudentRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.SCHOOL_ADMIN, Role.MASTER_ADMIN, Role.TUTOR))],
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
    return UserResponse.model_validate(user)


@router.get("/admin/schools", response_model=list[SchoolSummaryResponse])
def admin_schools(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
    organization_id: Annotated[uuid.UUID | None, Query(description="Master admin: filter by organization.")] = None,
):
    schools = list_schools_with_stats(db, current_user, organization_id=organization_id)
    return schools


@router.patch("/admin/schools/{school_id}", response_model=SchoolSummaryResponse)
def patch_school(
    school_id: uuid.UUID,
    payload: SchoolUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.MASTER_ADMIN))],
):
    summary = update_school(db, current_user, school_id, payload)
    db.commit()
    return summary


@router.delete("/admin/schools/{school_id}", response_model=MessageResponse)
def remove_school(
    school_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.MASTER_ADMIN))],
):
    delete_school(db, current_user, school_id)
    db.commit()
    return MessageResponse(message="School removed.")


@router.get("/admin/users", response_model=list[UserResponse])
def admin_users(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
):
    users = list_users(db, current_user)
    return [UserResponse.model_validate(u) for u in users]


@router.get("/tutor/students", response_model=list[UserResponse])
def tutor_students(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.TUTOR))],
):
    students = list_tutor_assigned_students(db, current_user)
    return [UserResponse.model_validate(u) for u in students]


@router.patch("/admin/users/{user_id}/status", response_model=UserResponse)
def patch_user_status(
    user_id: uuid.UUID,
    payload: UserStatusPatchRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
):
    user = set_user_status(db, current_user, user_id, payload.is_active)
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.patch("/admin/users/{user_id}/tutor", response_model=UserResponse)
def patch_tutor_user(
    user_id: uuid.UUID,
    payload: UpdateTutorRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
):
    user = update_tutor(db, current_user, user_id, payload)
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.delete("/admin/users/{user_id}/tutor", response_model=MessageResponse)
def remove_tutor_user(
    user_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
):
    delete_tutor_user(db, current_user, user_id)
    db.commit()
    return MessageResponse(message="Tutor removed.")


@router.patch("/admin/users/{user_id}/student", response_model=UserResponse)
def patch_student_user(
    user_id: uuid.UUID,
    payload: UpdateStudentRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
):
    user = update_student(db, current_user, user_id, payload)
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.delete("/admin/users/{user_id}/student", response_model=MessageResponse)
def remove_student_user(
    user_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.ORG_ADMIN, Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
):
    delete_student_user(db, current_user, user_id)
    db.commit()
    return MessageResponse(message="Student removed.")
