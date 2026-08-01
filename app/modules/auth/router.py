from typing import Annotated, Self

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import get_settings
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import get_current_user, require_roles
from app.modules.auth.schemas import (
    AdminCreateUserRequest,
    ForgotPasswordLookupRequest,
    ForgotPasswordLookupResponse,
    ForgotPasswordAccount,
    ForgotPasswordRequest,
    LoginRequest,
    LogoutRequest,
    MeProfileUpdateRequest,
    MessageResponse,
    RefreshRequest,
    ResetPasswordRequest,
    SchoolSummaryResponse,
    SchoolUpdateRequest,
    TokenResponse,
    UserResponse,
    UserSettingsResponse,
    UserSettingsUpdateRequest,
    UserStatusPatchRequest,
    VerifyResetOtpRequest,
    VerifyResetOtpResponse,
)
from app.modules.auth.public_ids import to_user_response, to_user_settings_response, heal_tutor_teaching_curriculum
from app.modules.auth.service import (
    create_school_admin,
    delete_school,
    issue_password_reset_otp,
    lookup_forgot_password_accounts,
    list_schools_with_stats,
    list_users,
    login,
    logout,
    refresh_access_token,
    reset_password_with_token,
    resolve_forgot_password_user,
    set_user_status,
    update_me_profile,
    update_school,
    update_user_settings,
    get_or_create_user_settings,
    reset_user_settings,
    verify_email_token,
    verify_password_reset_otp,
)
from app.services.mail import send_password_otp_email
from app.modules.users.models import User

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


class CreateSchoolAdminRequest(AdminCreateUserRequest):
    """Create a new school + first admin, or add another admin to ``school_id``."""

    school_id: int | None = None
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
    def _school_name_when_new(self) -> Self:
        if self.school_id is None:
            sn = self.school_name or ""
            if len(sn) < 2:
                raise ValueError("school_name is required (min 2 characters) when school_id is not provided.")
        return self



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


@router.post("/forgot-password/lookup", response_model=ForgotPasswordLookupResponse)
def forgot_password_lookup(
    payload: ForgotPasswordLookupRequest,
    db: Annotated[Session, Depends(get_db)],
):
    accounts = lookup_forgot_password_accounts(db, str(payload.email))
    db.commit()  # may create missing username settings rows
    return ForgotPasswordLookupResponse(
        accounts=[ForgotPasswordAccount(**a) for a in accounts],
    )


@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(payload: ForgotPasswordRequest, db: Annotated[Session, Depends(get_db)]):
    from app.modules.auth.exceptions import AuthException

    user = resolve_forgot_password_user(db, str(payload.email), payload.user_id)
    if not user:
        # no account leakage when lookup was skipped/stale
        return MessageResponse(message="If the account exists, a reset code has been sent.")

    otp = issue_password_reset_otp(db, user)
    ok = send_password_otp_email(
        to=user.email,
        full_name=user.full_name,
        otp=otp,
        expire_minutes=settings.password_otp_expire_minutes,
    )
    if not ok:
        db.rollback()
        raise AuthException(
            "We couldn't send the reset email. Please try again in a moment.",
            status.HTTP_502_BAD_GATEWAY,
        )
    db.commit()
    return MessageResponse(message="A reset code has been sent to your email.")


@router.post("/forgot-password/verify-otp", response_model=VerifyResetOtpResponse)
def forgot_password_verify_otp(
    payload: VerifyResetOtpRequest,
    db: Annotated[Session, Depends(get_db)],
):
    token = verify_password_reset_otp(db, str(payload.email), payload.user_id, payload.otp)
    db.commit()
    return VerifyResetOtpResponse(message="Code verified.", reset_token=token)


@router.post("/reset-password", response_model=MessageResponse)
def reset_password_route(payload: ResetPasswordRequest, db: Annotated[Session, Depends(get_db)]):
    reset_password_with_token(db, payload.reset_token, payload.new_password)
    db.commit()
    return MessageResponse(message="Password reset successful.")


@router.get("/verify-email", response_model=MessageResponse)
def verify_email(token: str, db: Annotated[Session, Depends(get_db)]):
    verify_email_token(db, token)
    db.commit()
    return MessageResponse(message="Email verified successfully.")


@router.get("/me", response_model=UserResponse)
def me(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    if heal_tutor_teaching_curriculum(db, current_user):
        db.commit()
        db.refresh(current_user)
    return to_user_response(db, current_user)


@router.patch("/me/profile", response_model=UserResponse)
def patch_me_profile(
    payload: MeProfileUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    user = update_me_profile(db, current_user, payload)
    db.commit()
    db.refresh(user)
    return to_user_response(db, user)


@router.get("/me/settings", response_model=UserSettingsResponse)
def get_me_settings(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    settings_row = get_or_create_user_settings(db, current_user)
    db.commit()
    db.refresh(settings_row)
    return to_user_settings_response(current_user, settings_row)


@router.patch("/me/settings", response_model=UserSettingsResponse)
def patch_me_settings(
    payload: UserSettingsUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    settings_row = update_user_settings(db, current_user, payload)
    db.commit()
    db.refresh(settings_row)
    return to_user_settings_response(current_user, settings_row)


@router.delete("/me/settings", response_model=UserSettingsResponse)
def delete_me_settings(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    settings_row = reset_user_settings(db, current_user)
    db.commit()
    db.refresh(settings_row)
    return to_user_settings_response(current_user, settings_row)


@router.post("/admin/create-school-admin", response_model=UserResponse, status_code=201)
def create_school_admin_route(
    payload: CreateSchoolAdminRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
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
    return to_user_response(db, user)


@router.get("/admin/schools", response_model=list[SchoolSummaryResponse])
def admin_schools(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
):
    schools = list_schools_with_stats(db, current_user)
    return schools


@router.patch("/admin/schools/{school_id}", response_model=SchoolSummaryResponse)
def patch_school(
    school_id: int,
    payload: SchoolUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
):
    summary = update_school(db, current_user, school_id, payload)
    db.commit()
    return summary


@router.delete("/admin/schools/{school_id}", response_model=MessageResponse)
def remove_school(
    school_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.MASTER_ADMIN))],
):
    delete_school(db, current_user, school_id)
    db.commit()
    return MessageResponse(message="School removed.")


@router.get("/admin/users", response_model=list[UserResponse])
def admin_users(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
):
    users = list_users(db, current_user)
    return [to_user_response(db, u) for u in users]


@router.patch("/admin/users/{user_id}/status", response_model=UserResponse)
def patch_user_status(
    user_id: int,
    payload: UserStatusPatchRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))],
):
    user = set_user_status(db, current_user, user_id, payload.is_active)
    db.commit()
    db.refresh(user)
    return to_user_response(db, user)
