import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import Request, status
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.student_messages import (
    ACCOUNT_INACTIVE,
    CURRENT_PASSWORD_REQUIRED,
    EMAIL_ALREADY_USED,
    INVALID_LOGIN,
    INVALID_RESET_TOKEN,
    INVALID_VERIFY_TOKEN,
    USERNAME_TAKEN,
    WRONG_CURRENT_PASSWORD,
)
from app.core.security import (
    create_access_token,
    create_email_token,
    decode_token,
    hash_password,
    hash_token,
    verify_password,
)
from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.auth.permissions import ADMIN_CREATE_MATRIX
from app.modules.auth.schemas import (
    AdminCreateUserRequest,
    LoginRequest,
    MeProfileUpdateRequest,
    OrganizationSignupRequest,
    OrganizationUpdateRequest,
    SchoolAdminBrief,
    SchoolSummaryResponse,
    SchoolUpdateRequest,
    StudentSignupRequest,
    TeachingClassAssignment,
    UpdateStudentRequest,
    UpdateTutorRequest,
    UserSettingsUpdateRequest,
)
from app.modules.auth.security import create_refresh_session
from app.modules.organizations.models import Organization
from app.modules.schools.models import School
from app.modules.sessions.models import SessionToken
from app.modules.users.models import User, UserSettings

settings = get_settings()


def _find_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.lower()))


def _find_user_by_login(db: Session, login: str) -> User | None:
    ident = (login or "").strip().lower()
    if not ident:
        return None

    # Standard email login path.
    user = _find_user_by_email(db, ident)
    if user:
        return user

    # Test/dev convenience: allow username-style login by local-part of email.
    if "@" not in ident:
        for candidate in db.scalars(select(User)):
            email = (candidate.email or "").lower()
            if email.split("@", 1)[0] == ident:
                return candidate
    return None


def _get_session_by_hash(db: Session, refresh_hash: str) -> SessionToken | None:
    return db.scalar(select(SessionToken).where(SessionToken.refresh_token_hash == refresh_hash))


def _validate_admin_create(actor: User, target_role: Role) -> None:
    allowed = ADMIN_CREATE_MATRIX.get(actor.role, set())
    if target_role not in allowed:
        raise AuthException("You do not have permission to create this role.", status.HTTP_403_FORBIDDEN)


def _ensure_active(user: User) -> None:
    if not user.is_active:
        raise AuthException(ACCOUNT_INACTIVE, status.HTTP_403_FORBIDDEN)


def signup_student(db: Session, payload: StudentSignupRequest) -> User:
    if _find_user_by_email(db, payload.email):
        raise AuthException(EMAIL_ALREADY_USED, status.HTTP_409_CONFLICT)

    user = User(
        full_name=payload.full_name,
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        role=Role.STUDENT,
        is_active=True,
        is_verified=False,
    )
    db.add(user)
    db.flush()
    return user


def signup_organization(db: Session, payload: OrganizationSignupRequest) -> User:
    if _find_user_by_email(db, payload.email):
        raise AuthException(EMAIL_ALREADY_USED, status.HTTP_409_CONFLICT)

    org = Organization(name=payload.organization_name, phone=payload.phone, address=payload.address, is_active=True)
    db.add(org)
    db.flush()

    user = User(
        full_name=payload.full_name,
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        role=Role.ORG_ADMIN,
        is_active=True,
        is_verified=False,
        organization_id=org.id,
    )
    db.add(user)
    db.flush()
    return user


def login(db: Session, payload: LoginRequest, request: Request) -> tuple[User, str, str]:
    user = _find_user_by_login(db, payload.email)
    if not user or not verify_password(payload.password, user.password_hash):
        raise AuthException(INVALID_LOGIN, status.HTTP_401_UNAUTHORIZED)
    _ensure_active(user)

    refresh_token, session = create_refresh_session(
        db,
        user_id=user.id,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )
    db.flush()

    access_token = create_access_token(subject=str(user.id), role=user.role.value)
    return user, access_token, refresh_token


def refresh_access_token(db: Session, refresh_token: str) -> tuple[User, str, str]:
    refresh_hash = hash_token(refresh_token)
    session = _get_session_by_hash(db, refresh_hash)
    if not session:
        raise AuthException("Invalid refresh token.", status.HTTP_401_UNAUTHORIZED)
    if session.revoked or session.expires_at <= datetime.now(UTC):
        raise AuthException("Refresh token expired or revoked.", status.HTTP_401_UNAUTHORIZED)

    user = db.get(User, session.user_id)
    if not user:
        raise AuthException("User not found.", status.HTTP_401_UNAUTHORIZED)
    _ensure_active(user)

    # rotate refresh token
    session.revoked = True
    new_refresh, new_session = create_refresh_session(
        db,
        user_id=user.id,
        user_agent=session.user_agent,
        ip_address=session.ip_address,
    )
    db.add(new_session)
    access_token = create_access_token(subject=str(user.id), role=user.role.value)
    return user, access_token, new_refresh


def logout(db: Session, user: User, refresh_token: str | None, all_devices: bool) -> None:
    if all_devices:
        db.execute(update(SessionToken).where(SessionToken.user_id == user.id).values(revoked=True))
        return
    if not refresh_token:
        raise AuthException("refresh_token is required when all_devices=false.")

    refresh_hash = hash_token(refresh_token)
    session = _get_session_by_hash(db, refresh_hash)
    if session and session.user_id == user.id:
        session.revoked = True


def admin_create_user(
    db: Session,
    actor: User,
    target_role: Role,
    payload: AdminCreateUserRequest,
    *,
    teaching_board: str | None = None,
    teaching_classes: list[TeachingClassAssignment] | None = None,
) -> User:
    _validate_admin_create(actor, target_role)
    if _find_user_by_email(db, payload.email):
        raise AuthException(EMAIL_ALREADY_USED, status.HTTP_409_CONFLICT)

    organization_id = payload.organization_id or actor.organization_id
    school_id = payload.school_id

    if actor.role == Role.ORG_ADMIN:
        organization_id = actor.organization_id
    if actor.role == Role.SCHOOL_ADMIN:
        organization_id = actor.organization_id
        school_id = actor.school_id
    if actor.role == Role.TUTOR:
        organization_id = actor.organization_id
        school_id = actor.school_id
        if target_role != Role.STUDENT:
            raise AuthException("Tutors can only onboard students.", status.HTTP_403_FORBIDDEN)

    if target_role == Role.TUTOR and actor.role == Role.ORG_ADMIN and school_id is None:
        raise AuthException("school_id is required when creating a tutor.", status.HTTP_400_BAD_REQUEST)
    if target_role == Role.TUTOR and actor.role == Role.MASTER_ADMIN and school_id is None:
        raise AuthException("school_id is required when creating a tutor.", status.HTTP_400_BAD_REQUEST)
    if target_role == Role.STUDENT and actor.role == Role.ORG_ADMIN and school_id is None:
        raise AuthException("school_id is required when creating a student.", status.HTTP_400_BAD_REQUEST)
    if target_role == Role.STUDENT and actor.role == Role.MASTER_ADMIN and school_id is None:
        raise AuthException("school_id is required when creating a student.", status.HTTP_400_BAD_REQUEST)
    if target_role == Role.STUDENT and actor.role == Role.TUTOR and school_id is None:
        raise AuthException("Tutor must be linked to a school before onboarding students.", status.HTTP_400_BAD_REQUEST)

    tb: str | None = None
    tc: list[dict] | None = None
    if target_role in (Role.TUTOR, Role.STUDENT):
        tb = teaching_board
        if not teaching_classes:
            raise AuthException(
                "At least one class (grade) with sections is required.",
                status.HTTP_400_BAD_REQUEST,
            )
        tc = [a.model_dump() for a in teaching_classes]
        if target_role == Role.STUDENT and actor.role == Role.TUTOR:
            tutor_rows = _normalized_class_rows(actor.teaching_classes)
            student_rows = _normalized_class_rows(tc)
            is_allowed = any(
                (t_grade == s_grade) and bool(t_sections.intersection(s_sections))
                for (t_grade, t_sections) in tutor_rows
                for (s_grade, s_sections) in student_rows
            )
            if not is_allowed:
                raise AuthException(
                    "Student class/section must match tutor class/section tags.",
                    status.HTTP_400_BAD_REQUEST,
                )

    user = User(
        full_name=payload.full_name,
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        role=target_role,
        is_active=True,
        is_verified=False,
        organization_id=organization_id,
        school_id=school_id,
        teaching_board=tb,
        teaching_classes=tc,
        created_by=actor.id,
    )
    db.add(user)
    db.flush()
    return user


def create_school_admin(
    db: Session,
    actor: User,
    payload: AdminCreateUserRequest,
    school_name: str | None,
    board: str | None,
    *,
    branch: str | None = None,
    school_id: uuid.UUID | None = None,
) -> User:
    _validate_admin_create(actor, Role.SCHOOL_ADMIN)
    if _find_user_by_email(db, payload.email):
        raise AuthException(EMAIL_ALREADY_USED, status.HTTP_409_CONFLICT)

    if school_id is not None:
        school = db.get(School, school_id)
        if not school:
            raise AuthException("School not found.", status.HTTP_404_NOT_FOUND)
        if actor.role == Role.ORG_ADMIN:
            if not actor.organization_id or school.organization_id != actor.organization_id:
                raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
        elif actor.role == Role.MASTER_ADMIN:
            if payload.organization_id is not None and school.organization_id != payload.organization_id:
                raise AuthException(
                    "School does not belong to the specified organization.",
                    status.HTTP_400_BAD_REQUEST,
                )
        org_id = school.organization_id
        target_school_id = school.id
    else:
        org_id = actor.organization_id if actor.role != Role.MASTER_ADMIN else payload.organization_id
        if not org_id:
            raise AuthException("organization_id is required for school admin creation.")
        if not school_name:
            raise AuthException(
                "school_name is required when school_id is not provided.",
                status.HTTP_400_BAD_REQUEST,
            )
        school = School(organization_id=org_id, name=school_name, branch=branch, board=board)
        db.add(school)
        db.flush()
        target_school_id = school.id

    user = User(
        full_name=payload.full_name,
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        role=Role.SCHOOL_ADMIN,
        is_active=True,
        is_verified=False,
        organization_id=org_id,
        school_id=target_school_id,
        created_by=actor.id,
    )
    db.add(user)
    db.flush()
    return user


def _assert_actor_manages_user(actor: User, target: User) -> None:
    if actor.role == Role.MASTER_ADMIN:
        return
    if actor.role == Role.ORG_ADMIN:
        if target.organization_id != actor.organization_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
        return
    if actor.role == Role.SCHOOL_ADMIN:
        if target.school_id != actor.school_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
        return
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


def list_users(db: Session, actor: User) -> list[User]:
    stmt = select(User).order_by(User.created_at.desc())
    if actor.role == Role.ORG_ADMIN:
        stmt = stmt.where(User.organization_id == actor.organization_id)
    elif actor.role == Role.SCHOOL_ADMIN:
        stmt = stmt.where(User.school_id == actor.school_id)
    return list(db.scalars(stmt))


def _normalized_class_rows(raw: Any) -> list[tuple[str, set[str]]]:
    rows: list[tuple[str, set[str]]] = []
    if not isinstance(raw, list):
        return rows
    if raw and isinstance(raw[0], str):
        return [(str(g).strip(), {"A"}) for g in raw if str(g).strip()]
    for item in raw:
        if not isinstance(item, dict):
            continue
        grade = str(item.get("grade", "")).strip()
        if not grade:
            continue
        sections = {
            str(s).strip().upper()
            for s in (item.get("sections") or [])
            if str(s).strip()
        }
        if sections:
            rows.append((grade, sections))
    return rows


def list_tutor_assigned_students(db: Session, tutor: User) -> list[User]:
    if tutor.role != Role.TUTOR:
        raise AuthException("Only tutors can access assigned students.", status.HTTP_403_FORBIDDEN)
    if not tutor.school_id:
        return []

    tutor_rows = _normalized_class_rows(tutor.teaching_classes)
    if not tutor_rows:
        return []

    students = list(
        db.scalars(
            select(User)
            .where(User.role == Role.STUDENT, User.school_id == tutor.school_id)
            .order_by(User.created_at.desc())
        )
    )

    out: list[User] = []
    for student in students:
        student_rows = _normalized_class_rows(student.teaching_classes)
        is_match = any(
            (t_grade == s_grade) and bool(t_sections.intersection(s_sections))
            for (t_grade, t_sections) in tutor_rows
            for (s_grade, s_sections) in student_rows
        )
        if is_match:
            out.append(student)
    return out


def _get_school_for_admin(db: Session, actor: User, school_id: uuid.UUID) -> School:
    school = db.get(School, school_id)
    if not school:
        raise AuthException("School not found.", status.HTTP_404_NOT_FOUND)
    if actor.role == Role.ORG_ADMIN:
        if not actor.organization_id or school.organization_id != actor.organization_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    elif actor.role == Role.MASTER_ADMIN:
        pass
    else:
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    return school


def _schools_to_summaries(db: Session, schools: list[School]) -> list[SchoolSummaryResponse]:
    if not schools:
        return []

    school_ids = [s.id for s in schools]
    org_ids = list({s.organization_id for s in schools})
    org_names = {
        row[0]: row[1]
        for row in db.execute(select(Organization.id, Organization.name).where(Organization.id.in_(org_ids))).all()
    }

    count_rows = db.execute(
        select(User.school_id, User.role, func.count(User.id))
        .where(User.school_id.in_(school_ids))
        .group_by(User.school_id, User.role)
    ).all()
    counts: dict[uuid.UUID, dict[str, int]] = {}
    for sid, role, cnt in count_rows:
        if sid is None:
            continue
        role_key = role.value if isinstance(role, Role) else str(role)
        counts.setdefault(sid, {})[role_key] = int(cnt)

    admins = list(
        db.scalars(
            select(User).where(User.role == Role.SCHOOL_ADMIN, User.school_id.in_(school_ids)).order_by(User.created_at)
        )
    )
    admins_by_school: dict[uuid.UUID, list[User]] = {}
    for u in admins:
        if u.school_id:
            admins_by_school.setdefault(u.school_id, []).append(u)

    out: list[SchoolSummaryResponse] = []
    for s in schools:
        c = counts.get(s.id, {})
        out.append(
            SchoolSummaryResponse(
                id=s.id,
                organization_id=s.organization_id,
                organization_name=org_names.get(s.organization_id),
                name=s.name,
                branch=s.branch,
                board=s.board,
                created_at=s.created_at,
                school_admins=[SchoolAdminBrief.model_validate(a) for a in admins_by_school.get(s.id, [])],
                tutor_count=c.get(Role.TUTOR.value, 0),
                student_count=c.get(Role.STUDENT.value, 0),
            )
        )
    return out


def list_schools_with_stats(
    db: Session,
    actor: User,
    organization_id: uuid.UUID | None = None,
) -> list[SchoolSummaryResponse]:
    """
    Org admins see schools in their organization. School admins only see their
    linked school. Master admins may filter by ``organization_id`` or list all
    schools.
    """
    stmt = select(School)
    if actor.role == Role.ORG_ADMIN:
        if not actor.organization_id:
            raise AuthException("Organization context missing.", status.HTTP_400_BAD_REQUEST)
        stmt = stmt.where(School.organization_id == actor.organization_id)
    elif actor.role == Role.SCHOOL_ADMIN:
        if not actor.school_id:
            raise AuthException("School context missing.", status.HTTP_400_BAD_REQUEST)
        stmt = stmt.where(School.id == actor.school_id)
    elif actor.role == Role.MASTER_ADMIN:
        if organization_id is not None:
            stmt = stmt.where(School.organization_id == organization_id)
    else:
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)

    schools = list(db.scalars(stmt.order_by(School.created_at.desc())))
    return _schools_to_summaries(db, schools)


def update_school(db: Session, actor: User, school_id: uuid.UUID, payload: SchoolUpdateRequest) -> SchoolSummaryResponse:
    school = _get_school_for_admin(db, actor, school_id)
    data = payload.model_dump(exclude_unset=True)
    if not data:
        raise AuthException("No fields to update.", status.HTTP_400_BAD_REQUEST)
    if "name" in data and data["name"] is not None:
        school.name = data["name"].strip()
    if "branch" in data:
        school.branch = data["branch"]
    if "board" in data:
        school.board = data["board"]
    db.flush()
    return _schools_to_summaries(db, [school])[0]


def delete_school(db: Session, actor: User, school_id: uuid.UUID) -> None:
    school = _get_school_for_admin(db, actor, school_id)
    db.delete(school)
    db.flush()


def set_user_status(db: Session, actor: User, user_id: uuid.UUID, is_active: bool) -> User:
    user = db.get(User, user_id)
    if not user:
        raise AuthException("User not found.", status.HTTP_404_NOT_FOUND)

    _assert_actor_manages_user(actor, user)

    user.is_active = is_active
    user.updated_at = datetime.now(UTC)
    return user


def update_tutor(db: Session, actor: User, user_id: uuid.UUID, payload: UpdateTutorRequest) -> User:
    user = db.get(User, user_id)
    if not user:
        raise AuthException("User not found.", status.HTTP_404_NOT_FOUND)
    if user.role != Role.TUTOR:
        raise AuthException("Not a tutor account.", status.HTTP_400_BAD_REQUEST)
    _assert_actor_manages_user(actor, user)

    school = db.get(School, payload.school_id)
    if not school:
        raise AuthException("School not found.", status.HTTP_404_NOT_FOUND)
    if actor.role == Role.ORG_ADMIN:
        if not actor.organization_id or school.organization_id != actor.organization_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    elif actor.role == Role.SCHOOL_ADMIN:
        if payload.school_id != actor.school_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    elif actor.role != Role.MASTER_ADMIN:
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)

    new_email = str(payload.email).lower()
    if new_email != user.email and _find_user_by_email(db, new_email):
        raise AuthException(EMAIL_ALREADY_USED, status.HTTP_409_CONFLICT)

    user.full_name = payload.full_name.strip()
    user.email = new_email
    user.school_id = payload.school_id
    user.organization_id = school.organization_id
    user.teaching_board = payload.teaching_board
    user.teaching_classes = [a.model_dump() for a in payload.teaching_classes]

    if payload.new_password:
        user.password_hash = hash_password(payload.new_password)
        db.execute(update(SessionToken).where(SessionToken.user_id == user.id).values(revoked=True))

    user.updated_at = datetime.now(UTC)
    db.flush()
    return user


def delete_tutor_user(db: Session, actor: User, user_id: uuid.UUID) -> None:
    user = db.get(User, user_id)
    if not user:
        raise AuthException("User not found.", status.HTTP_404_NOT_FOUND)
    if user.role != Role.TUTOR:
        raise AuthException("Only tutor accounts can be removed with this action.", status.HTTP_400_BAD_REQUEST)
    _assert_actor_manages_user(actor, user)
    db.delete(user)
    db.flush()


def update_student(db: Session, actor: User, user_id: uuid.UUID, payload: UpdateStudentRequest) -> User:
    user = db.get(User, user_id)
    if not user:
        raise AuthException("User not found.", status.HTTP_404_NOT_FOUND)
    if user.role != Role.STUDENT:
        raise AuthException("Not a student account.", status.HTTP_400_BAD_REQUEST)
    _assert_actor_manages_user(actor, user)

    school = db.get(School, payload.school_id)
    if not school:
        raise AuthException("School not found.", status.HTTP_404_NOT_FOUND)
    if actor.role == Role.ORG_ADMIN:
        if not actor.organization_id or school.organization_id != actor.organization_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    elif actor.role == Role.SCHOOL_ADMIN:
        if payload.school_id != actor.school_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    elif actor.role != Role.MASTER_ADMIN:
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)

    new_email = str(payload.email).lower()
    if new_email != user.email and _find_user_by_email(db, new_email):
        raise AuthException(EMAIL_ALREADY_USED, status.HTTP_409_CONFLICT)

    user.full_name = payload.full_name.strip()
    user.email = new_email
    user.school_id = payload.school_id
    user.organization_id = school.organization_id
    user.teaching_board = payload.teaching_board
    user.teaching_classes = [a.model_dump() for a in payload.teaching_classes]

    if payload.new_password:
        user.password_hash = hash_password(payload.new_password)
        db.execute(update(SessionToken).where(SessionToken.user_id == user.id).values(revoked=True))

    user.updated_at = datetime.now(UTC)
    db.flush()
    return user


def delete_student_user(db: Session, actor: User, user_id: uuid.UUID) -> None:
    user = db.get(User, user_id)
    if not user:
        raise AuthException("User not found.", status.HTTP_404_NOT_FOUND)
    if user.role != Role.STUDENT:
        raise AuthException("Only student accounts can be removed with this action.", status.HTTP_400_BAD_REQUEST)
    _assert_actor_manages_user(actor, user)
    db.delete(user)
    db.flush()


def get_organization_for_org_admin(db: Session, actor: User) -> Organization:
    if actor.role != Role.ORG_ADMIN:
        raise AuthException(
            "Only organization administrators can access this resource.",
            status.HTTP_403_FORBIDDEN,
        )
    if not actor.organization_id:
        raise AuthException("No organization is linked to this account.", status.HTTP_400_BAD_REQUEST)
    org = db.get(Organization, actor.organization_id)
    if not org:
        raise AuthException("Organization not found.", status.HTTP_404_NOT_FOUND)
    return org


def update_organization_for_org_admin(db: Session, actor: User, payload: OrganizationUpdateRequest) -> Organization:
    org = get_organization_for_org_admin(db, actor)
    org.name = payload.name
    org.phone = payload.phone
    org.address = payload.address
    db.flush()
    return org


def update_me_profile(db: Session, user: User, payload: MeProfileUpdateRequest) -> User:
    new_email = str(payload.email).lower().strip()
    if new_email != user.email:
        if _find_user_by_email(db, new_email):
            raise AuthException(EMAIL_ALREADY_USED, status.HTTP_409_CONFLICT)
        user.email = new_email

    user.full_name = payload.full_name

    if payload.new_password:
        if not payload.current_password:
            raise AuthException(CURRENT_PASSWORD_REQUIRED, status.HTTP_400_BAD_REQUEST)
        if not verify_password(payload.current_password, user.password_hash):
            raise AuthException(WRONG_CURRENT_PASSWORD, status.HTTP_403_FORBIDDEN)
        user.password_hash = hash_password(payload.new_password)
        db.execute(update(SessionToken).where(SessionToken.user_id == user.id).values(revoked=True))

    user.updated_at = datetime.now(UTC)
    db.flush()
    return user


def build_auth_tokens(user: User, access_token: str, refresh_token: str) -> dict:
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_exp_minutes * 60,
        "user": user,
    }


def issue_verify_email_token(user: User) -> str:
    return create_email_token(subject=str(user.id), purpose="verify_email", expires_minutes=60 * 24)


def issue_reset_password_token(user: User) -> str:
    return create_email_token(subject=str(user.id), purpose="reset_password", expires_minutes=30)


def verify_email_token(db: Session, token: str) -> None:
    payload = decode_token(token)
    if payload.get("type") != "verify_email":
        raise AuthException(INVALID_VERIFY_TOKEN, status.HTTP_400_BAD_REQUEST)
    user = db.get(User, payload.get("sub"))
    if not user:
        raise AuthException("User not found.", status.HTTP_404_NOT_FOUND)
    user.is_verified = True


def reset_password(db: Session, token: str, new_password: str) -> None:
    payload = decode_token(token)
    if payload.get("type") != "reset_password":
        raise AuthException(INVALID_RESET_TOKEN, status.HTTP_400_BAD_REQUEST)
    user = db.get(User, payload.get("sub"))
    if not user:
        raise AuthException("User not found.", status.HTTP_404_NOT_FOUND)
    user.password_hash = hash_password(new_password)
    user.updated_at = datetime.now(UTC)
    db.execute(update(SessionToken).where(SessionToken.user_id == user.id).values(revoked=True))


def _default_username(user: User) -> str:
    local = (user.email or "").split("@", 1)[0].strip().lower()
    return local or "user"


def _find_settings_by_username(db: Session, username: str, exclude_user_id: uuid.UUID | None = None) -> UserSettings | None:
    ident = (username or "").strip().lower()
    if not ident:
        return None
    stmt = select(UserSettings).where(func.lower(UserSettings.username) == ident)
    if exclude_user_id:
        stmt = stmt.where(UserSettings.user_id != exclude_user_id)
    return db.scalar(stmt)


def get_or_create_user_settings(db: Session, user: User) -> UserSettings:
    row = db.get(UserSettings, user.id)
    if row:
        return row
    base_username = _default_username(user)
    username = base_username
    suffix = 1
    while _find_settings_by_username(db, username):
        username = f"{base_username}{suffix}"
        suffix += 1
    row = UserSettings(
        user_id=user.id,
        username=username,
        language="en",
        theme="light",
    )
    db.add(row)
    db.flush()
    return row


def update_user_settings(db: Session, user: User, payload: UserSettingsUpdateRequest) -> UserSettings:
    row = get_or_create_user_settings(db, user)
    data = payload.model_dump(exclude_unset=True)

    if "username" in data:
        new_username = data["username"]
        if new_username:
            existing = _find_settings_by_username(db, new_username, exclude_user_id=user.id)
            if existing:
                raise AuthException(USERNAME_TAKEN, status.HTTP_409_CONFLICT)
            row.username = new_username.lower()
        else:
            row.username = None

    for field in ("language", "theme", "notify_email", "notify_push", "notify_assignments", "notify_sessions", "notify_messages"):
        if field in data and data[field] is not None:
            setattr(row, field, data[field])

    row.updated_at = datetime.now(UTC)
    db.flush()
    return row


def reset_user_settings(db: Session, user: User) -> UserSettings:
    existing = db.get(UserSettings, user.id)
    if existing:
        db.delete(existing)
        db.flush()
    return get_or_create_user_settings(db, user)
