from datetime import UTC, datetime, timedelta
from typing import Any
import secrets

from fastapi import Request, status
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.student_messages import (
    ACCOUNT_INACTIVE,
    CURRENT_PASSWORD_REQUIRED,
    INVALID_LOGIN,
    INVALID_RESET_OTP,
    INVALID_RESET_TOKEN,
    INVALID_VERIFY_TOKEN,
    LOGIN_USE_USERNAME,
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
from app.modules.auth.public_ids import (
    require_school_by_seq,
    require_user_by_account_id,
    resolve_school_uuid,
    to_school_admin_brief,
)
from app.modules.auth.schemas import (
    AdminCreateUserRequest,
    LoginRequest,
    MeProfileUpdateRequest,
    SchoolProfileUpdateRequest,
    SchoolSummaryResponse,
    SchoolUpdateRequest,
    StudentSignupRequest,
    TeachingClassAssignment,
    UpdateStudentRequest,
    UpdateTutorRequest,
    UserSettingsUpdateRequest,
)
from app.modules.auth.security import create_refresh_session
from app.modules.schools.models import School
from app.modules.sessions.models import SessionToken
from app.modules.users.models import User, UserSettings

settings = get_settings()


def _find_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.lower()))


def _users_by_email(db: Session, email: str) -> list[User]:
    return list(db.scalars(select(User).where(User.email == email.lower()).order_by(User.id)).all())


def _find_user_by_login(db: Session, login: str) -> User | None:
    from app.modules.auth.signup.validators import (
        find_user_by_username,
        normalize_phone,
    )

    ident = (login or "").strip()
    if not ident:
        return None

    # Username / user-id first (unique)
    username_user = find_user_by_username(db, ident)
    if username_user:
        return username_user

    # Email: only if exactly one account
    if "@" in ident:
        matches = _users_by_email(db, ident.lower())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AuthException(LOGIN_USE_USERNAME, status.HTTP_401_UNAUTHORIZED)
        return None

    # Phone: only if exactly one account
    phone = normalize_phone(ident)
    if phone:
        phone_matches = list(db.scalars(select(User).where(User.phone == phone).order_by(User.id)).all())
        if len(phone_matches) == 1:
            return phone_matches[0]
        if len(phone_matches) > 1:
            raise AuthException(LOGIN_USE_USERNAME, status.HTTP_401_UNAUTHORIZED)

    # Legacy: local-part of email (test/dev) — only if unique
    ident_lower = ident.lower()
    local_matches = [
        candidate
        for candidate in db.scalars(select(User)).all()
        if (candidate.email or "").lower().split("@", 1)[0] == ident_lower
    ]
    if len(local_matches) == 1:
        return local_matches[0]
    if len(local_matches) > 1:
        raise AuthException(LOGIN_USE_USERNAME, status.HTTP_401_UNAUTHORIZED)
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

    school_id = resolve_school_uuid(db, payload.school_id)

    if actor.role == Role.SCHOOL_ADMIN:
        school_id = actor.school_id
    if actor.role == Role.TUTOR:
        school_id = actor.school_id
        if target_role != Role.STUDENT:
            raise AuthException("Tutors can only onboard students.", status.HTTP_403_FORBIDDEN)

    if target_role == Role.TUTOR and actor.role == Role.MASTER_ADMIN and school_id is None:
        raise AuthException("school_id is required when creating a tutor.", status.HTTP_400_BAD_REQUEST)
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
    school_id: int | None = None,
) -> User:
    _validate_admin_create(actor, Role.SCHOOL_ADMIN)

    if school_id is not None:
        school = require_school_by_seq(db, school_id)
        if actor.role == Role.SCHOOL_ADMIN and actor.school_id != school.id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
        target_school_id = school.id
    else:
        if actor.role != Role.MASTER_ADMIN:
            raise AuthException("Only master admins can create a new school.", status.HTTP_403_FORBIDDEN)
        if not school_name:
            raise AuthException(
                "school_name is required when school_id is not provided.",
                status.HTTP_400_BAD_REQUEST,
            )
        school = School(name=school_name, branch=branch, board=board, is_active=True)
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
        school_id=target_school_id,
        created_by=actor.id,
    )
    db.add(user)
    db.flush()
    return user


def _assert_actor_manages_user(actor: User, target: User) -> None:
    if actor.role == Role.MASTER_ADMIN:
        return
    if actor.role == Role.SCHOOL_ADMIN:
        if target.school_id != actor.school_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
        return
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


def list_users(db: Session, actor: User) -> list[User]:
    stmt = select(User).order_by(User.created_at.desc())
    if actor.role == Role.SCHOOL_ADMIN:
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


def _teaching_class_ids(raw: Any) -> set[int]:
    ids: set[int] = set()
    if not isinstance(raw, list):
        return ids
    for item in raw:
        if not isinstance(item, dict):
            continue
        sc_id = item.get("school_class_id")
        if sc_id is not None and str(sc_id).strip().lstrip("-").isdigit():
            ids.add(int(sc_id))
    return ids


def list_tutor_assigned_students(db: Session, tutor: User) -> list[User]:
    if tutor.role != Role.TUTOR:
        raise AuthException("Only tutors can access assigned students.", status.HTTP_403_FORBIDDEN)
    if not tutor.school_id:
        # Individual tutor: ownership by created_by (class optional).
        return list(
            db.scalars(
                select(User)
                .where(
                    User.role == Role.STUDENT,
                    User.created_by == tutor.id,
                    User.school_id.is_(None),
                    User.is_active.is_(True),
                )
                .order_by(User.created_at.desc())
            )
        )

    tutor_rows = _normalized_class_rows(tutor.teaching_classes)
    tutor_class_ids = _teaching_class_ids(tutor.teaching_classes)

    students = list(
        db.scalars(
            select(User)
            .where(
                User.role == Role.STUDENT,
                User.school_id == tutor.school_id,
                User.is_active.is_(True),
            )
            .order_by(User.created_at.desc())
        )
    )

    out: list[User] = []
    excluded = {
        int(x)
        for x in (tutor.excluded_student_ids or [])
        if str(x).strip().lstrip("-").isdigit()
    }
    for student in students:
        if student.id in excluded:
            continue
        # Tutor-created students are auto-assigned to that tutor (even with no class tags yet).
        if student.created_by == tutor.id:
            out.append(student)
            continue
        if not tutor_rows and not tutor_class_ids:
            continue
        student_ids = _teaching_class_ids(student.teaching_classes)
        if tutor_class_ids and student_ids and tutor_class_ids.intersection(student_ids):
            out.append(student)
            continue
        student_rows = _normalized_class_rows(student.teaching_classes)
        is_match = any(
            (t_grade == s_grade) and bool(t_sections.intersection(s_sections))
            for (t_grade, t_sections) in tutor_rows
            for (s_grade, s_sections) in student_rows
        )
        if is_match:
            out.append(student)
    return out


def _get_school_for_admin(db: Session, actor: User, school_seq_id: int) -> School:
    school = require_school_by_seq(db, school_seq_id)
    if actor.role == Role.SCHOOL_ADMIN:
        if not actor.school_id or school.id != actor.school_id:
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

    count_rows = db.execute(
        select(User.school_id, User.role, func.count(User.id))
        .where(User.school_id.in_(school_ids))
        .group_by(User.school_id, User.role)
    ).all()
    counts: dict[int, dict[str, int]] = {}
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
    admins_by_school: dict[int, list[User]] = {}
    for u in admins:
        if u.school_id:
            admins_by_school.setdefault(u.school_id, []).append(u)

    out: list[SchoolSummaryResponse] = []
    for s in schools:
        c = counts.get(s.id, {})
        out.append(
            SchoolSummaryResponse(
                id=s.id,
                name=s.name,
                branch=s.branch,
                board=s.board,
                created_at=s.created_at,
                school_admins=[to_school_admin_brief(a) for a in admins_by_school.get(s.id, [])],
                tutor_count=c.get(Role.TUTOR.value, 0),
                student_count=c.get(Role.STUDENT.value, 0),
            )
        )
    return out


def list_schools_with_stats(
    db: Session,
    actor: User,
) -> list[SchoolSummaryResponse]:
    """School admins see their school; master admins see all schools."""
    stmt = select(School)
    if actor.role == Role.SCHOOL_ADMIN:
        if not actor.school_id:
            raise AuthException("School context missing.", status.HTTP_400_BAD_REQUEST)
        stmt = stmt.where(School.id == actor.school_id)
    elif actor.role == Role.MASTER_ADMIN:
        pass
    else:
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)

    schools = list(db.scalars(stmt.order_by(School.created_at.desc())))
    return _schools_to_summaries(db, schools)


def update_school(db: Session, actor: User, school_id: int, payload: SchoolUpdateRequest) -> SchoolSummaryResponse:
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
    if "email" in data:
        school.email = data["email"]
    if "phone" in data:
        school.phone = data["phone"]
    if "website" in data:
        school.website = data["website"]
    if "address" in data:
        school.address = data["address"]
    db.flush()
    return _schools_to_summaries(db, [school])[0]


def delete_school(db: Session, actor: User, school_id: int) -> None:
    school = _get_school_for_admin(db, actor, school_id)
    db.delete(school)
    db.flush()


def set_user_status(db: Session, actor: User, user_id: int, is_active: bool) -> User:
    user = require_user_by_account_id(db, user_id)

    _assert_actor_manages_user(actor, user)

    user.is_active = is_active
    user.updated_at = datetime.now(UTC)
    return user


def update_tutor(db: Session, actor: User, user_id: int, payload: UpdateTutorRequest) -> User:
    user = require_user_by_account_id(db, user_id)
    if user.role != Role.TUTOR:
        raise AuthException("Not a tutor account.", status.HTTP_400_BAD_REQUEST)
    _assert_actor_manages_user(actor, user)

    school = require_school_by_seq(db, payload.school_id)
    if actor.role == Role.SCHOOL_ADMIN:
        if school.id != actor.school_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    elif actor.role != Role.MASTER_ADMIN:
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)

    new_email = str(payload.email).lower()

    user.full_name = payload.full_name.strip()
    user.email = new_email
    user.school_id = school.id
    user.teaching_board = payload.teaching_board
    user.teaching_classes = [a.model_dump() for a in payload.teaching_classes]

    if payload.new_password:
        user.password_hash = hash_password(payload.new_password)
        db.execute(update(SessionToken).where(SessionToken.user_id == user.id).values(revoked=True))

    user.updated_at = datetime.now(UTC)
    db.flush()
    return user


def delete_tutor_user(db: Session, actor: User, user_id: int) -> None:
    user = require_user_by_account_id(db, user_id)
    if user.role != Role.TUTOR:
        raise AuthException("Only tutor accounts can be removed with this action.", status.HTTP_400_BAD_REQUEST)
    _assert_actor_manages_user(actor, user)
    db.delete(user)
    db.flush()


def update_student(db: Session, actor: User, user_id: int, payload: UpdateStudentRequest) -> User:
    user = require_user_by_account_id(db, user_id)
    if user.role != Role.STUDENT:
        raise AuthException("Not a student account.", status.HTTP_400_BAD_REQUEST)
    _assert_actor_manages_user(actor, user)

    school = require_school_by_seq(db, payload.school_id)
    if actor.role == Role.SCHOOL_ADMIN:
        if school.id != actor.school_id:
            raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    elif actor.role != Role.MASTER_ADMIN:
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)

    new_email = str(payload.email).lower()

    user.full_name = payload.full_name.strip()
    user.email = new_email
    user.school_id = school.id
    user.teaching_board = payload.teaching_board
    user.teaching_classes = [a.model_dump() for a in payload.teaching_classes]

    if payload.new_password:
        user.password_hash = hash_password(payload.new_password)
        db.execute(update(SessionToken).where(SessionToken.user_id == user.id).values(revoked=True))

    user.updated_at = datetime.now(UTC)
    db.flush()
    return user


def delete_student_user(db: Session, actor: User, user_id: int) -> None:
    user = require_user_by_account_id(db, user_id)
    if user.role != Role.STUDENT:
        raise AuthException("Only student accounts can be removed with this action.", status.HTTP_400_BAD_REQUEST)
    _assert_actor_manages_user(actor, user)
    db.delete(user)
    db.flush()


def get_school_for_school_admin(db: Session, actor: User) -> School:
    if actor.role != Role.SCHOOL_ADMIN:
        raise AuthException(
            "Only school administrators can access this resource.",
            status.HTTP_403_FORBIDDEN,
        )
    if not actor.school_id:
        raise AuthException("No school is linked to this account.", status.HTTP_400_BAD_REQUEST)
    school = db.get(School, actor.school_id)
    if not school:
        raise AuthException("School not found.", status.HTTP_404_NOT_FOUND)
    return school


def update_school_profile_for_admin(
    db: Session, actor: User, payload: SchoolProfileUpdateRequest
) -> School:
    from app.modules.auth.signup.enums import (
        CurriculumEnum,
        SchoolGradeRangeEnum,
        StudentStrengthEnum,
        parse_enum,
    )

    school = get_school_for_school_admin(db, actor)
    data = payload.model_dump(exclude_unset=True)

    school.name = data["name"] if "name" in data else school.name
    if "branch" in data:
        school.branch = data["branch"]
    if "board" in data:
        school.board = data["board"]
    if "email" in data:
        school.email = data["email"]
    if "phone" in data:
        school.phone = data["phone"]
    if "address" in data:
        school.address = data["address"]
    if "website" in data:
        school.website = data["website"]
    if "grades_offered" in data:
        school.grades_offered = (
            parse_enum(SchoolGradeRangeEnum, data["grades_offered"], "grades_offered")
            if data["grades_offered"]
            else None
        )
    if "student_strength" in data:
        school.student_strength = (
            parse_enum(StudentStrengthEnum, data["student_strength"], "student_strength")
            if data["student_strength"]
            else None
        )
    if "curricula" in data and data["curricula"] is not None:
        school.curricula = [
            parse_enum(CurriculumEnum, c, "curriculum") for c in data["curricula"] if str(c).strip()
        ] or None
        if school.curricula:
            school.board = school.curricula[0].value
    db.flush()
    return school


def update_me_profile(db: Session, user: User, payload: MeProfileUpdateRequest) -> User:
    from app.modules.auth.constants import Role
    from app.modules.auth.signup.models import UserSignupProfile
    from app.modules.auth.signup.validators import (
        parse_curricula,
        parse_learning_goals,
        parse_learning_method,
        parse_student_grade,
        parse_student_subjects,
    )

    new_email = str(payload.email).lower().strip()
    if new_email != user.email:
        user.email = new_email

    user.full_name = payload.full_name

    data = payload.model_dump(exclude_unset=True)
    if "phone" in data:
        user.phone = data["phone"]

    profile = db.get(UserSignupProfile, user.id)

    if "designation" in data:
        if profile is None:
            profile = UserSignupProfile(user_id=user.id)
            db.add(profile)
        profile.designation = data["designation"]

    if user.role == Role.STUDENT:
        # ponytail: Tagged students (school/tutor scope) do not own signup personal
        # details — grade/curricula/subjects come from tagging. Only individuals update them.
        is_tagged_student = user.school_id is not None or user.created_by is not None
        if not is_tagged_student:
            if profile is None:
                profile = UserSignupProfile(user_id=user.id)
                db.add(profile)

            if "grade" in data and data["grade"]:
                grade = parse_student_grade(data["grade"])
                profile.student_grade = grade
                user.teaching_classes = [{"grade": grade.value, "sections": ["A"]}]
            if "curricula" in data and data["curricula"]:
                curricula = parse_curricula(data["curricula"])
                profile.curricula = curricula
                if curricula:
                    user.teaching_board = curricula[0].value
            if "parent_email" in data:
                profile.parent_email = (
                    str(data["parent_email"]).lower().strip() if data["parent_email"] else None
                )
            if "favorite_subjects" in data:
                subjects = parse_student_subjects(data["favorite_subjects"] or [])
                profile.favorite_subjects = subjects or None
            if "learning_goals" in data:
                goals = parse_learning_goals(data["learning_goals"] or [])
                profile.learning_goals = goals or None
            if "preferred_learning_method" in data:
                profile.preferred_learning_method = parse_learning_method(
                    data["preferred_learning_method"]
                )

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


def lookup_forgot_password_accounts(db: Session, email: str) -> list[dict[str, Any]]:
    """Accounts tied to this email (name + username) for the forgot-password picker."""
    ident = email.lower().strip()
    users = list(db.scalars(select(User).where(User.email == ident).order_by(User.id)).all())
    accounts: list[dict[str, Any]] = []
    for user in users:
        settings_row = get_or_create_user_settings(db, user)
        accounts.append(
            {
                "id": user.id,
                "full_name": user.full_name,
                "username": settings_row.username or _default_username(user),
            }
        )
    return accounts


def resolve_forgot_password_user(db: Session, email: str, user_id: int) -> User | None:
    user = db.get(User, user_id)
    if not user or user.email.lower() != email.lower().strip():
        return None
    return user


def issue_password_reset_otp(db: Session, user: User) -> str:
    """Create a hashed OTP row and return the plaintext code (for email only)."""
    from app.modules.auth.otp_models import PasswordResetOtp

    settings = get_settings()
    length = max(4, min(12, settings.password_otp_length))
    otp = "".join(secrets.choice("0123456789") for _ in range(length))
    now = datetime.now(UTC)
    # Invalidate prior unused codes for this user
    db.execute(
        update(PasswordResetOtp)
        .where(
            PasswordResetOtp.user_id == user.id,
            PasswordResetOtp.used.is_(False),
        )
        .values(used=True)
    )
    row = PasswordResetOtp(
        user_id=user.id,
        code_hash=hash_password(otp),
        expires_at=now + timedelta(minutes=settings.password_otp_expire_minutes),
        attempts=0,
        used=False,
        created_at=now,
    )
    db.add(row)
    db.flush()
    return otp


def verify_email_token(db: Session, token: str) -> None:
    payload = decode_token(token)
    if payload.get("type") != "verify_email":
        raise AuthException(INVALID_VERIFY_TOKEN, status.HTTP_400_BAD_REQUEST)
    try:
        user = db.get(User, int(payload.get("sub")))
    except (TypeError, ValueError) as exc:
        raise AuthException(INVALID_VERIFY_TOKEN, status.HTTP_400_BAD_REQUEST) from exc
    if not user:
        raise AuthException("User not found.", status.HTTP_404_NOT_FOUND)
    user.is_verified = True


def _consume_valid_reset_otp(db: Session, user: User, otp: str) -> None:
    from app.modules.auth.otp_models import PasswordResetOtp

    settings = get_settings()
    now = datetime.now(UTC)
    row = db.scalar(
        select(PasswordResetOtp)
        .where(
            PasswordResetOtp.user_id == user.id,
            PasswordResetOtp.used.is_(False),
            PasswordResetOtp.expires_at > now,
        )
        .order_by(PasswordResetOtp.created_at.desc())
    )
    if not row:
        raise AuthException(INVALID_RESET_OTP, status.HTTP_400_BAD_REQUEST)

    if row.attempts >= settings.password_otp_max_attempts:
        row.used = True
        raise AuthException(INVALID_RESET_OTP, status.HTTP_400_BAD_REQUEST)

    if not verify_password(otp.strip(), row.code_hash):
        row.attempts += 1
        if row.attempts >= settings.password_otp_max_attempts:
            row.used = True
        raise AuthException(INVALID_RESET_OTP, status.HTTP_400_BAD_REQUEST)

    row.used = True


def verify_password_reset_otp(db: Session, email: str, user_id: int, otp: str) -> str:
    """Validate OTP and return a short-lived token for setting a new password."""
    user = resolve_forgot_password_user(db, email, user_id)
    if not user:
        raise AuthException(INVALID_RESET_OTP, status.HTTP_400_BAD_REQUEST)
    _consume_valid_reset_otp(db, user, otp)
    return create_email_token(
        subject=str(user.id),
        purpose="password_reset",
        expires_minutes=get_settings().password_otp_expire_minutes,
    )


def reset_password_with_token(db: Session, reset_token: str, new_password: str) -> None:
    try:
        payload = decode_token(reset_token)
    except ValueError as exc:
        raise AuthException(INVALID_RESET_TOKEN, status.HTTP_400_BAD_REQUEST) from exc
    if payload.get("type") != "password_reset":
        raise AuthException(INVALID_RESET_TOKEN, status.HTTP_400_BAD_REQUEST)
    try:
        user = db.get(User, int(payload.get("sub")))
    except (TypeError, ValueError) as exc:
        raise AuthException(INVALID_RESET_TOKEN, status.HTTP_400_BAD_REQUEST) from exc
    if not user:
        raise AuthException(INVALID_RESET_TOKEN, status.HTTP_400_BAD_REQUEST)
    user.password_hash = hash_password(new_password)
    user.updated_at = datetime.now(UTC)
    db.execute(update(SessionToken).where(SessionToken.user_id == user.id).values(revoked=True))


def reset_password_with_otp(
    db: Session,
    email: str,
    otp: str,
    new_password: str,
    *,
    user_id: int,
) -> None:
    """One-shot OTP + password (legacy). Prefer verify_password_reset_otp + reset_password_with_token."""
    user = resolve_forgot_password_user(db, email, user_id)
    if not user:
        raise AuthException(INVALID_RESET_OTP, status.HTTP_400_BAD_REQUEST)
    _consume_valid_reset_otp(db, user, otp)
    user.password_hash = hash_password(new_password)
    user.updated_at = datetime.now(UTC)
    db.execute(update(SessionToken).where(SessionToken.user_id == user.id).values(revoked=True))


def issue_reset_password_token(user: User) -> str:
    """Deprecated JWT reset; prefer issue_password_reset_otp."""
    return create_email_token(subject=str(user.id), purpose="reset_password", expires_minutes=30)


def reset_password(db: Session, token: str, new_password: str) -> None:
    """Deprecated JWT reset; prefer reset_password_with_otp."""
    payload = decode_token(token)
    if payload.get("type") != "reset_password":
        raise AuthException(INVALID_RESET_TOKEN, status.HTTP_400_BAD_REQUEST)
    try:
        user = db.get(User, int(payload.get("sub")))
    except (TypeError, ValueError) as exc:
        raise AuthException(INVALID_RESET_TOKEN, status.HTTP_400_BAD_REQUEST) from exc
    if not user:
        raise AuthException("User not found.", status.HTTP_404_NOT_FOUND)
    user.password_hash = hash_password(new_password)
    user.updated_at = datetime.now(UTC)
    db.execute(update(SessionToken).where(SessionToken.user_id == user.id).values(revoked=True))

def _default_username(user: User) -> str:
    local = (user.email or "").split("@", 1)[0].strip().lower()
    return local or "user"


def _find_settings_by_username(db: Session, username: str, exclude_user_id: int | None = None) -> UserSettings | None:
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
