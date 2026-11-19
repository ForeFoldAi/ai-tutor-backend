from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import status
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.orm import Session

IST = ZoneInfo("Asia/Kolkata")

from app.core.security import hash_password
from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.auth.public_ids import require_user_by_account_id
from app.modules.auth.service import get_or_create_user_settings
from app.modules.school_admin.credentials.constants import (
    DEFAULT_PAGE_LIMIT,
    DELIVERY_EMAIL_SENT,
    DELIVERY_FAILED,
    DELIVERY_IN_PROCESS,
    DELIVERY_LABELS,
    DELIVERY_NOT_SENT,
    FIRST_LOGIN_COMPLETED,
    FIRST_LOGIN_NOT_STARTED,
    FIRST_LOGIN_PENDING,
    MAX_PAGE_LIMIT,
    PAGE_SIZE_OPTIONS,
    ROLE_STUDENT,
    ROLE_TEACHER,
)
from app.modules.school_admin.credentials.passwords import make_credential_password
from app.modules.school_admin.credentials.schemas import (
    CredentialActionRequest,
    CredentialCandidate,
    CredentialCandidatesResponse,
    CredentialGenerateResponse,
    CredentialGeneratedItem,
    CredentialListResponse,
    CredentialMetrics,
    CredentialRecord,
    CredentialSendResponse,
    PaginationMeta,
)
from app.modules.school_admin.credentials.xlsx import build_xlsx
from app.modules.school_admin.teachers.validators import resolve_target_school_id
from app.modules.sessions.models import SessionToken
from app.modules.users.models import User, UserSettings


def _is_individual_tutor(actor: User) -> bool:
    return actor.role == Role.TUTOR and actor.school_id is None


def _assert_credential_manager(actor: User) -> None:
    if actor.role in (Role.SCHOOL_ADMIN, Role.MASTER_ADMIN):
        return
    if _is_individual_tutor(actor):
        return
    raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)


def _scoped_users_where(actor: User):
    """School admin/master: school roster. Individual tutor: owned students only.

    Inactive teachers/students are omitted from credentials entirely.
    """
    _assert_credential_manager(actor)
    if _is_individual_tutor(actor):
        return (
            User.created_by == actor.id,
            User.role == Role.STUDENT,
            User.school_id.is_(None),
            User.is_active.is_(True),
        )
    school_id = resolve_target_school_id(actor)
    return (
        User.school_id == school_id,
        User.role.in_([Role.TUTOR, Role.STUDENT]),
        User.is_active.is_(True),
    )


def _assert_user_in_scope(actor: User, user: User, role_filter: Role) -> None:
    if not user.is_active:
        raise AuthException(
            f"User {user.id} is inactive.",
            status.HTTP_404_NOT_FOUND,
        )
    if _is_individual_tutor(actor):
        if (
            user.created_by != actor.id
            or user.role != Role.STUDENT
            or user.school_id is not None
            or role_filter != Role.STUDENT
        ):
            raise AuthException(
                f"User {user.id} not found in your students.",
                status.HTTP_404_NOT_FOUND,
            )
        return
    school_id = resolve_target_school_id(actor)
    if user.school_id != school_id or user.role != role_filter:
        raise AuthException(f"User {user.id} not found in this school.", status.HTTP_404_NOT_FOUND)


def _reject_teacher_role_for_individual(actor: User, role_filter: Role | None) -> None:
    if _is_individual_tutor(actor) and role_filter == Role.TUTOR:
        raise AuthException(
            "Teachers are not managed for individual accounts.",
            status.HTTP_400_BAD_REQUEST,
        )


def _username_for_user(db: Session, user_id: int) -> str:
    row = db.get(UserSettings, user_id)
    if row and row.username:
        return row.username
    user = db.get(User, user_id)
    if user:
        settings = get_or_create_user_settings(db, user)
        return settings.username or user.email.split("@", 1)[0]
    return ""


def _class_fields(user: User) -> tuple[str | None, str | None, str | None]:
    raw = user.teaching_classes
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        grade = str(raw[0].get("grade", "")).strip() or None
        secs = raw[0].get("sections") or []
        section = str(secs[0]).strip().upper() if secs else None
        curriculum = str(raw[0].get("curriculum", "")).strip() or None
        if not curriculum and user.teaching_board:
            curriculum = user.teaching_board.strip() or None
        return grade, section, curriculum
    board = user.teaching_board.strip() if user.teaching_board else None
    return None, None, board


def _role_label(role: Role) -> str:
    return ROLE_TEACHER if role == Role.TUTOR else ROLE_STUDENT


def _parse_role(role: str | None) -> Role | None:
    if not role or role == "all":
        return None
    normalized = role.strip().lower()
    if normalized in ("teacher", "tutor"):
        return Role.TUTOR
    if normalized == "student":
        return Role.STUDENT
    raise AuthException("role must be Teacher or Student.", status.HTTP_400_BAD_REQUEST)


def _delivery_label(raw: str | None, generated: bool) -> str:
    if not generated:
        return DELIVERY_LABELS[DELIVERY_NOT_SENT]
    key = (raw or DELIVERY_NOT_SENT).strip().lower()
    return DELIVERY_LABELS.get(key, DELIVERY_LABELS[DELIVERY_NOT_SENT])


def _first_login_status(*, generated: bool, has_logged_in: bool) -> str:
    if has_logged_in:
        return FIRST_LOGIN_COMPLETED
    if generated:
        return FIRST_LOGIN_PENDING
    return FIRST_LOGIN_NOT_STARTED


def _format_shared(shared_at: datetime | None) -> str | None:
    if not shared_at:
        return None
    return shared_at.astimezone(UTC).strftime("%Y-%m-%d")


def _format_last_login(last_login: datetime | None) -> str | None:
    if not last_login:
        return None
    # Credentials UI is India-facing — show wall time in IST (no tz suffix in UI).
    return last_login.astimezone(IST).strftime("%d %b %Y, %I:%M %p")


def _last_login_subquery():
    return (
        select(
            SessionToken.user_id.label("uid"),
            func.max(SessionToken.created_at).label("last_login"),
        )
        .group_by(SessionToken.user_id)
        .subquery()
    )


def _scoped_users_stmt(actor: User, role: Role | None = None):
    stmt = select(User).where(*_scoped_users_where(actor))
    if role is not None:
        stmt = stmt.where(User.role == role)
    return stmt


def credential_stats(db: Session, actor: User) -> CredentialMetrics:
    last_login = _last_login_subquery()
    stmt = (
        select(User, last_login.c.last_login)
        .outerjoin(last_login, last_login.c.uid == User.id)
        .where(*_scoped_users_where(actor))
    )
    rows = list(db.execute(stmt).all())
    generated = 0
    teachers_pending = 0
    students_pending = 0
    not_shared = 0
    week_ago = datetime.now(UTC) - timedelta(days=7)
    generated_week = 0
    for user, last in rows:
        if user.credentials_generated_at:
            generated += 1
            if user.credentials_generated_at >= week_ago:
                generated_week += 1
            if user.credentials_shared_at is None:
                not_shared += 1
            if last is None:
                if user.role == Role.TUTOR:
                    teachers_pending += 1
                else:
                    students_pending += 1
    return CredentialMetrics(
        generated=generated,
        teachers_pending_login=teachers_pending,
        students_pending_login=students_pending,
        not_shared=not_shared,
        generated_trend=f"{generated_week} this week",
        teachers_pending_trend=f"{teachers_pending} awaiting login",
        students_pending_trend=f"{students_pending} awaiting login",
        not_shared_trend=f"{not_shared} not shared",
        not_shared_trend_up=not_shared > 0,
    )


def list_credentials(
    db: Session,
    actor: User,
    *,
    q: str | None = None,
    role: str | None = None,
    first_login_status: str | None = None,
    delivery_status: str | None = None,
    shared: str | None = None,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
) -> CredentialListResponse:
    limit = max(1, min(limit, MAX_PAGE_LIMIT))
    offset = max(0, offset)
    role_filter = _parse_role(role)
    _reject_teacher_role_for_individual(actor, role_filter)

    last_login = _last_login_subquery()
    stmt = (
        select(User, last_login.c.last_login)
        .outerjoin(last_login, last_login.c.uid == User.id)
        .where(*_scoped_users_where(actor))
    )
    if role_filter is not None:
        stmt = stmt.where(User.role == role_filter)

    if q and q.strip():
        term = f"%{q.strip().lower()}%"
        stmt = stmt.outerjoin(UserSettings, UserSettings.user_id == User.id).where(
            or_(
                func.lower(User.full_name).like(term),
                func.lower(User.email).like(term),
                func.lower(cast(UserSettings.username, String)).like(term),
            )
        )

    rows = list(db.execute(stmt.order_by(User.full_name.asc())).all())

    def include(user: User, last: datetime | None) -> bool:
        generated = user.credentials_generated_at is not None
        fl = _first_login_status(generated=generated, has_logged_in=last is not None)
        delivery = _delivery_label(user.credential_delivery_status, generated)
        if first_login_status and first_login_status != "all" and fl != first_login_status:
            return False
        if delivery_status and delivery_status != "all" and delivery != delivery_status:
            return False
        if shared == "shared" and not user.credentials_shared_at:
            return False
        if shared == "not_shared" and user.credentials_shared_at:
            return False
        return True

    filtered = [(u, last) for u, last in rows if include(u, last)]
    total = len(filtered)
    page = filtered[offset : offset + limit]
    items = []
    for user, last in page:
        generated = user.credentials_generated_at is not None
        grade, section, curriculum = _class_fields(user)
        items.append(
            CredentialRecord(
                id=user.id,
                name=user.full_name,
                role=_role_label(user.role),
                user_id=_username_for_user(db, user.id),
                grade=grade,
                section=section,
                curriculum=curriculum,
                has_credentials=generated,
                credential_shared=_format_shared(user.credentials_shared_at),
                first_login_status=_first_login_status(
                    generated=generated,
                    has_logged_in=last is not None,
                ),
                last_login=_format_last_login(last),
                delivery_status=_delivery_label(
                    user.credential_delivery_status,
                    generated,
                ),
            )
        )
    return CredentialListResponse(
        items=items,
        meta=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            default_limit=DEFAULT_PAGE_LIMIT,
        ),
    )


def list_candidates(
    db: Session,
    actor: User,
    *,
    role: str,
    q: str | None = None,
    grade: str | None = None,
    section: str | None = None,
    credential_status: str | None = None,
) -> CredentialCandidatesResponse:
    role_filter = _parse_role(role)
    if role_filter is None:
        raise AuthException("role is required.", status.HTTP_400_BAD_REQUEST)
    _reject_teacher_role_for_individual(actor, role_filter)

    users = list(db.scalars(_scoped_users_stmt(actor, role_filter).order_by(User.full_name.asc())))
    grades: set[str] = set()
    sections: set[str] = set()
    items: list[CredentialCandidate] = []
    term = (q or "").strip().lower()

    for user in users:
        username = _username_for_user(db, user.id)
        g, s, _curriculum = _class_fields(user)
        if g:
            grades.add(g)
        if s:
            sections.add(s)
        has_creds = user.credentials_generated_at is not None
        if credential_status == "needs" and has_creds:
            continue
        if credential_status == "has" and not has_creds:
            continue
        if term and term not in user.full_name.lower() and term not in username.lower():
            continue
        if role_filter == Role.STUDENT:
            if grade and grade != "all" and g != grade:
                continue
            if section and section != "all" and s != section:
                continue
        detail = None
        if role_filter == Role.STUDENT and (g or s):
            detail = f"Grade {g or '—'} · Section {s or '—'}"
        elif role_filter == Role.TUTOR and user.teaching_board:
            detail = user.teaching_board
        items.append(
            CredentialCandidate(
                id=user.id,
                name=user.full_name,
                role=_role_label(user.role),
                user_id=username,
                grade=g,
                section=s,
                detail=detail,
                has_credentials=has_creds,
            )
        )

    return CredentialCandidatesResponse(
        items=items,
        grades=sorted(grades, key=lambda x: (len(x), x)),
        sections=sorted(sections),
    )


def generate_credentials(
    db: Session,
    actor: User,
    payload: CredentialActionRequest,
) -> CredentialGenerateResponse:
    role_filter = _parse_role(payload.role)
    if role_filter is None:
        raise AuthException("role is required.", status.HTTP_400_BAD_REQUEST)
    _reject_teacher_role_for_individual(actor, role_filter)

    now = datetime.now(UTC)
    items: list[CredentialGeneratedItem] = []
    for user_id in payload.user_ids:
        user = require_user_by_account_id(db, user_id)
        _assert_user_in_scope(actor, user, role_filter)
        username = _username_for_user(db, user.id)
        password = make_credential_password(user.full_name, username)
        user.password_hash = hash_password(password)
        user.credentials_generated_at = now
        if not user.credential_delivery_status:
            user.credential_delivery_status = DELIVERY_NOT_SENT
        user.updated_at = now
        items.append(
            CredentialGeneratedItem(
                id=user.id,
                name=user.full_name,
                role=_role_label(user.role),
                user_id=username,
                password=password,
            )
        )
    db.flush()
    return CredentialGenerateResponse(
        message=f"Generated credentials for {len(items)} user(s).",
        items=items,
    )


def send_credentials(
    db: Session,
    actor: User,
    payload: CredentialActionRequest,
) -> CredentialSendResponse:
    from app.core.config import get_settings
    from app.services.mail import send_credentials_email

    role_filter = _parse_role(payload.role)
    if role_filter is None:
        raise AuthException("role is required.", status.HTTP_400_BAD_REQUEST)
    _reject_teacher_role_for_individual(actor, role_filter)

    settings = get_settings()
    now = datetime.now(UTC)
    updated = 0
    failed = 0
    for user_id in payload.user_ids:
        user = require_user_by_account_id(db, user_id)
        _assert_user_in_scope(actor, user, role_filter)
        username = _username_for_user(db, user.id)
        password = make_credential_password(user.full_name, username)
        user.password_hash = hash_password(password)
        if not user.credentials_generated_at:
            user.credentials_generated_at = now

        # Mark In Process so the table can show progress during a multi-send
        user.credential_delivery_status = DELIVERY_IN_PROCESS
        user.updated_at = now
        db.commit()
        db.refresh(user)

        if not settings.smtp_enabled:
            failed += 1
            user.credential_delivery_status = DELIVERY_FAILED
            user.updated_at = datetime.now(UTC)
            db.commit()
            continue

        ok = send_credentials_email(
            to=user.email,
            full_name=user.full_name,
            username=username,
            password=password,
            role_label=_role_label(user.role),
        )
        if not ok:
            failed += 1
            user.credential_delivery_status = DELIVERY_FAILED
            user.updated_at = datetime.now(UTC)
            db.commit()
            continue

        user.credentials_shared_at = datetime.now(UTC)
        user.credential_delivery_status = DELIVERY_EMAIL_SENT
        user.updated_at = datetime.now(UTC)
        db.commit()
        updated += 1

    if not settings.smtp_enabled:
        return CredentialSendResponse(
            message="Email is not enabled (SMTP_ENABLED=false). Delivery marked Failed.",
            updated=0,
        )
    if failed:
        return CredentialSendResponse(
            message=f"Emailed {updated} credential pack(s); {failed} failed.",
            updated=updated,
        )
    return CredentialSendResponse(
        message=f"Emailed {updated} credential pack(s).",
        updated=updated,
    )


def export_credentials_xlsx(
    db: Session,
    actor: User,
    *,
    q: str | None = None,
    role: str | None = None,
    first_login_status: str | None = None,
    delivery_status: str | None = None,
    shared: str | None = None,
) -> bytes:
    _assert_credential_manager(actor)
    # Reuse list with a high limit to apply the same filters
    listed = list_credentials(
        db,
        actor,
        q=q,
        role=role,
        first_login_status=first_login_status,
        delivery_status=delivery_status,
        shared=shared,
        limit=MAX_PAGE_LIMIT,
        offset=0,
    )
    # If more than one page, walk remainer
    items = list(listed.items)
    total = listed.meta.total
    offset = MAX_PAGE_LIMIT
    while offset < total:
        more = list_credentials(
            db,
            actor,
            q=q,
            role=role,
            first_login_status=first_login_status,
            delivery_status=delivery_status,
            shared=shared,
            limit=MAX_PAGE_LIMIT,
            offset=offset,
        )
        items.extend(more.items)
        offset += MAX_PAGE_LIMIT

    headers = [
        "Name",
        "Role",
        "User ID",
        "Temp Password",
        "Credential Shared",
        "First Login",
        "Last Login",
        "Delivery Status",
    ]
    rows: list[list[str]] = []
    for rec in items:
        user = require_user_by_account_id(db, rec.id)
        if _is_individual_tutor(actor):
            if user.created_by != actor.id or user.school_id is not None:
                continue
        else:
            school_id = resolve_target_school_id(actor)
            if user.school_id != school_id:
                continue
        username = rec.user_id
        password = ""
        if user.credentials_generated_at:
            password = make_credential_password(user.full_name, username)
            # Keep hash in sync with the Excel password (e.g. after formula length bump).
            user.password_hash = hash_password(password)
            user.updated_at = datetime.now(UTC)
        rows.append(
            [
                rec.name,
                rec.role,
                rec.user_id,
                password,
                rec.credential_shared or "",
                rec.first_login_status,
                rec.last_login or "",
                rec.delivery_status,
            ]
        )
    db.flush()
    return build_xlsx(headers, rows)


def page_size_options() -> list[int]:
    return list(PAGE_SIZE_OPTIONS)
