"""Live tutor sessions — create/list for tutors; list/join for matching students."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.live_sessions.models import LiveSessionJoin, LiveTutorSession
from app.modules.live_sessions.schemas import (
    LiveSessionCreateRequest,
    LiveSessionJoinResponse,
    LiveSessionOut,
)
from app.modules.notifications.service import notify_users
from app.modules.users.models import User


def _session_status(starts_at: datetime, duration_minutes: int, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    start = starts_at if starts_at.tzinfo else starts_at.replace(tzinfo=UTC)
    end = start + timedelta(minutes=max(duration_minutes, 1))
    if now < start:
        return "upcoming"
    if now <= end:
        return "live"
    return "completed"


def _student_class_keys(student: User) -> set[tuple[str, str]]:
    """(grade, section) pairs from teaching_classes."""
    keys: set[tuple[str, str]] = set()
    for entry in student.teaching_classes or []:
        if not isinstance(entry, dict):
            continue
        grade = str(entry.get("grade", "")).strip()
        sections = entry.get("sections") or []
        for sec in sections:
            section = str(sec).strip().upper()
            if grade and section:
                keys.add((grade, section))
    return keys


def _attendee_counts(db: Session, session_ids: list[int]) -> dict[int, int]:
    if not session_ids:
        return {}
    rows = db.execute(
        select(LiveSessionJoin.session_id, func.count(LiveSessionJoin.id))
        .where(LiveSessionJoin.session_id.in_(session_ids))
        .group_by(LiveSessionJoin.session_id)
    ).all()
    return {int(sid): int(cnt) for sid, cnt in rows}


def _to_out(
    row: LiveTutorSession,
    tutor_name: str,
    attendees: int,
    *,
    joined: bool = False,
    include_link: bool = True,
) -> LiveSessionOut:
    return LiveSessionOut(
        id=row.id,
        title=row.title,
        subject=row.subject,
        chapter_id=row.chapter_id,
        chapter=row.chapter,
        grade=row.grade,
        section=row.section,
        curriculum=row.curriculum,
        starts_at=row.starts_at,
        duration_minutes=row.duration_minutes,
        meeting_link=row.meeting_link if include_link else None,
        notes=row.notes,
        tutor_id=row.tutor_id,
        tutor_name=tutor_name,
        attendees=attendees,
        status=_session_status(row.starts_at, row.duration_minutes),  # type: ignore[arg-type]
        joined=joined,
    )


def create_session(db: Session, tutor: User, payload: LiveSessionCreateRequest) -> LiveSessionOut:
    if tutor.role != Role.TUTOR:
        raise AuthException("Only tutors can create sessions.", status.HTTP_403_FORBIDDEN)
    row = LiveTutorSession(
        tutor_id=tutor.id,
        school_id=tutor.school_id,
        title=payload.title.strip(),
        subject=payload.subject.strip(),
        chapter_id=payload.chapter_id,
        chapter=payload.chapter,
        grade=payload.grade.strip(),
        section=payload.section.strip().upper(),
        curriculum=(payload.curriculum or tutor.teaching_board or None),
        starts_at=payload.starts_at if payload.starts_at.tzinfo else payload.starts_at.replace(tzinfo=UTC),
        duration_minutes=payload.duration_minutes,
        meeting_link=payload.meeting_link,
        notes=payload.notes.strip() if payload.notes else None,
    )
    db.add(row)
    db.flush()
    # Reuse assignment roster matching so session alerts hit the same students.
    from app.modules.teacher.assignments.service import match_students_for_scope

    students = match_students_for_scope(
        db,
        tutor,
        grade=row.grade,
        section=row.section,
        curriculum=row.curriculum or "",
        subject=row.subject,
    )
    if students:
        when = row.starts_at.strftime("%b %d, %Y %H:%M") if row.starts_at else ""
        notify_users(
            db,
            recipient_ids=[s.id for s in students],
            type="session_created",
            title="New live session",
            body=f"{row.title} — {row.subject}" + (f" · {when}" if when else ""),
            actor_id=tutor.id,
            school_id=tutor.school_id,
            link="/live-classes",
        )
    return _to_out(row, tutor.full_name or "Tutor", 0, include_link=True)


def list_tutor_sessions(db: Session, tutor: User) -> list[LiveSessionOut]:
    if tutor.role != Role.TUTOR:
        raise AuthException("Only tutors can list their sessions.", status.HTTP_403_FORBIDDEN)
    rows = list(
        db.scalars(
            select(LiveTutorSession)
            .where(LiveTutorSession.tutor_id == tutor.id)
            .order_by(LiveTutorSession.starts_at.desc())
        )
    )
    counts = _attendee_counts(db, [r.id for r in rows])
    name = tutor.full_name or "Tutor"
    return [_to_out(r, name, counts.get(r.id, 0), include_link=True) for r in rows]


def delete_tutor_session(db: Session, tutor: User, session_id: int) -> None:
    row = db.get(LiveTutorSession, session_id)
    if row is None or row.tutor_id != tutor.id:
        raise AuthException("Session not found.", status.HTTP_404_NOT_FOUND)
    db.delete(row)
    db.flush()


def update_tutor_session(
    db: Session,
    tutor: User,
    session_id: int,
    payload: LiveSessionCreateRequest,
) -> LiveSessionOut:
    if tutor.role != Role.TUTOR:
        raise AuthException("Only tutors can update sessions.", status.HTTP_403_FORBIDDEN)
    row = db.get(LiveTutorSession, session_id)
    if row is None or row.tutor_id != tutor.id:
        raise AuthException("Session not found.", status.HTTP_404_NOT_FOUND)

    row.title = payload.title.strip()
    row.subject = payload.subject.strip()
    row.chapter_id = payload.chapter_id
    row.chapter = payload.chapter
    row.grade = payload.grade.strip()
    row.section = payload.section.strip().upper()
    row.curriculum = payload.curriculum or tutor.teaching_board or None
    row.starts_at = (
        payload.starts_at if payload.starts_at.tzinfo else payload.starts_at.replace(tzinfo=UTC)
    )
    row.duration_minutes = payload.duration_minutes
    row.meeting_link = payload.meeting_link
    row.notes = payload.notes.strip() if payload.notes else None
    db.add(row)
    db.flush()
    counts = _attendee_counts(db, [row.id])
    return _to_out(row, tutor.full_name or "Tutor", counts.get(row.id, 0), include_link=True)


def list_student_sessions(db: Session, student: User) -> list[LiveSessionOut]:
    if student.role != Role.STUDENT:
        raise AuthException("Only students can view class sessions.", status.HTTP_403_FORBIDDEN)

    keys = _student_class_keys(student)

    if student.school_id is not None:
        if not keys:
            return []
        grade_section_filters = [
            (LiveTutorSession.grade == g) & (LiveTutorSession.section == s) for g, s in keys
        ]
        rows = list(
            db.scalars(
                select(LiveTutorSession)
                .where(
                    LiveTutorSession.school_id == student.school_id,
                    or_(*grade_section_filters),
                )
                .order_by(LiveTutorSession.starts_at.asc())
            )
        )
    else:
        if student.created_by is None:
            return []
        q = select(LiveTutorSession).where(LiveTutorSession.tutor_id == student.created_by)
        rows = list(db.scalars(q.order_by(LiveTutorSession.starts_at.asc())))
        if keys:
            rows = [r for r in rows if (r.grade.strip(), r.section.strip().upper()) in keys]

    if not rows:
        return []

    tutor_ids = {r.tutor_id for r in rows}
    tutors = {
        u.id: u for u in db.scalars(select(User).where(User.id.in_(list(tutor_ids)))).all()
    }
    # Drop sessions from tutors who excluded this student / inactive
    filtered: list[LiveTutorSession] = []
    for r in rows:
        tutor = tutors.get(r.tutor_id)
        if tutor is None or not tutor.is_active:
            continue
        excluded = {
            int(x)
            for x in (tutor.excluded_student_ids or [])
            if str(x).strip().lstrip("-").isdigit()
        }
        if student.id in excluded:
            continue
        filtered.append(r)
    rows = filtered
    if not rows:
        return []

    counts = _attendee_counts(db, [r.id for r in rows])
    joined_ids = set(
        db.scalars(
            select(LiveSessionJoin.session_id).where(
                LiveSessionJoin.student_id == student.id,
                LiveSessionJoin.session_id.in_([r.id for r in rows]),
            )
        ).all()
    )

    out: list[LiveSessionOut] = []
    for r in rows:
        tutor = tutors[r.tutor_id]
        out.append(
            _to_out(
                r,
                tutor.full_name or "Tutor",
                counts.get(r.id, 0),
                joined=r.id in joined_ids,
                include_link=True,
            )
        )
    return out


def join_session(db: Session, student: User, session_id: int) -> LiveSessionJoinResponse:
    if student.role != Role.STUDENT:
        raise AuthException("Only students can join sessions.", status.HTTP_403_FORBIDDEN)

    visible_ids = {s.id for s in list_student_sessions(db, student)}
    if session_id not in visible_ids:
        raise AuthException("You are not invited to this session.", status.HTTP_403_FORBIDDEN)

    row = db.get(LiveTutorSession, session_id)
    if row is None:
        raise AuthException("Session not found.", status.HTTP_404_NOT_FOUND)

    status_label = _session_status(row.starts_at, row.duration_minutes)
    if status_label == "upcoming":
        raise AuthException(
            "This session has not started yet. Join when it goes live.",
            status.HTTP_400_BAD_REQUEST,
        )
    if status_label == "completed":
        raise AuthException(
            "This session has ended.",
            status.HTTP_400_BAD_REQUEST,
        )
    if not row.meeting_link:
        raise AuthException(
            "Meeting link is not available yet. Ask your tutor to add one.",
            status.HTTP_400_BAD_REQUEST,
        )

    existing = db.scalar(
        select(LiveSessionJoin).where(
            LiveSessionJoin.session_id == session_id,
            LiveSessionJoin.student_id == student.id,
        )
    )
    if existing is None:
        existing = LiveSessionJoin(session_id=session_id, student_id=student.id)
        db.add(existing)
        db.flush()
        db.refresh(existing)

    return LiveSessionJoinResponse(
        session_id=session_id,
        meeting_link=row.meeting_link,
        joined_at=existing.joined_at,
    )
