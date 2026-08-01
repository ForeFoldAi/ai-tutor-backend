from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from fastapi import status
from sqlalchemy.orm import Session

from app.modules.auth.constants import Role
from app.modules.auth.exceptions import AuthException
from app.modules.auth.public_ids import to_user_response
from app.modules.auth.service import list_schools_with_stats, list_users
from app.modules.live_sessions.service import list_student_sessions, list_tutor_sessions
from app.modules.school_admin.classes.service import list_classes
from app.modules.school_admin.classes.validators import is_individual_tutor
from app.modules.school_admin.credentials.constants import MAX_PAGE_LIMIT as CREDS_MAX
from app.modules.school_admin.credentials.service import list_credentials
from app.modules.school_admin.students.constants import MAX_PAGE_LIMIT as STUDENTS_MAX
from app.modules.school_admin.students.service import list_students
from app.modules.school_admin.teachers.constants import MAX_PAGE_LIMIT as TEACHERS_MAX
from app.modules.school_admin.teachers.service import list_teachers
from app.modules.search.constants import (
    ENTITY_CLASSES,
    ENTITY_CREDENTIALS,
    ENTITY_HREF,
    ENTITY_LESSONS,
    ENTITY_ROLES,
    ENTITY_SCHOOLS,
    ENTITY_SESSIONS,
    ENTITY_SUBJECTS,
    ENTITY_STUDENTS,
    ENTITY_TEACHERS,
    ENTITY_USERS,
    GLOBAL_PER_ENTITY_LIMIT,
    ROLE_GLOBAL_ENTITIES,
)
from app.modules.search.schemas import EntitySearchResponse, GlobalSearchResponse, SearchHit
from app.modules.student_learning import service as learning_service
from app.modules.teacher.students.constants import MAX_PAGE_LIMIT as TUTOR_STUDENTS_MAX
from app.modules.teacher.students.service import list_assigned_students
from app.modules.users.models import User

_SESSION_STATUS_RANK = {"live": 0, "upcoming": 1, "completed": 2}

def _require_q(q: str | None) -> str:
    term = (q or "").strip()
    if not term:
        raise AuthException("Search query is required.", status.HTTP_400_BAD_REQUEST)
    return term


def _assert_entity_allowed(actor: User, entity: str) -> None:
    allowed = ENTITY_ROLES.get(entity)
    if allowed is None:
        raise AuthException("Unknown search entity.", status.HTTP_404_NOT_FOUND)
    if actor.role not in allowed:
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    # School-tagged tutors cannot manage credentials (same rule as credentials service).
    if entity == ENTITY_CREDENTIALS and actor.role == Role.TUTOR and not is_individual_tutor(actor):
        raise AuthException("Forbidden.", status.HTTP_403_FORBIDDEN)
    # Master admin classes require school_id on list_classes — skip tab search without context.
    if entity == ENTITY_CLASSES and actor.role == Role.MASTER_ADMIN:
        raise AuthException(
            "Class search requires a school context; use school admin scope.",
            status.HTTP_400_BAD_REQUEST,
        )


def _href_for(actor: User, entity: str) -> str:
    if entity == ENTITY_STUDENTS and actor.role == Role.TUTOR:
        return "/tutor/students"
    if entity == ENTITY_SESSIONS and actor.role == Role.STUDENT:
        return "/live-classes"
    if entity == ENTITY_CLASSES and actor.role == Role.TUTOR:
        return "/classes"
    if entity == ENTITY_CREDENTIALS and actor.role == Role.TUTOR:
        return "/credentials"
    return ENTITY_HREF[entity]


def _lesson_href(
    *,
    board: str,
    class_level: str,
    subject_name: str,
    subject_id: int,
    chapter_id: int,
    chapter_name: str | None,
) -> str:
    params: dict[str, str] = {
        "board": board,
        "class": class_level,
        "subject": subject_name,
        "subjectId": str(subject_id),
        "chapters": str(chapter_id),
    }
    if chapter_name:
        params["chapterNames"] = chapter_name
    return f"/ai-tutor?{urlencode(params)}"


def _hit(entity: str, id_: str | int, title: str, subtitle: str | None, href: str) -> SearchHit:
    return SearchHit(entity=entity, id=str(id_), title=title, subtitle=subtitle, href=href)


def _model_dump(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    if isinstance(item, dict):
        return item
    return dict(item)


def _filter_text(haystack: str, q: str) -> bool:
    return q.lower() in haystack.lower()


def _search_students_items(db: Session, actor: User, q: str, *, limit: int) -> tuple[list[Any], int]:
    if actor.role == Role.TUTOR:
        resp = list_assigned_students(db, actor, q=q, limit=limit, offset=0)
    else:
        resp = list_students(db, actor, q=q, limit=limit, offset=0)
    return list(resp.items), int(resp.meta.total)


def _search_teachers_items(db: Session, actor: User, q: str, *, limit: int) -> tuple[list[Any], int]:
    resp = list_teachers(db, actor, q=q, limit=limit, offset=0)
    return list(resp.items), int(resp.meta.total)


def _search_classes_items(db: Session, actor: User, q: str, *, limit: int | None) -> tuple[list[Any], int]:
    resp = list_classes(db, actor, q=q, limit=limit, offset=0)
    return list(resp.items), int(resp.meta.total)


def _search_credentials_items(db: Session, actor: User, q: str, *, limit: int) -> tuple[list[Any], int]:
    resp = list_credentials(db, actor, q=q, limit=limit, offset=0)
    return list(resp.items), int(resp.meta.total)


def _search_sessions_items(db: Session, actor: User, q: str) -> tuple[list[Any], int]:
    if actor.role == Role.TUTOR:
        rows = list_tutor_sessions(db, actor)
    elif actor.role == Role.STUDENT:
        # Live Classes UI hides completed — mirror that so search isn't flooded by history.
        rows = [r for r in list_student_sessions(db, actor) if r.status != "completed"]
    else:
        return [], 0
    matched = [
        r
        for r in rows
        if _filter_text(
            " ".join(
                [
                    r.title,
                    r.subject,
                    r.chapter or "",
                    r.grade,
                    r.section,
                    r.tutor_name,
                    r.status,
                ]
            ),
            q,
        )
    ]
    matched.sort(
        key=lambda r: (
            _SESSION_STATUS_RANK.get(r.status, 9),
            r.starts_at,
        )
    )
    return matched, len(matched)


def _search_subjects_items(db: Session, actor: User, q: str) -> tuple[list[Any], int]:
    if actor.role != Role.STUDENT:
        return [], 0
    overview = learning_service.get_overview(db, actor)
    matched = [s for s in overview.subjects if _filter_text(s.subject_name, q)]
    return matched, len(matched)


def _lesson_dicts_from_overview(overview: Any, q: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for subject in overview.subjects:
        for chapter in subject.chapters:
            hay = " ".join(
                [
                    chapter.chapter or "",
                    chapter.file_name or "",
                    subject.subject_name,
                    chapter.status,
                ]
            )
            if not _filter_text(hay, q):
                continue
            matched.append(
                {
                    "id": chapter.id,
                    "title": chapter.chapter or chapter.file_name or subject.subject_name,
                    "subtitle": f"{subject.subject_name} · {chapter.progress}%",
                    "board": str(subject.board),
                    "class_level": str(subject.class_level),
                    "subject_name": subject.subject_name,
                    "subject_id": subject.id,
                    "chapter_id": chapter.id,
                    "chapter_name": chapter.chapter,
                }
            )
            if limit is not None and len(matched) >= limit:
                return matched
    return matched


def _search_lessons_items(db: Session, actor: User, q: str) -> tuple[list[Any], int]:
    if actor.role != Role.STUDENT:
        return [], 0
    overview = learning_service.get_overview(db, actor)
    matched = _lesson_dicts_from_overview(overview, q)
    return matched, len(matched)


def _search_users_items(db: Session, actor: User, q: str) -> tuple[list[Any], int]:
    rows = list_users(db, actor)
    matched = [
        to_user_response(db, u)
        for u in rows
        if _filter_text(f"{u.full_name} {u.email} {u.phone or ''} {u.role}", q)
    ]
    return matched, len(matched)


def _search_schools_items(db: Session, actor: User, q: str) -> tuple[list[Any], int]:
    rows = list_schools_with_stats(db, actor)
    matched = [
        r
        for r in rows
        if _filter_text(f"{r.name} {r.branch or ''} {r.board or ''}", q)
    ]
    return matched, len(matched)


def _items_to_hits(actor: User, entity: str, items: list[Any]) -> list[SearchHit]:
    href = _href_for(actor, entity)
    hits: list[SearchHit] = []
    for item in items:
        if entity == ENTITY_STUDENTS:
            hits.append(
                _hit(
                    entity,
                    item.id,
                    item.full_name,
                    getattr(item, "email", None) or getattr(item, "user_id", None),
                    href,
                )
            )
        elif entity == ENTITY_TEACHERS:
            hits.append(_hit(entity, item.id, item.full_name, item.email, href))
        elif entity == ENTITY_CLASSES:
            title = f"Grade {item.grade} · {item.section}"
            hits.append(_hit(entity, item.id, title, item.curriculum, href))
        elif entity == ENTITY_CREDENTIALS:
            hits.append(
                _hit(
                    entity,
                    item.id,
                    item.name,
                    f"{item.role} · {item.user_id}",
                    href,
                )
            )
        elif entity == ENTITY_SESSIONS:
            hits.append(
                _hit(
                    entity,
                    item.id,
                    item.title,
                    f"{item.subject} · {item.status} · Grade {item.grade}{item.section}",
                    href,
                )
            )
        elif entity == ENTITY_SUBJECTS:
            hits.append(
                _hit(
                    entity,
                    item.id,
                    item.subject_name,
                    f"{item.progress}% · {item.completed_chapters}/{item.total_chapters} chapters",
                    f"/ai-learning-studio/subject/{item.id}",
                )
            )
        elif entity == ENTITY_LESSONS:
            hits.append(
                _hit(
                    entity,
                    item["id"],
                    item["title"],
                    item["subtitle"],
                    _lesson_href(
                        board=item["board"],
                        class_level=item["class_level"],
                        subject_name=item["subject_name"],
                        subject_id=item["subject_id"],
                        chapter_id=item["chapter_id"],
                        chapter_name=item["chapter_name"],
                    ),
                )
            )
        elif entity == ENTITY_USERS:
            hits.append(
                _hit(entity, item.id, item.full_name, f"{item.role} · {item.email}", href)
            )
        elif entity == ENTITY_SCHOOLS:
            hits.append(
                _hit(entity, item.id, item.name, item.branch or item.board, href)
            )
    return hits


def _fetch_entity(
    db: Session,
    actor: User,
    entity: str,
    q: str,
    *,
    limit: int | None,
) -> tuple[list[Any], int]:
    if entity == ENTITY_STUDENTS:
        cap = limit if limit is not None else (
            TUTOR_STUDENTS_MAX if actor.role == Role.TUTOR else STUDENTS_MAX
        )
        return _search_students_items(db, actor, q, limit=cap)
    if entity == ENTITY_TEACHERS:
        return _search_teachers_items(db, actor, q, limit=limit or TEACHERS_MAX)
    if entity == ENTITY_CLASSES:
        # None = all rows (classes list already supports this).
        return _search_classes_items(db, actor, q, limit=limit)
    if entity == ENTITY_CREDENTIALS:
        return _search_credentials_items(db, actor, q, limit=limit or CREDS_MAX)
    if entity == ENTITY_SESSIONS:
        return _search_sessions_items(db, actor, q)
    if entity == ENTITY_SUBJECTS:
        return _search_subjects_items(db, actor, q)
    if entity == ENTITY_LESSONS:
        return _search_lessons_items(db, actor, q)
    if entity == ENTITY_USERS:
        return _search_users_items(db, actor, q)
    if entity == ENTITY_SCHOOLS:
        return _search_schools_items(db, actor, q)
    raise AuthException("Unknown search entity.", status.HTTP_404_NOT_FOUND)


def global_search(db: Session, actor: User, q: str | None) -> GlobalSearchResponse:
    term = (q or "").strip()
    if not term:
        return GlobalSearchResponse(q="", hits=[])

    entities = list(ROLE_GLOBAL_ENTITIES.get(actor.role, ()))
    # Individual tutors also search classes + credentials from the dashboard.
    if actor.role == Role.TUTOR and is_individual_tutor(actor):
        for extra in (ENTITY_CLASSES, ENTITY_CREDENTIALS):
            if extra not in entities:
                entities.append(extra)

    # ponytail: one overview for subjects+lessons — upgrade if more student entities need it.
    student_overview = None
    if actor.role == Role.STUDENT and (
        ENTITY_SUBJECTS in entities or ENTITY_LESSONS in entities
    ):
        student_overview = learning_service.get_overview(db, actor)

    hits: list[SearchHit] = []
    for entity in entities:
        try:
            _assert_entity_allowed(actor, entity)
        except AuthException:
            continue
        limit = GLOBAL_PER_ENTITY_LIMIT
        if entity == ENTITY_SUBJECTS and student_overview is not None:
            items = [
                s for s in student_overview.subjects if _filter_text(s.subject_name, term)
            ][:limit]
        elif entity == ENTITY_LESSONS and student_overview is not None:
            items = _lesson_dicts_from_overview(student_overview, term, limit=limit)
        elif entity == ENTITY_CLASSES:
            items, _total = _fetch_entity(db, actor, entity, term, limit=limit)
        elif entity in (ENTITY_SESSIONS, ENTITY_USERS, ENTITY_SCHOOLS):
            items, _total = _fetch_entity(db, actor, entity, term, limit=None)
            items = items[:GLOBAL_PER_ENTITY_LIMIT]
        else:
            items, _total = _fetch_entity(db, actor, entity, term, limit=limit)
        hits.extend(_items_to_hits(actor, entity, items))

    return GlobalSearchResponse(q=term, hits=hits)


def entity_search(db: Session, actor: User, entity: str, q: str | None) -> EntitySearchResponse:
    key = (entity or "").strip().lower()
    _assert_entity_allowed(actor, key)
    term = _require_q(q)

    # Tab search: return all matches (clamped at each module's MAX where applicable).
    if key == ENTITY_CLASSES:
        items, total = _fetch_entity(db, actor, key, term, limit=None)
    else:
        items, total = _fetch_entity(db, actor, key, term, limit=None)

    return EntitySearchResponse(
        entity=key,
        q=term,
        total=total,
        items=[_model_dump(i) for i in items],
    )
