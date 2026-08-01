"""Student learning overview, sessions, progress, streak, reset."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import status
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.modules.auth.exceptions import AuthException
from app.modules.catalog.models import TextbookUpload
from app.modules.student_learning.enrollment import (
    compute_scope_key,
    individual_learning_scope,
    list_enrolled_subjects,
    resolve_student_school_class,
    effective_subject_names,
    enrolled_subject_names,
    _is_individual_student,
)
from app.modules.student_learning.models import (
    StudentChapterProgress,
    StudentLearningStreak,
    StudentStudySession,
    StudentTutorChat,
)
from app.modules.student_learning.schemas import (
    ChapterCompleteResponse,
    ContinueLearningOut,
    LearningChapterOut,
    LearningOverviewResponse,
    LearningStatsOut,
    LearningSubjectOut,
    RecentLessonOut,
    RecommendedTopicOut,
    SessionEndResponse,
    SessionHeartbeatResponse,
    SessionStartRequest,
    SessionStartResponse,
    TutorChatMessageOut,
    TutorChatPutRequest,
    TutorChatResponse,
    TutorChatThreadOut,
)
from app.modules.student_learning.tutor_chat_storage import (
    cap_threads,
    new_thread_id,
    parse_chat_storage,
    sanitize_chat_messages,
    serialize_chat_storage,
    thread_title,
)
from app.modules.users.models import User

# Cap per-heartbeat to avoid tab-sleep inflation
_HEARTBEAT_MAX_DELTA = 60

# ponytail: IST calendar day for streak — use settings.timezone if multi-region.
_STREAK_TZ = ZoneInfo("Asia/Kolkata")


def _streak_today() -> date:
    return datetime.now(_STREAK_TZ).date()


def _apply_streak_decay(streak: StudentLearningStreak | None, *, today: date | None = None) -> int:
    """If a calendar day was missed after last study, current streak is 0 (next study starts at 1)."""
    if streak is None or streak.last_study_date is None:
        return 0
    today = today or _streak_today()
    yesterday = today - timedelta(days=1)
    if streak.last_study_date >= yesterday:
        return int(streak.current_streak or 0)
    if streak.current_streak:
        streak.current_streak = 0
    return 0


def _bump_streak(db: Session, user_id: int, scope_key: str) -> StudentLearningStreak:
    today = _streak_today()
    streak = db.scalar(select(StudentLearningStreak).where(StudentLearningStreak.user_id == user_id))
    if streak is None:
        streak = StudentLearningStreak(
            user_id=user_id,
            current_streak=1,
            longest_streak=1,
            last_study_date=today,
            scope_key=scope_key,
        )
        db.add(streak)
        db.flush()
        return streak

    streak.scope_key = scope_key
    _apply_streak_decay(streak, today=today)
    if streak.last_study_date == today:
        return streak
    if streak.last_study_date == today - timedelta(days=1):
        streak.current_streak = int(streak.current_streak or 0) + 1
    else:
        # Missed one or more days — streak restarts at 1
        streak.current_streak = 1
    streak.longest_streak = max(int(streak.longest_streak or 0), streak.current_streak)
    streak.last_study_date = today
    db.flush()
    return streak


def reset_student_learning(db: Session, user_id: int) -> None:
    db.execute(delete(StudentStudySession).where(StudentStudySession.user_id == user_id))
    db.execute(delete(StudentChapterProgress).where(StudentChapterProgress.user_id == user_id))
    db.execute(delete(StudentLearningStreak).where(StudentLearningStreak.user_id == user_id))
    db.execute(delete(StudentTutorChat).where(StudentTutorChat.user_id == user_id))
    db.flush()


def ensure_scope_or_reset(db: Session, user: User) -> str:
    """If class/subjects changed, wipe learning data and return new scope_key."""
    if _is_individual_student(user):
        _, scope_key = individual_learning_scope(db, user)
    else:
        school_class = resolve_student_school_class(db, user)
        if school_class is None:
            scope_key = compute_scope_key(None, [])
        else:
            mapped_names = enrolled_subject_names(db, school_class)
            names = effective_subject_names(
                db, user=user, school_class=school_class, mapped_names=mapped_names
            )
            scope_key = compute_scope_key(school_class, names)

    streak = db.scalar(select(StudentLearningStreak).where(StudentLearningStreak.user_id == user.id))
    if streak is None:
        return scope_key
    if streak.scope_key != scope_key:
        reset_student_learning(db, user.id)
    return scope_key


def _upsert_progress(
    db: Session,
    *,
    user_id: int,
    chapter_id: int,
    subject_name: str,
    chapter_name: str | None,
    scope_key: str,
    mark_complete: bool = False,
) -> StudentChapterProgress:
    now = datetime.now(UTC)
    row = db.scalar(
        select(StudentChapterProgress).where(
            StudentChapterProgress.user_id == user_id,
            StudentChapterProgress.chapter_id == chapter_id,
        )
    )
    if row is None:
        row = StudentChapterProgress(
            user_id=user_id,
            chapter_id=chapter_id,
            subject_name=subject_name,
            chapter_name=chapter_name,
            status="completed" if mark_complete else "in_progress",
            first_accessed_at=now,
            last_accessed_at=now,
            completed_at=now if mark_complete else None,
            scope_key=scope_key,
            progress_pct=100 if mark_complete else 0,
            topics_total=0,
            covered_topics=[],
        )
        db.add(row)
    else:
        row.last_accessed_at = now
        row.subject_name = subject_name
        if chapter_name:
            row.chapter_name = chapter_name
        row.scope_key = scope_key
        if mark_complete:
            row.status = "completed"
            row.completed_at = now
            row.progress_pct = 100
        elif row.status != "completed":
            row.status = "in_progress"
    db.flush()
    return row


def start_session(db: Session, user: User, payload: SessionStartRequest) -> SessionStartResponse:
    scope_key = ensure_scope_or_reset(db, user)
    now = datetime.now(UTC)
    chapter_name = payload.chapter_name
    if payload.chapter_id is not None:
        tb = db.get(TextbookUpload, payload.chapter_id)
        if tb is None:
            raise AuthException("Chapter not found.", status.HTTP_404_NOT_FOUND)
        chapter_name = chapter_name or tb.chapter
        _upsert_progress(
            db,
            user_id=user.id,
            chapter_id=payload.chapter_id,
            subject_name=payload.subject_name.strip(),
            chapter_name=chapter_name,
            scope_key=scope_key,
        )

    session = StudentStudySession(
        user_id=user.id,
        subject_name=payload.subject_name.strip(),
        chapter_id=payload.chapter_id,
        chapter_name=chapter_name,
        mode=payload.mode,
        agent_mode=payload.agent_mode or "free",
        started_at=now,
        last_heartbeat_at=now,
        duration_seconds=0,
        scope_key=scope_key,
    )
    db.add(session)
    _bump_streak(db, user.id, scope_key)
    db.flush()
    return SessionStartResponse(session_id=session.id, started_at=session.started_at)


def heartbeat_session(db: Session, user: User, session_id: int) -> SessionHeartbeatResponse:
    scope_key = ensure_scope_or_reset(db, user)
    session = db.get(StudentStudySession, session_id)
    if session is None or session.user_id != user.id:
        raise AuthException("Session not found.", status.HTTP_404_NOT_FOUND)
    if session.ended_at is not None:
        return SessionHeartbeatResponse(session_id=session.id, duration_seconds=session.duration_seconds)

    now = datetime.now(UTC)
    delta = int((now - session.last_heartbeat_at).total_seconds())
    delta = max(0, min(delta, _HEARTBEAT_MAX_DELTA))
    session.duration_seconds += delta
    session.last_heartbeat_at = now
    if session.chapter_id is not None:
        _upsert_progress(
            db,
            user_id=user.id,
            chapter_id=session.chapter_id,
            subject_name=session.subject_name,
            chapter_name=session.chapter_name,
            scope_key=scope_key,
        )
    _bump_streak(db, user.id, scope_key)
    db.flush()
    return SessionHeartbeatResponse(session_id=session.id, duration_seconds=session.duration_seconds)


def end_session(db: Session, user: User, session_id: int) -> SessionEndResponse:
    heartbeat_session(db, user, session_id)
    session = db.get(StudentStudySession, session_id)
    if session is None or session.user_id != user.id:
        raise AuthException("Session not found.", status.HTTP_404_NOT_FOUND)
    now = datetime.now(UTC)
    if session.ended_at is None:
        session.ended_at = now
        db.flush()
    from app.services.learning_intelligence.clients.lia_client import emit_study_session_end

    emit_study_session_end(
        student_user_id=user.id,
        subject_name=session.subject_name,
        duration_seconds=session.duration_seconds,
        mode=session.mode,
    )
    return SessionEndResponse(
        session_id=session.id,
        duration_seconds=session.duration_seconds,
        ended_at=session.ended_at,
    )


def complete_chapter(db: Session, user: User, chapter_id: int) -> ChapterCompleteResponse:
    scope_key = ensure_scope_or_reset(db, user)
    tb = db.get(TextbookUpload, chapter_id)
    if tb is None:
        raise AuthException("Chapter not found.", status.HTTP_404_NOT_FOUND)
    row = _upsert_progress(
        db,
        user_id=user.id,
        chapter_id=chapter_id,
        subject_name=tb.subject_name,
        chapter_name=tb.chapter,
        scope_key=scope_key,
        mark_complete=True,
    )
    assert row.completed_at is not None
    from app.services.learning_intelligence.clients.lia_client import emit_chapter_completed

    emit_chapter_completed(
        student_user_id=user.id,
        subject_name=tb.subject_name,
        chapter_name=tb.chapter or "",
        chapter_id=chapter_id,
    )
    return ChapterCompleteResponse(
        chapter_id=chapter_id,
        status="completed",
        completed_at=row.completed_at,
    )


def _messages_to_out(raw: list[dict]) -> list[TutorChatMessageOut]:
    return [
        TutorChatMessageOut(
            id=m["id"],
            role=m["role"],
            content=m["content"],
            created_at=m.get("created_at"),
        )
        for m in raw
    ]


def _chat_response_from_row(chapter_id: int, row: StudentTutorChat | None) -> TutorChatResponse:
    if row is None:
        return TutorChatResponse(chapter_id=chapter_id, subject_name="", messages=[])
    doc = parse_chat_storage(row.messages)
    threads_out = [
        TutorChatThreadOut(
            id=t["id"],
            title=t.get("title") or "Chat",
            messages=_messages_to_out(t.get("messages") or []),
            updated_at=t.get("updated_at"),
        )
        for t in doc["threads"]
    ]
    active_id = doc.get("active_thread_id") or None
    active_msgs: list[TutorChatMessageOut] = []
    for t in threads_out:
        if t.id == active_id:
            active_msgs = t.messages
            break
    if not active_msgs and threads_out:
        active_msgs = threads_out[-1].messages
        active_id = threads_out[-1].id
    return TutorChatResponse(
        chapter_id=chapter_id,
        subject_name=row.subject_name or "",
        messages=active_msgs,
        threads=threads_out,
        active_thread_id=active_id,
        updated_at=row.updated_at,
    )


def get_tutor_chat(db: Session, user: User, chapter_id: int) -> TutorChatResponse:
    row = db.scalar(
        select(StudentTutorChat).where(
            StudentTutorChat.user_id == user.id,
            StudentTutorChat.chapter_id == chapter_id,
        )
    )
    return _chat_response_from_row(chapter_id, row)


def put_tutor_chat(
    db: Session, user: User, chapter_id: int, payload: TutorChatPutRequest
) -> TutorChatResponse:
    tb = db.get(TextbookUpload, chapter_id)
    if tb is None:
        raise AuthException("Chapter not found.", status.HTTP_404_NOT_FOUND)

    cleaned = sanitize_chat_messages([m.model_dump() for m in payload.messages])
    row = db.scalar(
        select(StudentTutorChat).where(
            StudentTutorChat.user_id == user.id,
            StudentTutorChat.chapter_id == chapter_id,
        )
    )
    now = datetime.now(UTC)
    now_iso = now.isoformat()
    subject = (payload.subject_name or tb.subject_name or "").strip()[:120]
    doc = parse_chat_storage(row.messages if row is not None else [])

    if payload.new_thread:
        if doc["threads"] and doc.get("active_thread_id"):
            for t in doc["threads"]:
                if t["id"] == doc["active_thread_id"] and t.get("messages"):
                    t["title"] = thread_title(t["messages"], fallback=t.get("title") or "Chat")
                    t["updated_at"] = now_iso
                    break
        new_id = new_thread_id()
        doc["threads"].append(
            {
                "id": new_id,
                "title": "New Chat",
                "messages": cleaned,
                "updated_at": now_iso,
            }
        )
        doc["active_thread_id"] = new_id
    else:
        thread_id = (payload.thread_id or payload.active_thread_id or doc.get("active_thread_id") or "").strip()
        if not thread_id:
            thread_id = new_thread_id()
            doc["threads"].append(
                {"id": thread_id, "title": "Chat", "messages": [], "updated_at": now_iso}
            )
            doc["active_thread_id"] = thread_id

        found = False
        for t in doc["threads"]:
            if t["id"] == thread_id:
                if cleaned or payload.messages == []:
                    t["messages"] = cleaned
                t["title"] = thread_title(t["messages"], fallback=t.get("title") or "Chat")
                t["updated_at"] = now_iso
                found = True
                break
        if not found:
            doc["threads"].append(
                {
                    "id": thread_id,
                    "title": thread_title(cleaned),
                    "messages": cleaned,
                    "updated_at": now_iso,
                }
            )

        if payload.active_thread_id:
            doc["active_thread_id"] = payload.active_thread_id
        elif not doc.get("active_thread_id"):
            doc["active_thread_id"] = thread_id
        else:
            doc["active_thread_id"] = thread_id

    active = doc.get("active_thread_id") or ""
    cap_threads(doc, active=active)

    storage = serialize_chat_storage(doc)
    if row is None:
        row = StudentTutorChat(
            user_id=user.id,
            chapter_id=chapter_id,
            subject_name=subject,
            messages=storage,
            updated_at=now,
        )
        db.add(row)
    else:
        row.subject_name = subject or row.subject_name
        row.messages = storage
        row.updated_at = now
        from sqlalchemy.orm.attributes import flag_modified

        flag_modified(row, "messages")
    db.flush()
    return _chat_response_from_row(chapter_id, row)


def _build_recommended_topics(
    subjects_out: list[LearningSubjectOut],
    progress_by_chapter: dict[int, StudentChapterProgress],
    *,
    limit: int = 4,
) -> list[RecommendedTopicOut]:
    """Suggest topics/chapters from progress — remaining headings, then next chapters."""
    from app.modules.student_learning.topic_progress import list_chapter_topics

    recs: list[RecommendedTopicOut] = []
    seen: set[str] = set()

    def _add(
        *,
        title: str,
        reason: str,
        subject: LearningSubjectOut,
        chapter: LearningChapterOut,
    ) -> bool:
        key = title.strip().lower()
        if not key or key in seen:
            return False
        seen.add(key)
        recs.append(
            RecommendedTopicOut(
                title=title.strip(),
                reason=reason,
                subject_id=subject.id,
                subject_name=subject.subject_name,
                board=subject.board,
                class_level=subject.class_level,
                chapter_id=chapter.id,
                chapter_name=chapter.chapter,
            )
        )
        return len(recs) >= limit

    # 1) Remaining topics inside in-progress chapters (real headings when available)
    for subject in subjects_out:
        for chapter in subject.chapters:
            if chapter.status != "in_progress":
                continue
            row = progress_by_chapter.get(chapter.id)
            covered = set(row.covered_topics or []) if row else set()
            try:
                topics = list_chapter_topics(
                    chapter_id=chapter.id,
                    board=str(subject.board),
                    class_level=str(subject.class_level),
                    subject_name=subject.subject_name,
                    chapter_name=chapter.chapter or "",
                )
            except Exception:
                topics = []
            remaining = [t for t in topics if t.get("key") not in covered]
            for topic in remaining[:2]:
                title = str(topic.get("title") or "").strip()
                if not title:
                    continue
                if _add(
                    title=title,
                    reason=f"Next in {chapter.chapter or subject.subject_name}",
                    subject=subject,
                    chapter=chapter,
                ):
                    return recs
            # Fallback: the chapter itself
            chapter_title = (chapter.chapter or chapter.file_name or "").strip()
            if chapter_title and _add(
                title=chapter_title,
                reason="Continue where you left off",
                subject=subject,
                chapter=chapter,
            ):
                return recs

    # 2) Next not-started chapters
    for subject in subjects_out:
        for chapter in subject.chapters:
            if chapter.status != "not_started":
                continue
            chapter_title = (chapter.chapter or chapter.file_name or "").strip()
            if not chapter_title:
                continue
            if _add(
                title=chapter_title,
                reason=f"Start next in {subject.subject_name}",
                subject=subject,
                chapter=chapter,
            ):
                return recs

    return recs


def get_overview(db: Session, user: User) -> LearningOverviewResponse:
    scope_key = ensure_scope_or_reset(db, user)
    enrolled, scope_key = list_enrolled_subjects(db, user)

    progress_rows = list(
        db.scalars(select(StudentChapterProgress).where(StudentChapterProgress.user_id == user.id))
    )
    progress_by_chapter = {p.chapter_id: p for p in progress_rows}

    subjects_out: list[LearningSubjectOut] = []
    for subj in enrolled:
        chapters_out: list[LearningChapterOut] = []
        completed = 0
        chapter_pct_sum = 0
        for ch in subj.chapters:
            p = progress_by_chapter.get(ch.id)
            if p is None:
                st: str = "not_started"
                ch_pct = 0
                topics_total = 0
                topics_covered = 0
                remaining: list[str] = []
            elif p.status == "completed":
                st = "completed"
                completed += 1
                ch_pct = 100
                topics_total = int(p.topics_total or 0)
                topics_covered = topics_total or len(p.covered_topics or [])
                remaining = []
            else:
                st = "in_progress"
                ch_pct = int(p.progress_pct or 0)
                topics_total = int(p.topics_total or 0)
                covered_keys = list(p.covered_topics or [])
                topics_covered = len(covered_keys)
                remaining = []  # titles filled lazily only when needed; keep overview light
                if topics_total and topics_covered and ch_pct == 0:
                    ch_pct = min(99, int(round(100 * topics_covered / topics_total)))
            chapter_pct_sum += ch_pct
            chapters_out.append(
                LearningChapterOut(
                    id=ch.id,
                    chapter=ch.chapter,
                    file_name=ch.file_name,
                    status=st,  # type: ignore[arg-type]
                    progress=ch_pct,
                    topics_total=topics_total,
                    topics_covered=topics_covered,
                    topics_remaining=remaining,
                )
            )
        total = len(chapters_out)
        # Subject % = average of chapter topic-coverage percentages
        progress = int(round(chapter_pct_sum / total)) if total else 0
        if total == 0 or progress == 0:
            subj_status = "not_started"
        elif completed >= total and total > 0:
            subj_status = "completed"
        else:
            subj_status = "in_progress"
        subjects_out.append(
            LearningSubjectOut(
                id=subj.id,
                board=subj.board,
                class_level=subj.class_level,
                subject_name=subj.subject_name,
                chapters=chapters_out,
                completed_chapters=completed,
                total_chapters=total,
                progress=progress,
                status=subj_status,  # type: ignore[arg-type]
            )
        )

    lessons_completed = sum(1 for p in progress_rows if p.status == "completed")
    total_study = int(
        db.scalar(
            select(func.coalesce(func.sum(StudentStudySession.duration_seconds), 0)).where(
                StudentStudySession.user_id == user.id
            )
        )
        or 0
    )
    streak = db.scalar(select(StudentLearningStreak).where(StudentLearningStreak.user_id == user.id))
    current_streak = _apply_streak_decay(streak)

    # Continue + recent from progress ordered by last_accessed
    chapter_index: dict[int, tuple[LearningSubjectOut, LearningChapterOut]] = {}
    for s in subjects_out:
        for c in s.chapters:
            chapter_index[c.id] = (s, c)

    accessed = sorted(
        [p for p in progress_rows if p.chapter_id in chapter_index and p.last_accessed_at is not None],
        key=lambda p: p.last_accessed_at,
        reverse=True,
    )

    continue_learning: ContinueLearningOut | None = None
    if accessed:
        p0 = accessed[0]
        s0, c0 = chapter_index[p0.chapter_id]
        continue_learning = ContinueLearningOut(
            subject_id=s0.id,
            subject_name=s0.subject_name,
            board=s0.board,
            class_level=s0.class_level,
            chapter_id=c0.id,
            chapter_name=c0.chapter,
            file_name=c0.file_name,
            status=c0.status,
            progress=c0.progress if c0.progress else s0.progress,
            last_accessed_at=p0.last_accessed_at,
        )

    # Recent lessons for My Learning (scrollable); dashboard takes first 10.
    recent: list[RecentLessonOut] = []
    seen_chapters: set[int] = set()
    for p in accessed:
        if p.chapter_id in seen_chapters:
            continue
        seen_chapters.add(p.chapter_id)
        s, c = chapter_index[p.chapter_id]
        recent.append(
            RecentLessonOut(
                subject_id=s.id,
                subject_name=s.subject_name,
                board=s.board,
                class_level=s.class_level,
                chapter_id=c.id,
                chapter_name=c.chapter,
                file_name=c.file_name,
                status=c.status,
                last_accessed_at=p.last_accessed_at,
            )
        )
        if len(recent) >= 30:
            break

    recommended = _build_recommended_topics(subjects_out, progress_by_chapter)

    return LearningOverviewResponse(
        stats=LearningStatsOut(
            enrolled_subjects=len(subjects_out),
            lessons_completed=lessons_completed,
            total_study_seconds=total_study,
            current_streak=current_streak,
        ),
        subjects=subjects_out,
        continue_learning=continue_learning,
        recent_lessons=recent,
        recommended_topics=recommended,
        scope_key=scope_key,
    )
