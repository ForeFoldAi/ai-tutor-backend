from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.modules.auth.exceptions import AuthException
from app.modules.auth.service import list_tutor_assigned_students
from app.modules.school_admin.students.service import _class_fields
from app.modules.notifications.service import notify_users
from app.modules.teacher.assignments.constants import (
    AssignableArtifactType,
    AssignmentStatus,
    SubmissionStatus,
)
from app.modules.teacher.assignments.content import (
    grade_submission,
    homework_items_need_repair,
    normalize_artifact,
    worksheet_items_need_repair,
)
from app.modules.teacher.assignments.models import AssignmentSubmission, TeacherAssignment
from app.modules.teacher.assignments.schemas import (
    AssignmentCounts,
    AssignmentResultsResponse,
    CreateAssignmentRequest,
    CreateAssignmentsResponse,
    PatchAssignmentRequest,
    PreviewMatchResponse,
    StudentAssignmentDetail,
    StudentAssignmentListItem,
    StudentAssignmentListResponse,
    StudentResultRow,
    SubmissionResultItem,
    TutorAssignmentListItem,
    TutorAssignmentListResponse,
)
from app.modules.teacher.lesson_planner.constants import ArtifactType
from app.modules.teacher.lesson_planner.models import LessonArtifact, LessonPlan
from app.modules.teacher.students.service import _subjects_for_students
from app.modules.users.models import User

ASSIGNABLE = {
    AssignableArtifactType.WORKSHEET.value,
    AssignableArtifactType.QUIZ.value,
    AssignableArtifactType.HOMEWORK.value,
}


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def _grade_key(grade: str) -> str:
    g = grade.strip()
    if g.upper().startswith("CLASS_"):
        g = g[6:]
    return g.strip().lower().lstrip("0") or "0"


def match_students_for_scope(
    db: Session,
    tutor: User,
    *,
    grade: str,
    section: str,
    curriculum: str = "",
    subject: str | None = None,
) -> list[User]:
    roster = list_tutor_assigned_students(db, tutor)
    want_grade = _grade_key(grade)
    want_section = section.strip().upper()
    want_curr = _norm(curriculum)
    want_subject = _norm(subject) if subject else ""

    subjects_map = _subjects_for_students(db, tutor, roster) if want_subject else {}

    matched: list[User] = []
    for student in roster:
        s_grade, s_section, s_curr = _class_fields(student)
        if not s_grade or not s_section:
            continue
        if _grade_key(s_grade) != want_grade:
            continue
        if s_section.strip().upper() != want_section:
            continue
        if want_curr and s_curr and _norm(s_curr) != want_curr:
            continue
        if want_subject:
            names = {_norm(n) for n in subjects_map.get(student.id, [])}
            if names and want_subject not in names:
                continue
        matched.append(student)
    return matched


def preview_match(
    db: Session,
    tutor: User,
    *,
    grade: str,
    section: str,
    curriculum: str = "",
    subject: str | None = None,
) -> PreviewMatchResponse:
    matched = match_students_for_scope(
        db, tutor, grade=grade, section=section, curriculum=curriculum, subject=subject
    )
    return PreviewMatchResponse(
        matched_count=len(matched),
        grade=grade,
        section=section,
        curriculum=curriculum or "",
        subject=subject,
    )


def _latest_artifact(db: Session, plan: LessonPlan, artifact_type: str) -> LessonArtifact | None:
    rows = [
        a
        for a in (plan.artifacts or [])
        if a.artifact_type.value == artifact_type and a.content
    ]
    if not rows:
        # reload
        rows = list(
            db.scalars(
                select(LessonArtifact)
                .where(
                    LessonArtifact.lesson_plan_id == plan.id,
                    LessonArtifact.artifact_type == ArtifactType(artifact_type),
                )
                .order_by(LessonArtifact.version_number.desc())
            )
        )
    if not rows:
        return None
    return max(rows, key=lambda a: a.version_number)


def _repair_answer_key(db: Session, assignment: TeacherAssignment) -> bool:
    """Re-parse answer key from lesson artifact when missing (old assignments)."""
    if (assignment.answer_key or {}).get("answers"):
        return False
    if not assignment.lesson_plan_id:
        return False
    plan = db.get(LessonPlan, assignment.lesson_plan_id)
    if not plan:
        return False
    artifact = _latest_artifact(db, plan, assignment.artifact_type)
    if not artifact or not artifact.content:
        return False
    _, answer_key = normalize_artifact(assignment.artifact_type, artifact.content)
    if not (answer_key.get("answers") or {}):
        return False
    assignment.answer_key = answer_key
    db.add(assignment)
    db.flush()
    return True


def _repair_homework_content(db: Session, assignment: TeacherAssignment) -> bool:
    """Re-parse homework/worksheet when old blob or line-per-task junk is stored."""
    atype = assignment.artifact_type
    items = (assignment.student_content or {}).get("items") or []
    if atype == "homework":
        if not homework_items_need_repair(items):
            return False
    elif atype == "worksheet":
        if not worksheet_items_need_repair(items):
            return False
    else:
        return False
    if not assignment.lesson_plan_id:
        return False
    plan = db.get(LessonPlan, assignment.lesson_plan_id)
    if not plan:
        return False
    artifact = _latest_artifact(db, plan, atype)
    if not artifact or not artifact.content:
        return False
    student_content, answer_key = normalize_artifact(atype, artifact.content)
    new_items = student_content.get("items") or []
    if not new_items:
        return False
    if atype == "homework" and homework_items_need_repair(new_items):
        return False
    if atype == "worksheet" and worksheet_items_need_repair(new_items):
        return False
    assignment.student_content = student_content
    if (answer_key.get("answers") or {}) and not (assignment.answer_key or {}).get("answers"):
        assignment.answer_key = answer_key
    assignment.updated_at = datetime.now(UTC)
    db.add(assignment)
    db.flush()
    return True


def _regrade_if_needed(db: Session, assignment: TeacherAssignment, sub: AssignmentSubmission) -> bool:
    """Fill score/correct answers when answer key was missing. Does not commit."""
    if assignment.artifact_type in ("worksheet", "homework"):
        return False
    if sub.status not in (SubmissionStatus.SUBMITTED.value, SubmissionStatus.GRADED.value):
        return False
    if sub.score is not None:
        return False
    if not sub.answers:
        return False
    if not (assignment.answer_key or {}).get("answers"):
        return False
    score, max_score, result, status_val = grade_submission(
        assignment.student_content or {},
        assignment.answer_key or {},
        sub.answers or {},
    )
    if score is None:
        return False
    sub.score = score
    sub.max_score = max_score
    sub.result = result
    sub.status = status_val
    sub.updated_at = datetime.now(UTC)
    db.add(sub)
    return True


def _maybe_regrade_submission(db: Session, assignment: TeacherAssignment, sub: AssignmentSubmission) -> None:
    """Fill score/correct answers for submissions that were saved without an answer key."""
    repaired = _repair_answer_key(db, assignment)
    repaired = _repair_homework_content(db, assignment) or repaired
    if _regrade_if_needed(db, assignment, sub) or repaired:
        db.commit()
        db.refresh(sub)
        db.refresh(assignment)


def _counts_for(assignment: TeacherAssignment, now: datetime | None = None) -> AssignmentCounts:
    now = now or datetime.now(UTC)
    c = AssignmentCounts()
    for sub in assignment.submissions or []:
        st = sub.status
        effective = st
        if st in (SubmissionStatus.PENDING.value, SubmissionStatus.IN_PROGRESS.value):
            deadline = assignment.deadline
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=UTC)
            if deadline < now:
                effective = "overdue"
                c.overdue += 1
        if effective == SubmissionStatus.PENDING.value:
            c.pending += 1
        elif effective == SubmissionStatus.IN_PROGRESS.value:
            c.in_progress += 1
        elif effective == SubmissionStatus.SUBMITTED.value:
            c.submitted += 1
        elif effective == SubmissionStatus.GRADED.value:
            c.graded += 1
        c.total += 1
    return c


def _to_tutor_item(assignment: TeacherAssignment) -> TutorAssignmentListItem:
    return TutorAssignmentListItem(
        id=assignment.id,
        lesson_plan_id=assignment.lesson_plan_id,
        title=assignment.title,
        artifact_type=assignment.artifact_type,
        subject=assignment.subject,
        grade=assignment.grade,
        section=assignment.section,
        curriculum=assignment.curriculum,
        chapter_name=assignment.chapter_name,
        deadline=assignment.deadline,
        status=assignment.status,
        counts=_counts_for(assignment),
        created_at=assignment.created_at,
    )


def create_assignments(
    db: Session,
    tutor: User,
    payload: CreateAssignmentRequest,
) -> CreateAssignmentsResponse:
    plan = db.scalar(
        select(LessonPlan)
        .where(LessonPlan.id == payload.lesson_plan_id, LessonPlan.deleted_at.is_(None))
        .options(selectinload(LessonPlan.artifacts))
    )
    if not plan or plan.user_id != tutor.id:
        raise AuthException("Lesson plan not found.", status.HTTP_404_NOT_FOUND)

    subject = (payload.subject or plan.subject or "").strip()
    grade = payload.grade.strip()
    section = payload.section.strip().upper()
    curriculum = (payload.curriculum or plan.board or "").strip()

    students = match_students_for_scope(
        db,
        tutor,
        grade=grade,
        section=section,
        curriculum=curriculum,
        subject=subject,
    )
    if not students:
        raise AuthException(
            "No students match this class, section, and curriculum.",
            status.HTTP_400_BAD_REQUEST,
        )

    deadline = payload.deadline
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    if deadline <= datetime.now(UTC):
        raise AuthException("Deadline must be in the future.", status.HTTP_400_BAD_REQUEST)

    created: list[TeacherAssignment] = []
    now = datetime.now(UTC)

    for art in payload.artifact_types:
        atype = art.value
        if atype not in ASSIGNABLE:
            continue
        existing = db.scalar(
            select(TeacherAssignment).where(
                TeacherAssignment.teacher_id == tutor.id,
                TeacherAssignment.lesson_plan_id == plan.id,
                TeacherAssignment.artifact_type == atype,
                TeacherAssignment.status == AssignmentStatus.ACTIVE.value,
            )
        )
        if existing:
            raise AuthException(
                f"{atype.replace('_', ' ').title()} is already assigned for this lesson.",
                status.HTTP_409_CONFLICT,
            )
        artifact = _latest_artifact(db, plan, atype)
        if not artifact or not artifact.content:
            raise AuthException(
                f"No saved {atype} content on this lesson plan.",
                status.HTTP_400_BAD_REQUEST,
            )
        student_content, answer_key = normalize_artifact(atype, artifact.content)
        if not (student_content.get("items") or []):
            raise AuthException(
                f"Could not build assignable {atype} questions.",
                status.HTTP_400_BAD_REQUEST,
            )

        title = f"{plan.title} — {atype.replace('_', ' ').title()}"
        assignment = TeacherAssignment(
            teacher_id=tutor.id,
            lesson_plan_id=plan.id,
            artifact_type=atype,
            title=title,
            grade=grade,
            section=section,
            curriculum=curriculum,
            subject=subject,
            chapter_name=plan.chapter_name or "",
            deadline=deadline,
            student_content=student_content,
            answer_key=answer_key,
            status=AssignmentStatus.ACTIVE.value,
            created_at=now,
            updated_at=now,
        )
        db.add(assignment)
        db.flush()
        for student in students:
            db.add(
                AssignmentSubmission(
                    assignment_id=assignment.id,
                    student_id=student.id,
                    status=SubmissionStatus.PENDING.value,
                    created_at=now,
                    updated_at=now,
                )
            )
        created.append(assignment)

    student_ids = [s.id for s in students]
    for assignment in created:
        kind = assignment.artifact_type.replace("_", " ")
        notify_users(
            db,
            recipient_ids=student_ids,
            type=f"assignment_{assignment.artifact_type}",
            title=f"New {kind}",
            body=assignment.title,
            actor_id=tutor.id,
            school_id=tutor.school_id,
            link="/assignments",
        )

    db.commit()
    for a in created:
        db.refresh(a)
        a.submissions = list(
            db.scalars(
                select(AssignmentSubmission).where(AssignmentSubmission.assignment_id == a.id)
            )
        )

    items = [_to_tutor_item(a) for a in created]
    return CreateAssignmentsResponse(
        created=items,
        message=f"Assigned to {len(students)} student(s).",
    )


def list_tutor_assignments(db: Session, tutor: User) -> TutorAssignmentListResponse:
    rows = list(
        db.scalars(
            select(TeacherAssignment)
            .where(
                TeacherAssignment.teacher_id == tutor.id,
                TeacherAssignment.status == AssignmentStatus.ACTIVE.value,
            )
            .options(selectinload(TeacherAssignment.submissions))
            .order_by(TeacherAssignment.created_at.desc())
        )
    )
    items = [_to_tutor_item(a) for a in rows]
    return TutorAssignmentListResponse(items=items, total=len(items))


def get_assignment_results(db: Session, tutor: User, assignment_id: int) -> AssignmentResultsResponse:
    assignment = db.scalar(
        select(TeacherAssignment)
        .where(TeacherAssignment.id == assignment_id, TeacherAssignment.teacher_id == tutor.id)
        .options(selectinload(TeacherAssignment.submissions))
    )
    if not assignment:
        raise AuthException("Assignment not found.", status.HTTP_404_NOT_FOUND)

    _repair_answer_key(db, assignment)
    _repair_homework_content(db, assignment)
    changed = False
    for sub in assignment.submissions:
        if _regrade_if_needed(db, assignment, sub):
            changed = True
    if changed:
        db.commit()
        db.refresh(assignment)
        assignment.submissions = list(
            db.scalars(
                select(AssignmentSubmission).where(AssignmentSubmission.assignment_id == assignment.id)
            )
        )

    student_ids = [s.student_id for s in assignment.submissions]
    names: dict[int, str] = {}
    if student_ids:
        for u in db.scalars(select(User).where(User.id.in_(student_ids))):
            names[u.id] = u.full_name

    now = datetime.now(UTC)
    students: list[StudentResultRow] = []
    for sub in assignment.submissions:
        st = sub.status
        deadline = assignment.deadline
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        if st in (SubmissionStatus.PENDING.value, SubmissionStatus.IN_PROGRESS.value) and deadline < now:
            st = "overdue"
        result = sub.result if isinstance(sub.result, dict) else {}
        raw_items = result.get("items") if isinstance(result.get("items"), list) else []
        result_items = [
            SubmissionResultItem(
                id=str(item.get("id") or ""),
                question=str(item.get("question") or ""),
                student_answer=str(item.get("student_answer") or ""),
                correct_answer=item.get("correct_answer"),
                is_correct=item.get("is_correct"),
            )
            for item in raw_items
            if isinstance(item, dict)
        ]
        students.append(
            StudentResultRow(
                submission_id=sub.id,
                student_id=sub.student_id,
                student_name=names.get(sub.student_id, "Student"),
                status=st,
                score=sub.score,
                max_score=sub.max_score,
                started_at=sub.started_at,
                submitted_at=sub.submitted_at,
                result_items=result_items,
                auto_scored=result.get("auto_scored"),
            )
        )
    students.sort(key=lambda r: r.student_name.lower())
    return AssignmentResultsResponse(assignment=_to_tutor_item(assignment), students=students)


def patch_assignment_deadline(
    db: Session,
    tutor: User,
    assignment_id: int,
    payload: PatchAssignmentRequest,
) -> TutorAssignmentListItem:
    assignment = db.scalar(
        select(TeacherAssignment)
        .where(TeacherAssignment.id == assignment_id, TeacherAssignment.teacher_id == tutor.id)
        .options(selectinload(TeacherAssignment.submissions))
    )
    if not assignment:
        raise AuthException("Assignment not found.", status.HTTP_404_NOT_FOUND)
    deadline = payload.deadline
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    assignment.deadline = deadline
    assignment.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(assignment)
    return _to_tutor_item(assignment)


def _effective_student_status(assignment: TeacherAssignment, sub: AssignmentSubmission) -> str:
    st = sub.status
    if st in (SubmissionStatus.PENDING.value, SubmissionStatus.IN_PROGRESS.value):
        deadline = assignment.deadline
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)
        if deadline < datetime.now(UTC):
            return "overdue"
    # Worksheet / homework: graded → submitted (no scores for students).
    if (
        assignment.artifact_type in ("worksheet", "homework")
        and st == SubmissionStatus.GRADED.value
    ):
        return SubmissionStatus.SUBMITTED.value
    return st


def _student_facing_scores(
    assignment: TeacherAssignment, sub: AssignmentSubmission
) -> tuple[float | None, float | None]:
    if assignment.artifact_type in ("worksheet", "homework"):
        return None, None
    return sub.score, sub.max_score


def _student_facing_result(
    assignment: TeacherAssignment, result: dict | None
) -> dict | None:
    if not result or assignment.artifact_type not in ("worksheet", "homework"):
        return result
    items = []
    for row in result.get("items") or []:
        if not isinstance(row, dict):
            continue
        items.append({**row, "is_correct": None, "correct_answer": None})
    return {**result, "auto_scored": False, "items": items}


def list_student_assignments(db: Session, student: User) -> StudentAssignmentListResponse:
    rows = list(
        db.scalars(
            select(AssignmentSubmission)
            .where(AssignmentSubmission.student_id == student.id)
            .options(selectinload(AssignmentSubmission.assignment))
            .order_by(AssignmentSubmission.created_at.desc())
        )
    )
    teacher_ids = {r.assignment.teacher_id for r in rows if r.assignment}
    teachers: dict[int, str] = {}
    if teacher_ids:
        for u in db.scalars(select(User).where(User.id.in_(teacher_ids))):
            teachers[u.id] = u.full_name

    items: list[StudentAssignmentListItem] = []
    for sub in rows:
        a = sub.assignment
        if not a or a.status != AssignmentStatus.ACTIVE.value:
            continue
        repaired = _repair_answer_key(db, a)
        repaired = _repair_homework_content(db, a) or repaired
        if repaired or _regrade_if_needed(db, a, sub):
            db.commit()
            db.refresh(sub)
            db.refresh(a)
        items_content = (a.student_content or {}).get("items") or []
        score, max_score = _student_facing_scores(a, sub)
        items.append(
            StudentAssignmentListItem(
                id=a.id,
                submission_id=sub.id,
                title=a.title,
                artifact_type=a.artifact_type,
                subject=a.subject,
                teacher_name=teachers.get(a.teacher_id, "Teacher"),
                grade=a.grade,
                section=a.section,
                curriculum=a.curriculum,
                chapter_name=a.chapter_name,
                deadline=a.deadline,
                status=_effective_student_status(a, sub),
                question_count=len(items_content),
                score=score,
                max_score=max_score,
                submitted_at=sub.submitted_at,
            )
        )
    return StudentAssignmentListResponse(items=items, total=len(items))


def _get_student_submission(
    db: Session, student: User, assignment_id: int
) -> tuple[TeacherAssignment, AssignmentSubmission]:
    assignment = db.get(TeacherAssignment, assignment_id)
    if not assignment or assignment.status != AssignmentStatus.ACTIVE.value:
        raise AuthException("Assignment not found.", status.HTTP_404_NOT_FOUND)
    sub = db.scalar(
        select(AssignmentSubmission).where(
            AssignmentSubmission.assignment_id == assignment_id,
            AssignmentSubmission.student_id == student.id,
        )
    )
    if not sub:
        raise AuthException("Assignment not found.", status.HTTP_404_NOT_FOUND)
    return assignment, sub


def get_student_assignment_detail(
    db: Session, student: User, assignment_id: int
) -> StudentAssignmentDetail:
    assignment, sub = _get_student_submission(db, student, assignment_id)
    _maybe_regrade_submission(db, assignment, sub)
    teacher = db.get(User, assignment.teacher_id)
    done = sub.status in (SubmissionStatus.SUBMITTED.value, SubmissionStatus.GRADED.value)
    started = sub.status != SubmissionStatus.PENDING.value
    score, max_score = _student_facing_scores(assignment, sub) if done else (None, None)
    return StudentAssignmentDetail(
        id=assignment.id,
        submission_id=sub.id,
        title=assignment.title,
        artifact_type=assignment.artifact_type,
        subject=assignment.subject,
        teacher_name=teacher.full_name if teacher else "Teacher",
        deadline=assignment.deadline,
        status=_effective_student_status(assignment, sub),
        content=assignment.student_content if started else None,
        answers=sub.answers if done else None,
        result=_student_facing_result(assignment, sub.result if isinstance(sub.result, dict) else None)
        if done
        else None,
        score=score,
        max_score=max_score,
        submitted_at=sub.submitted_at,
    )


def start_assignment(db: Session, student: User, assignment_id: int) -> StudentAssignmentDetail:
    assignment, sub = _get_student_submission(db, student, assignment_id)
    repaired = _repair_homework_content(db, assignment)
    now = datetime.now(UTC)
    if sub.status == SubmissionStatus.PENDING.value:
        sub.status = SubmissionStatus.IN_PROGRESS.value
        sub.started_at = now
        sub.updated_at = now
        db.commit()
        db.refresh(sub)
        db.refresh(assignment)
    elif sub.status in (SubmissionStatus.SUBMITTED.value, SubmissionStatus.GRADED.value):
        raise AuthException("Assignment already submitted.", status.HTTP_400_BAD_REQUEST)
    elif repaired:
        db.commit()
        db.refresh(assignment)

    teacher = db.get(User, assignment.teacher_id)
    return StudentAssignmentDetail(
        id=assignment.id,
        submission_id=sub.id,
        title=assignment.title,
        artifact_type=assignment.artifact_type,
        subject=assignment.subject,
        teacher_name=teacher.full_name if teacher else "Teacher",
        deadline=assignment.deadline,
        status=_effective_student_status(assignment, sub),
        content=assignment.student_content,
        answers=None,
        result=None,
        score=None,
        max_score=None,
        submitted_at=None,
    )


def submit_assignment(
    db: Session,
    student: User,
    assignment_id: int,
    answers: dict[str, Any],
) -> StudentAssignmentDetail:
    assignment, sub = _get_student_submission(db, student, assignment_id)
    if sub.status in (SubmissionStatus.SUBMITTED.value, SubmissionStatus.GRADED.value):
        raise AuthException("Assignment already submitted.", status.HTTP_400_BAD_REQUEST)

    now = datetime.now(UTC)
    if sub.status == SubmissionStatus.PENDING.value:
        sub.started_at = now
    _repair_answer_key(db, assignment)
    _repair_homework_content(db, assignment)
    score, max_score, result, status_val = grade_submission(
        assignment.student_content or {},
        assignment.answer_key or {},
        answers or {},
    )
    # Worksheet / homework: submit only — no auto-grade or scores for students.
    if assignment.artifact_type in ("worksheet", "homework"):
        status_val = SubmissionStatus.SUBMITTED.value
        score, max_score = None, None
        if isinstance(result, dict):
            for row in result.get("items") or []:
                if isinstance(row, dict):
                    row["is_correct"] = None
                    row["correct_answer"] = None
            result = {**result, "auto_scored": False, "items": result.get("items") or []}

    sub.answers = answers or {}
    sub.score = score
    sub.max_score = max_score
    sub.result = result
    sub.status = status_val
    sub.submitted_at = now
    sub.updated_at = now
    db.commit()
    db.refresh(sub)

    from app.services.learning_intelligence.clients.lia_client import emit_assignment_submitted

    items = []
    if isinstance(sub.result, dict):
        items = sub.result.get("items") or []
    emit_assignment_submitted(
        student_user_id=student.id,
        subject_name=assignment.subject or "",
        score=score,
        max_score=max_score,
        assignment_title=assignment.title,
        result_items=items if isinstance(items, list) else [],
        chapter_name=assignment.chapter_name or "",
    )

    teacher = db.get(User, assignment.teacher_id)
    return StudentAssignmentDetail(
        id=assignment.id,
        submission_id=sub.id,
        title=assignment.title,
        artifact_type=assignment.artifact_type,
        subject=assignment.subject,
        teacher_name=teacher.full_name if teacher else "Teacher",
        deadline=assignment.deadline,
        status=sub.status,
        content=assignment.student_content,
        answers=sub.answers,
        result=sub.result,
        score=sub.score,
        max_score=sub.max_score,
        submitted_at=sub.submitted_at,
    )
