from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.teacher.students.schemas import (
    TutorStudentListResponse,
    TutorStudentOptionsResponse,
    TutorStudentProfileResponse,
)
from app.modules.teacher.students.service import (
    assigned_student_options,
    export_assigned_students_csv,
    get_student_profile,
    list_assigned_students,
)
from app.modules.users.models import User

router = APIRouter(prefix="/auth/tutor/students", tags=["tutor-students"])

TutorUser = Annotated[User, Depends(require_roles(Role.TUTOR))]


@router.get("/options", response_model=TutorStudentOptionsResponse)
def tutor_students_options(
    db: Annotated[Session, Depends(get_db)],
    current_user: TutorUser,
):
    return assigned_student_options(db, current_user)


@router.get("/export")
def tutor_students_export(
    db: Annotated[Session, Depends(get_db)],
    current_user: TutorUser,
    q: Annotated[str | None, Query()] = None,
    grade: Annotated[str | None, Query()] = None,
    subject: Annotated[str | None, Query()] = None,
    risk_level: Annotated[str | None, Query()] = None,
):
    csv_text = export_assigned_students_csv(
        db,
        current_user,
        q=q,
        grade=grade,
        subject=subject,
        risk_level=risk_level,
    )
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="tutor-students-export.csv"'},
    )


@router.get("", response_model=TutorStudentListResponse)
def tutor_students_list(
    db: Annotated[Session, Depends(get_db)],
    current_user: TutorUser,
    q: Annotated[str | None, Query(description="Search name or user id")] = None,
    grade: Annotated[str | None, Query()] = None,
    subject: Annotated[str | None, Query()] = None,
    risk_level: Annotated[str | None, Query()] = None,
    limit: Annotated[int | None, Query(ge=1, le=1000)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    opts = assigned_student_options(db, current_user)
    return list_assigned_students(
        db,
        current_user,
        q=q,
        grade=grade,
        subject=subject,
        risk_level=risk_level,
        limit=limit or opts.default_limit,
        offset=offset,
    )


@router.get("/{student_id}/profile", response_model=TutorStudentProfileResponse)
def tutor_student_profile(
    student_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: TutorUser,
):
    return get_student_profile(db, current_user, student_id)
