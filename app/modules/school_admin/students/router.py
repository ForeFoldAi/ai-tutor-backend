from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.auth.schemas import MessageResponse
from app.modules.school_admin.students.schemas import (
    StudentActionResponse,
    StudentAssignTeacherRequest,
    StudentBulkCreateRequest,
    StudentBulkCreateResponse,
    StudentCreateItem,
    StudentListResponse,
    StudentMoveClassRequest,
    StudentOptionsResponse,
    StudentResponse,
    StudentUpdateRequest,
)
from app.modules.school_admin.students.service import (
    assign_teacher_to_students,
    bulk_create_students,
    create_student,
    delete_student,
    export_students_csv,
    get_student,
    list_students,
    move_students_to_class,
    student_options,
    update_student,
)
from app.modules.users.models import User

router = APIRouter(prefix="/auth/admin/students", tags=["students"])

SchoolAdminUser = Annotated[User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))]
SchoolStaffUser = Annotated[
    User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN, Role.TUTOR))
]


@router.get("/options", response_model=StudentOptionsResponse)
def students_options_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
):
    return student_options(db, current_user)


@router.get("", response_model=StudentListResponse)
def students_list_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
    q: Annotated[str | None, Query(description="Search name, email, phone")] = None,
    grade: Annotated[str | None, Query()] = None,
    section: Annotated[str | None, Query()] = None,
    curriculum: Annotated[str | None, Query()] = None,
    learning_type: Annotated[str | None, Query()] = None,
    status: Annotated[str | None, Query(description="active | inactive")] = None,
    school_id: Annotated[int | None, Query()] = None,
    limit: Annotated[int | None, Query(ge=1, le=1000)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    opts = student_options(db, current_user)
    return list_students(
        db,
        current_user,
        school_id=school_id,
        q=q,
        grade=grade,
        section=section,
        curriculum=curriculum,
        learning_type=learning_type,
        status_filter=status,
        limit=limit or opts.default_limit,
        offset=offset,
    )


@router.get("/export")
def students_export_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
    q: Annotated[str | None, Query()] = None,
    grade: Annotated[str | None, Query()] = None,
    section: Annotated[str | None, Query()] = None,
    curriculum: Annotated[str | None, Query()] = None,
    learning_type: Annotated[str | None, Query()] = None,
    status: Annotated[str | None, Query()] = None,
    school_id: Annotated[int | None, Query()] = None,
):
    csv_text = export_students_csv(
        db,
        current_user,
        school_id=school_id,
        q=q,
        grade=grade,
        section=section,
        curriculum=curriculum,
        learning_type=learning_type,
        status_filter=status,
    )
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="students-export.csv"'},
    )


@router.get("/{student_id}", response_model=StudentResponse)
def students_get_route(
    student_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    return get_student(db, current_user, student_id)


@router.post("", response_model=StudentResponse, status_code=201)
def students_create_route(
    payload: StudentCreateItem,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    result = create_student(db, current_user, payload)
    db.commit()
    return result


@router.post("/bulk", response_model=StudentBulkCreateResponse, status_code=201)
def students_bulk_route(
    payload: StudentBulkCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
):
    result = bulk_create_students(db, current_user, payload.students)
    db.commit()
    return result


@router.post("/assign-teacher", response_model=StudentActionResponse)
def students_assign_teacher_route(
    payload: StudentAssignTeacherRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
):
    result = assign_teacher_to_students(db, current_user, payload)
    db.commit()
    return result


@router.post("/move-class", response_model=StudentActionResponse)
def students_move_class_route(
    payload: StudentMoveClassRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
):
    result = move_students_to_class(db, current_user, payload)
    db.commit()
    return result


@router.patch("/{student_id}", response_model=StudentResponse)
def students_update_route(
    student_id: int,
    payload: StudentUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    result = update_student(db, current_user, student_id, payload)
    db.commit()
    return result


@router.delete("/{student_id}", response_model=MessageResponse)
def students_delete_route(
    student_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    delete_student(db, current_user, student_id)
    db.commit()
    return MessageResponse(message="Student removed.")
