from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.auth.schemas import MessageResponse
from app.modules.school_admin.teachers.schemas import (
    TeacherAssignClassesRequest,
    TeacherAssignResponse,
    TeacherAssignSubjectsRequest,
    TeacherBulkCreateRequest,
    TeacherBulkCreateResponse,
    TeacherCreateItem,
    TeacherDetailResponse,
    TeacherListResponse,
    TeacherOptionsResponse,
    TeacherResponse,
    TeacherUnassignRequest,
    TeacherUpdateRequest,
)
from app.modules.school_admin.teachers.service import (
    assign_classes_to_teachers,
    assign_subjects_to_teachers,
    bulk_create_teachers,
    create_teacher,
    delete_teacher,
    export_teachers_csv,
    get_teacher,
    list_teachers,
    teacher_options,
    unassign_teacher_items,
    update_teacher,
)
from app.modules.users.models import User

router = APIRouter(prefix="/auth/admin/teachers", tags=["teachers"])

SchoolAdminUser = Annotated[User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN))]
SchoolStaffUser = Annotated[
    User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN, Role.TUTOR))
]


@router.get("/options", response_model=TeacherOptionsResponse)
def teachers_options_route(_current_user: SchoolStaffUser):
    return teacher_options()


@router.get("", response_model=TeacherListResponse)
def teachers_list_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
    q: Annotated[str | None, Query(description="Search name, email, phone")] = None,
    subject: Annotated[str | None, Query(description="Filter by subject/board")] = None,
    curriculum: Annotated[str | None, Query(description="Filter by class curriculum")] = None,
    status: Annotated[str | None, Query(description="active | inactive")] = None,
    school_id: Annotated[int | None, Query(description="Master admin: school filter")] = None,
    limit: Annotated[int | None, Query(ge=1, le=100)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return list_teachers(
        db,
        current_user,
        school_id=school_id,
        q=q,
        subject=subject,
        curriculum=curriculum,
        status_filter=status,
        limit=limit or teacher_options().default_limit,
        offset=offset,
    )


@router.get("/export")
def teachers_export_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
    q: Annotated[str | None, Query()] = None,
    subject: Annotated[str | None, Query()] = None,
    status: Annotated[str | None, Query()] = None,
    school_id: Annotated[int | None, Query()] = None,
):
    csv_text = export_teachers_csv(
        db,
        current_user,
        school_id=school_id,
        q=q,
        subject=subject,
        status_filter=status,
    )
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="teachers-export.csv"'},
    )


@router.get("/{teacher_id}", response_model=TeacherDetailResponse)
def teachers_get_route(
    teacher_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    return get_teacher(db, current_user, teacher_id)


@router.post("/{teacher_id}/unassign", response_model=TeacherDetailResponse)
def teachers_unassign_route(
    teacher_id: int,
    payload: TeacherUnassignRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    result = unassign_teacher_items(db, current_user, teacher_id, payload)
    db.commit()
    return result


@router.post("", response_model=TeacherResponse, status_code=201)
def teachers_create_route(
    payload: TeacherCreateItem,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
    school_id: Annotated[int | None, Query()] = None,
):
    teacher = create_teacher(db, current_user, payload, school_id=school_id)
    db.commit()
    return teacher


@router.post("/bulk", response_model=TeacherBulkCreateResponse, status_code=201)
def teachers_bulk_create_route(
    payload: TeacherBulkCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
    school_id: Annotated[int | None, Query()] = None,
):
    result = bulk_create_teachers(db, current_user, payload.teachers, school_id=school_id)
    db.commit()
    return result


@router.post("/assign-classes", response_model=TeacherAssignResponse)
def teachers_assign_classes_route(
    payload: TeacherAssignClassesRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
    school_id: Annotated[int | None, Query()] = None,
):
    result = assign_classes_to_teachers(
        db,
        current_user,
        payload.teacher_ids,
        payload.class_ids,
        school_id=school_id,
    )
    db.commit()
    return result


@router.post("/assign-subjects", response_model=TeacherAssignResponse)
def teachers_assign_subjects_route(
    payload: TeacherAssignSubjectsRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
    school_id: Annotated[int | None, Query()] = None,
):
    result = assign_subjects_to_teachers(
        db,
        current_user,
        payload.teacher_ids,
        payload.subject_ids,
        school_id=school_id,
    )
    db.commit()
    return result


@router.patch("/{teacher_id}", response_model=TeacherDetailResponse)
def teachers_update_route(
    teacher_id: int,
    payload: TeacherUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    teacher = update_teacher(db, current_user, teacher_id, payload)
    db.commit()
    return teacher


@router.delete("/{teacher_id}", response_model=MessageResponse)
def teachers_delete_route(
    teacher_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    delete_teacher(db, current_user, teacher_id)
    db.commit()
    return MessageResponse(message="Teacher removed.")
