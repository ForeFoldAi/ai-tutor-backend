from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.auth.schemas import MessageResponse
from app.modules.school_admin.classes.schemas import (
    ClassBulkCreateRequest,
    ClassBulkCreateResponse,
    ClassListResponse,
    ClassOptionsResponse,
    ClassResponse,
    ClassSubjectMappingSaveRequest,
    ClassSubjectMappingsResponse,
    ClassUpdateRequest,
    SubjectBulkCreateRequest,
    SubjectBulkCreateResponse,
    SubjectListResponse,
    SubjectResponse,
    SubjectUpdateRequest,
)
from app.modules.school_admin.classes.service import (
    bulk_create_classes,
    bulk_create_subjects,
    class_options,
    delete_class,
    delete_subject,
    get_class_subject_mappings,
    list_classes,
    list_subjects,
    save_class_subject_mapping,
    update_class,
    update_subject,
)
from app.modules.users.models import User

router = APIRouter(prefix="/auth/admin/classes", tags=["classes"])

# Write/options/subjects/mappings: TUTOR allowed at router; service assert_can_manage_classes
# rejects school-tagged tutors. GET list stays SchoolStaffUser for move-class.
SchoolStaffUser = Annotated[
    User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN, Role.TUTOR))
]


@router.get("/options", response_model=ClassOptionsResponse)
def classes_options_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
    school_id: Annotated[int | None, Query()] = None,
):
    return class_options(db, current_user, school_id=school_id)


@router.get("/subjects", response_model=SubjectListResponse)
def subjects_list_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
    school_id: Annotated[int | None, Query()] = None,
    limit: Annotated[int | None, Query(ge=1)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return list_subjects(
        db,
        current_user,
        school_id=school_id,
        limit=limit,
        offset=offset,
    )


@router.post("/subjects/bulk", response_model=SubjectBulkCreateResponse, status_code=201)
def subjects_bulk_create_route(
    payload: SubjectBulkCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
    school_id: Annotated[int | None, Query()] = None,
):
    result = bulk_create_subjects(db, current_user, payload.subjects, school_id=school_id)
    db.commit()
    return result


@router.patch("/subjects/{subject_id}", response_model=SubjectResponse)
def subjects_update_route(
    subject_id: int,
    payload: SubjectUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
):
    result = update_subject(db, current_user, subject_id, payload)
    db.commit()
    return result


@router.delete("/subjects/{subject_id}", response_model=MessageResponse)
def subjects_delete_route(
    subject_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
):
    delete_subject(db, current_user, subject_id)
    db.commit()
    return MessageResponse(message="Subject removed.")


@router.get("/mappings", response_model=ClassSubjectMappingsResponse)
def class_mappings_list_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
    school_id: Annotated[int | None, Query()] = None,
):
    return get_class_subject_mappings(db, current_user, school_id=school_id)


@router.get("", response_model=ClassListResponse)
def classes_list_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
    school_id: Annotated[int | None, Query()] = None,
    q: Annotated[str | None, Query(description="Search grade, section, curriculum")] = None,
    limit: Annotated[int | None, Query(ge=1)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return list_classes(
        db,
        current_user,
        school_id=school_id,
        q=q,
        limit=limit,
        offset=offset,
    )


@router.post("/bulk", response_model=ClassBulkCreateResponse, status_code=201)
def classes_bulk_create_route(
    payload: ClassBulkCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
    school_id: Annotated[int | None, Query()] = None,
):
    result = bulk_create_classes(db, current_user, payload.classes, school_id=school_id)
    db.commit()
    return result


@router.put("/{class_id}/subjects", response_model=ClassSubjectMappingsResponse)
def class_subjects_save_route(
    class_id: int,
    payload: ClassSubjectMappingSaveRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
):
    result = save_class_subject_mapping(db, current_user, class_id, payload)
    db.commit()
    return result


@router.patch("/{class_id}", response_model=ClassResponse)
def classes_update_route(
    class_id: int,
    payload: ClassUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
):
    result = update_class(db, current_user, class_id, payload)
    db.commit()
    return result


@router.delete("/{class_id}", response_model=MessageResponse)
def classes_delete_route(
    class_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolStaffUser,
):
    delete_class(db, current_user, class_id)
    db.commit()
    return MessageResponse(message="Class removed.")
