from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.school_admin.credentials.constants import DEFAULT_PAGE_LIMIT
from app.modules.school_admin.credentials.schemas import (
    CredentialActionRequest,
    CredentialCandidatesResponse,
    CredentialGenerateResponse,
    CredentialListResponse,
    CredentialMetrics,
    CredentialSendResponse,
)
from app.modules.school_admin.credentials.service import (
    credential_stats,
    export_credentials_xlsx,
    generate_credentials,
    list_candidates,
    list_credentials,
    send_credentials,
)
from app.modules.users.models import User

router = APIRouter(prefix="/auth/admin/credentials", tags=["school-admin-credentials"])

SchoolAdminUser = Annotated[
    User, Depends(require_roles(Role.SCHOOL_ADMIN, Role.MASTER_ADMIN, Role.TUTOR))
]


@router.get("/stats", response_model=CredentialMetrics)
def credentials_stats_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    return credential_stats(db, current_user)


@router.get("/candidates", response_model=CredentialCandidatesResponse)
def credentials_candidates_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
    role: Annotated[str, Query(description="Teacher | Student")],
    q: Annotated[str | None, Query()] = None,
    grade: Annotated[str | None, Query()] = None,
    section: Annotated[str | None, Query()] = None,
    credential_status: Annotated[str | None, Query(description="needs | has | all")] = None,
):
    return list_candidates(
        db,
        current_user,
        role=role,
        q=q,
        grade=grade,
        section=section,
        credential_status=credential_status,
    )


@router.get("/export")
def credentials_export_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
    q: Annotated[str | None, Query()] = None,
    role: Annotated[str | None, Query()] = None,
    first_login_status: Annotated[str | None, Query()] = None,
    delivery_status: Annotated[str | None, Query()] = None,
    shared: Annotated[str | None, Query()] = None,
):
    content = export_credentials_xlsx(
        db,
        current_user,
        q=q,
        role=role,
        first_login_status=first_login_status,
        delivery_status=delivery_status,
        shared=shared,
    )
    db.commit()
    filename = f"credentials-{datetime_stamp()}.xlsx"
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def datetime_stamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%d")


@router.get("", response_model=CredentialListResponse)
def list_credentials_route(
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
    q: Annotated[str | None, Query()] = None,
    role: Annotated[str | None, Query()] = None,
    first_login_status: Annotated[str | None, Query()] = None,
    delivery_status: Annotated[str | None, Query()] = None,
    shared: Annotated[str | None, Query()] = None,
    limit: Annotated[int | None, Query(ge=1, le=100)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return list_credentials(
        db,
        current_user,
        q=q,
        role=role,
        first_login_status=first_login_status,
        delivery_status=delivery_status,
        shared=shared,
        limit=limit or DEFAULT_PAGE_LIMIT,
        offset=offset,
    )


@router.post("/generate", response_model=CredentialGenerateResponse)
def credentials_generate_route(
    payload: CredentialActionRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    result = generate_credentials(db, current_user, payload)
    db.commit()
    return result


@router.post("/send", response_model=CredentialSendResponse)
def credentials_send_route(
    payload: CredentialActionRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: SchoolAdminUser,
):
    result = send_credentials(db, current_user, payload)
    db.commit()
    return result
