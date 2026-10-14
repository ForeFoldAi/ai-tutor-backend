from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.live_sessions import service as live_service
from app.modules.live_sessions.schemas import (
    LiveSessionCreateRequest,
    LiveSessionJoinResponse,
    LiveSessionOut,
    LiveSessionUpdateRequest,
)
from app.modules.auth.schemas import MessageResponse
from app.modules.users.models import User

tutor_router = APIRouter(prefix="/auth/tutor/live-sessions", tags=["tutor-live-sessions"])
student_router = APIRouter(prefix="/auth/student/live-sessions", tags=["student-live-sessions"])

TutorUser = Annotated[User, Depends(require_roles(Role.TUTOR))]
StudentUser = Annotated[User, Depends(require_roles(Role.STUDENT))]


@tutor_router.get("", response_model=list[LiveSessionOut])
def tutor_list_sessions(
    db: Annotated[Session, Depends(get_db)],
    current_user: TutorUser,
):
    return live_service.list_tutor_sessions(db, current_user)


@tutor_router.post("", response_model=LiveSessionOut, status_code=status.HTTP_201_CREATED)
def tutor_create_session(
    payload: LiveSessionCreateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: TutorUser,
):
    result = live_service.create_session(db, current_user, payload)
    db.commit()
    return result


@tutor_router.patch("/{session_id}", response_model=LiveSessionOut)
def tutor_update_session(
    session_id: int,
    payload: LiveSessionUpdateRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: TutorUser,
):
    result = live_service.update_tutor_session(db, current_user, session_id, payload)
    db.commit()
    return result


@tutor_router.delete("/{session_id}", response_model=MessageResponse)
def tutor_delete_session(
    session_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: TutorUser,
):
    live_service.delete_tutor_session(db, current_user, session_id)
    db.commit()
    return MessageResponse(message="Session deleted.")


@student_router.get("", response_model=list[LiveSessionOut])
def student_list_sessions(
    db: Annotated[Session, Depends(get_db)],
    current_user: StudentUser,
):
    return live_service.list_student_sessions(db, current_user)


@student_router.post("/{session_id}/join", response_model=LiveSessionJoinResponse)
def student_join_session(
    session_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: StudentUser,
):
    result = live_service.join_session(db, current_user, session_id)
    db.commit()
    return result
