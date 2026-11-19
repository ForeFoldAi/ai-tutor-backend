from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.dependencies import get_current_user
from app.modules.student_learning import service as learning_service
from app.modules.student_learning.schemas import (
    ChapterCompleteResponse,
    LearningOverviewResponse,
    SessionEndResponse,
    SessionHeartbeatResponse,
    SessionStartRequest,
    SessionStartResponse,
    TutorChatPutRequest,
    TutorChatResponse,
)
from app.modules.users.models import User

router = APIRouter(prefix="/auth/student/learning", tags=["student-learning"])


@router.get("/overview", response_model=LearningOverviewResponse)
def learning_overview(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    result = learning_service.get_overview(db, current_user)
    db.commit()
    return result


@router.post("/sessions/start", response_model=SessionStartResponse)
def start_learning_session(
    payload: SessionStartRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    result = learning_service.start_session(db, current_user, payload)
    db.commit()
    return result


@router.post("/sessions/{session_id}/heartbeat", response_model=SessionHeartbeatResponse)
def heartbeat_learning_session(
    session_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    result = learning_service.heartbeat_session(db, current_user, session_id)
    db.commit()
    return result


@router.post("/sessions/{session_id}/end", response_model=SessionEndResponse)
def end_learning_session(
    session_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    result = learning_service.end_session(db, current_user, session_id)
    db.commit()
    return result


@router.post("/chapters/{chapter_id}/complete", response_model=ChapterCompleteResponse)
def complete_learning_chapter(
    chapter_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    result = learning_service.complete_chapter(db, current_user, chapter_id)
    db.commit()
    return result


@router.get("/chapters/{chapter_id}/chat", response_model=TutorChatResponse)
def get_chapter_tutor_chat(
    chapter_id: int,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    return learning_service.get_tutor_chat(db, current_user, chapter_id)


@router.put("/chapters/{chapter_id}/chat", response_model=TutorChatResponse)
def put_chapter_tutor_chat(
    chapter_id: int,
    payload: TutorChatPutRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
):
    result = learning_service.put_tutor_chat(db, current_user, chapter_id, payload)
    db.commit()
    return result
