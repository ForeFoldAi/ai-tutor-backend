"""LIA REST APIs — internal + tutor-facing intelligence (never student chat)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import get_current_user, require_roles
from app.modules.learning_intelligence import service as lia_svc
from app.modules.learning_intelligence.dependencies import require_lia_internal_token
from app.modules.learning_intelligence.schemas import (
    KnowledgeMapOut,
    LearningEventIn,
    LearningEventResponse,
    PredictionOut,
    StudentDigitalTwinOut,
    TeacherSummaryOut,
    TutorClassInsightsOut,
    TutorGuidanceObject,
    TutorGuidanceRequest,
)
from app.modules.teacher.students.service import assert_tutor_assigned_student
from app.modules.users.models import User

internal_router = APIRouter(
    prefix="/internal/lia",
    tags=["lia-internal"],
    dependencies=[Depends(require_lia_internal_token)],
)

tutor_router = APIRouter(prefix="/auth/tutor/lia", tags=["lia-teacher"])

TutorUser = Annotated[User, Depends(require_roles(Role.TUTOR))]
DbSession = Annotated[Session, Depends(get_db)]


@internal_router.post("/concept-graph/sync")
def sync_concept_graph_internal(
    db: DbSession,
    board: str | None = None,
    class_level: str | None = None,
    subject_name: str | None = None,
):
    return lia_svc.sync_concept_graph(
        db, board=board, class_level=class_level, subject_name=subject_name
    )


@internal_router.post("/events", response_model=LearningEventResponse)
def post_event(payload: LearningEventIn, db: DbSession):
    result = lia_svc.ingest_event(db, payload)
    db.commit()
    return result


@internal_router.get("/student/{student_id}/tutor-guidance", response_model=TutorGuidanceObject)
def get_tutor_guidance_internal(
    student_id: int,
    db: DbSession,
    query: str = "",
    topic: str = "",
    subject_name: str = "",
    chapter: str = "",
    class_level: str = "",
    board: str = "",
    agent_mode: str | None = None,
):
    req = TutorGuidanceRequest(
        student_user_id=student_id,
        query=query,
        topic=topic or query,
        subject_name=subject_name,
        chapter=chapter,
        class_level=class_level,
        board=board,
        agent_mode=agent_mode,
    )
    result = lia_svc.fetch_tutor_guidance(db, req)
    db.commit()
    return result


@internal_router.get("/student/{student_id}/profile", response_model=StudentDigitalTwinOut)
def get_profile_internal(student_id: int, db: DbSession):
    result = lia_svc.fetch_student_profile(db, student_id)
    db.commit()
    return result


@internal_router.get("/student/{student_id}/knowledge-map", response_model=KnowledgeMapOut)
def get_knowledge_map_internal(student_id: int, db: DbSession):
    return lia_svc.fetch_knowledge_map(db, student_id)


@internal_router.post("/student/{student_id}/predict", response_model=PredictionOut)
def predict_internal(student_id: int, db: DbSession, concept_key: str | None = None):
    result = lia_svc.run_prediction(db, student_id, concept_key)
    db.commit()
    return result


@tutor_router.get("/insights", response_model=TutorClassInsightsOut)
def tutor_class_insights(db: DbSession, current_user: TutorUser):
    result = lia_svc.fetch_class_insights(db, current_user)
    db.commit()
    return result


@tutor_router.get("/students/{student_id}/teacher-summary", response_model=TeacherSummaryOut)
def tutor_teacher_summary(student_id: int, db: DbSession, current_user: TutorUser):
    assert_tutor_assigned_student(db, current_user, student_id)
    return lia_svc.fetch_teacher_summary(db, student_id)


@tutor_router.get("/students/{student_id}/profile", response_model=StudentDigitalTwinOut)
def tutor_student_twin(student_id: int, db: DbSession, current_user: TutorUser):
    assert_tutor_assigned_student(db, current_user, student_id)
    result = lia_svc.fetch_student_profile(db, student_id)
    db.commit()
    return result


@tutor_router.get("/students/{student_id}/knowledge-map", response_model=KnowledgeMapOut)
def tutor_knowledge_map(student_id: int, db: DbSession, current_user: TutorUser):
    assert_tutor_assigned_student(db, current_user, student_id)
    return lia_svc.fetch_knowledge_map(db, student_id)
