from __future__ import annotations

import uuid
from datetime import UTC, datetime

import factory
from factory.alchemy import SQLAlchemyModelFactory

from app.core.database import SessionLocal
from app.modules.teacher.lesson_planner.constants import ArtifactStatus, ArtifactType, JobStatus, LessonPlanStatus
from app.modules.teacher.lesson_planner.models import LessonArtifact, LessonJob, LessonPlan
from app.modules.users.models import User


class LessonPlanFactory(SQLAlchemyModelFactory):
    class Meta:
        model = LessonPlan
        sqlalchemy_session = SessionLocal()
        sqlalchemy_session_persistence = "commit"

    id = factory.LazyFunction(uuid.uuid4)
    user_id = factory.LazyFunction(uuid.uuid4)
    title = factory.Sequence(lambda n: f"Lesson Plan {n}")
    grade = "CLASS_8"
    subject = "Science"
    board = "CBSE"
    chapter_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    chapter_name = "Photosynthesis"
    duration_minutes = 45
    learning_objectives = "Understand photosynthesis"
    status = LessonPlanStatus.DRAFT
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))


class LessonJobFactory(SQLAlchemyModelFactory):
    class Meta:
        model = LessonJob
        sqlalchemy_session = SessionLocal()
        sqlalchemy_session_persistence = "commit"

    id = factory.LazyFunction(uuid.uuid4)
    lesson_plan = factory.SubFactory(LessonPlanFactory)
    lesson_plan_id = factory.SelfAttribute("lesson_plan.id")
    user_id = factory.SelfAttribute("lesson_plan.user_id")
    status = JobStatus.QUEUED
    progress = 0
    requested_artifacts = factory.LazyFunction(lambda: [ArtifactType.LESSON_PLAN.value])
    input_payload = factory.LazyFunction(dict)
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))


class LessonArtifactFactory(SQLAlchemyModelFactory):
    class Meta:
        model = LessonArtifact
        sqlalchemy_session = SessionLocal()
        sqlalchemy_session_persistence = "commit"

    id = factory.LazyFunction(uuid.uuid4)
    lesson_plan = factory.SubFactory(LessonPlanFactory)
    lesson_plan_id = factory.SelfAttribute("lesson_plan.id")
    artifact_type = ArtifactType.LESSON_PLAN
    content = factory.LazyFunction(
        lambda: {"title": "Test", "duration": 45, "objectives": ["Learn"], "activities": []}
    )
    status = ArtifactStatus.COMPLETED
    version_number = 1
    created_at = factory.LazyFunction(lambda: datetime.now(UTC))
    updated_at = factory.LazyFunction(lambda: datetime.now(UTC))
