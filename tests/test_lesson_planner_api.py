from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_db
from app.main import app
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import get_current_user
from app.modules.users.models import User


@pytest.fixture
def tutor_user() -> User:
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.full_name = "Test Tutor"
    user.email = "tutor-lesson@test.local"
    user.role = Role.TUTOR
    user.is_active = True
    return user


@pytest.fixture
def authed_client(tutor_user: User):
    db = MagicMock()

    def override_db():
        yield db

    app.dependency_overrides[get_current_user] = lambda: tutor_user
    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as client:
        yield client, db
    app.dependency_overrides.clear()


def test_lesson_planner_metrics_endpoint():
    with TestClient(app) as c:
        resp = c.get("/metrics/lesson-planner")
    assert resp.status_code == 200
    assert "lesson_planner_generation_jobs_total" in resp.text


def test_list_plans_requires_auth():
    with TestClient(app) as c:
        resp = c.get("/api/lesson-planner")
    assert resp.status_code == 401


def test_chapter_topics_returns_headings(authed_client, tutor_user: User):
    client, _db = authed_client
    with patch(
        "app.modules.student_learning.topic_progress.list_chapter_topics",
        return_value=[{"key": "mixed numbers", "title": "Mixed Numbers"}],
    ):
        resp = client.get(
            "/api/lesson-planner/chapter-topics",
            params={
                "chapter_id": "12",
                "subject": "Mathematics",
                "board": "CBSE",
                "class_level": "CLASS_9",
                "chapter_name": "Fractions",
            },
            headers={"Authorization": "Bearer test"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chapter_id"] == "12"
    assert body["topics"] == [{"key": "mixed numbers", "title": "Mixed Numbers"}]


@patch("app.modules.teacher.lesson_planner.router._check_rate_limit")
@patch("app.modules.teacher.lesson_planner.router.lesson_generation_worker")
def test_generate_returns_202(mock_worker: MagicMock, mock_rate: MagicMock, authed_client, tutor_user: User):
    client, db = authed_client
    plan_id = uuid.uuid4()
    job_id = uuid.uuid4()

    mock_worker.delay.return_value = MagicMock(id="celery-task-1")

    with patch("app.modules.teacher.lesson_planner.router.create_lesson_plan_from_request") as mock_plan:
        with patch("app.modules.teacher.lesson_planner.router.create_generation_job") as mock_job:
            plan = MagicMock()
            plan.id = plan_id
            mock_plan.return_value = plan

            job = MagicMock()
            job.id = job_id
            job.celery_task_id = None
            mock_job.return_value = job

            resp = client.post(
                "/api/lesson-planner/generate",
                json={
                    "grade": "CLASS_8",
                    "subject": "Science",
                    "chapter_name": "Photosynthesis",
                    "duration_minutes": 45,
                    "learning_objectives": "Understand photosynthesis",
                    "requested_artifacts": ["lesson_plan", "worksheet"],
                },
                headers={"Authorization": "Bearer test"},
            )

    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] == str(job_id)
    assert body["lesson_plan_id"] == str(plan_id)
    assert body["websocket_url"] == f"/ws/lesson-planner/{job_id}"
    mock_worker.delay.assert_called_once_with(str(job_id))
    assert db.commit.called
