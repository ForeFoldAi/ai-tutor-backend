"""Progress analytics pure helpers (no DB)."""

from __future__ import annotations

from app.modules.teacher.progress_analytics.schemas import ProgressAnalyticsResponse
from app.modules.teacher.progress_analytics.service import _health_status


def test_health_status_thresholds() -> None:
    assert _health_status(75) == "Good"
    assert _health_status(70) == "Good"
    assert _health_status(50) == "Fair"
    assert _health_status(45) == "Fair"
    assert _health_status(20) == "Needs Attention"


def test_progress_analytics_response_shape() -> None:
    payload = ProgressAnalyticsResponse.model_validate(
        {
            "class_health": {
                "score": 62,
                "status": "Fair",
                "trend_percent": 4,
                "trend_up": True,
            },
            "topic_mastery": [{"topic": "Fractions", "mastery": 55}],
            "completion_trend": [{"date": "Jul 15", "completion": 48}],
            "at_risk_students": [
                {
                    "id": 1,
                    "slug": "aaron-patel",
                    "name": "Aaron Patel",
                    "grade": "7",
                    "subject": "Algebra",
                    "risk_level": "High",
                }
            ],
        }
    )
    assert payload.class_health.score == 62
    assert payload.topic_mastery[0].topic == "Fractions"
    assert payload.at_risk_students[0].risk_level == "High"
