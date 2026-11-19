"""Tutor dashboard schema smoke test (no DB)."""

from __future__ import annotations

from app.modules.teacher.dashboard.schemas import (
    TutorAiRecommendation,
    TutorDashboardMetrics,
    TutorDashboardSummaryResponse,
)


def test_tutor_dashboard_summary_shape() -> None:
    payload = TutorDashboardSummaryResponse(
        metrics=TutorDashboardMetrics(total_assigned=3, attention_required=1),
        ai_recommendations=[
            TutorAiRecommendation(
                id="1",
                title="Fractions revision",
                description="Students struggling with fractions",
                icon="worksheet",
                action_label="Generate Worksheet",
                action_href="/tutor/lesson-planner",
            )
        ],
    )
    dumped = payload.model_dump()
    assert dumped["metrics"]["total_assigned"] == 3
    assert dumped["ai_recommendations"][0]["action_href"] == "/tutor/lesson-planner"
