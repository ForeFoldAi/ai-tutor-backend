"""Long-horizon trend analysis from metric history."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.learning_intelligence.constants import METRIC_CONFIDENCE, METRIC_ENGAGEMENT, METRIC_PERFORMANCE
from app.modules.learning_intelligence.models import LiaMetricHistory


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((xs[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n)) or 1.0
    return num / den


def metric_series(
    db: Session,
    student_user_id: int,
    metric_type: str,
    *,
    days: int = 30,
    limit: int = 200,
) -> list[float]:
    since = datetime.now(UTC) - timedelta(days=days)
    rows = db.scalars(
        select(LiaMetricHistory)
        .where(
            LiaMetricHistory.student_user_id == student_user_id,
            LiaMetricHistory.metric_type == metric_type,
            LiaMetricHistory.recorded_at >= since,
        )
        .order_by(LiaMetricHistory.recorded_at.asc())
        .limit(limit)
    ).all()
    return [float(r.value) for r in rows]


def compute_learning_trends(db: Session, student_user_id: int) -> dict[str, float]:
    """Improvement/regression slopes from confidence, engagement, mastery snapshots."""
    conf = metric_series(db, student_user_id, METRIC_CONFIDENCE, days=30)
    eng = metric_series(db, student_user_id, METRIC_ENGAGEMENT, days=30)
    perf = metric_series(db, student_user_id, METRIC_PERFORMANCE, days=30)

    conf_slope = _slope(conf)
    eng_slope = _slope(eng)
    perf_slope = _slope(perf)

    composite = conf_slope * 0.4 + eng_slope * 0.3 + perf_slope * 0.3
    improvement = max(0.0, composite)
    regression = max(0.0, -composite)

    return {
        "confidence_slope": round(conf_slope, 4),
        "engagement_slope": round(eng_slope, 4),
        "mastery_slope": round(perf_slope, 4),
        "improvement_trend": round(improvement, 4),
        "regression_trend": round(regression, 4),
    }
