from __future__ import annotations

from prometheus_client import Counter, Histogram

GENERATION_JOBS = Counter(
    "lesson_planner_generation_jobs_total",
    "Lesson planner generation jobs",
    ["status"],
)
GENERATION_LATENCY = Histogram(
    "lesson_planner_generation_seconds",
    "Lesson planner generation latency",
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1200),
)
ARTIFACT_GENERATION = Counter(
    "lesson_planner_artifacts_total",
    "Lesson planner artifacts generated",
    ["artifact_type", "status"],
)
