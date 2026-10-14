"""Shared lesson-artifact evaluation logic."""

from __future__ import annotations

from typing import Any

from evaluation.core.adapters.lesson import generate_artifact_offline, validate_artifact_payload
from evaluation.core.types import PhaseReport
from evaluation.phases._helpers import add


def evaluate_artifact_kind(
    ctx,
    report: PhaseReport,
    *,
    kind: str,
    feature: str,
    quality_checks: Any,
) -> None:
    thr_key = {
        "lesson_plan": "lesson_quality_min",
        "quiz": "lesson_quality_min",
        "worksheet": "lesson_quality_min",
        "homework": "lesson_quality_min",
        "teaching_notes": "lesson_quality_min",
    }.get(kind, "lesson_quality_min")
    thr = float(ctx.config.threshold(thr_key))
    scores = []

    for ds in ctx.datasets:
        golden = ctx.goldens.load(ds.name, kind, default={}) or {}
        cases = golden.get("cases") or [{"topic": golden.get("topic") or ds.name.replace("-", " ")}]
        for i, case in enumerate(cases):
            topic = case.get("topic") or ds.name
            context = case.get("context") or f"CBSE {ds.class_level} {ds.subject_name} chapter on {topic}"
            # Prefer golden expected payload for schema regression when provided
            if case.get("expected_payload"):
                payload = case["expected_payload"]
                source = "golden_expected"
            else:
                gen = generate_artifact_offline(kind, topic, context)
                payload = gen.get("payload") or {}
                source = gen.get("source", "unknown")

            ok, reason, dumped = validate_artifact_payload(kind, payload)
            add(
                report,
                feature=feature,
                check_id=f"{ds.name}.{kind}.case{i}.schema",
                ok=ok,
                critical=True,
                reason=reason if not ok else "schema_ok",
                expected=f"{kind} pydantic schema",
                actual=payload if not ok else {"keys": list(payload.keys()), "source": source},
            )
            if not ok or dumped is None:
                continue

            q = quality_checks(dumped, case)
            scores.append(q["score"])
            ctx.artifacts.write_json(
                f"{ds.name}/{kind}/case_{i}.json",
                {"topic": topic, "source": source, "payload": dumped, "quality": q},
            )
            for chk in q.get("checks", []):
                add(
                    report,
                    feature=feature,
                    check_id=f"{ds.name}.{kind}.case{i}.{chk['id']}",
                    ok=chk["ok"],
                    critical=chk.get("critical", False),
                    reason=chk["reason"],
                    expected=chk.get("expected"),
                    actual=chk.get("actual"),
                )
            add(
                report,
                feature=feature,
                check_id=f"{ds.name}.{kind}.case{i}.quality",
                ok=q["score"] >= thr,
                critical=True,
                reason=f"{kind} quality below threshold",
                expected=thr,
                actual=q["score"],
                metric_name=f"{kind}_quality_score",
                metric_value=q["score"],
            )

    if scores:
        report.metrics[f"{kind}_quality_score"] = sum(scores) / len(scores)
        report.metrics["quality_score"] = report.metrics[f"{kind}_quality_score"]
