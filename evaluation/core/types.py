"""Canonical result types for every evaluation check and phase."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"
    WARN = "WARN"


@dataclass
class CheckResult:
    """One atomic assertion inside a feature evaluator."""

    feature: str
    check_id: str
    status: Status
    reason: str
    expected: Any = None
    actual: Any = None
    evidence: dict[str, Any] = field(default_factory=dict)
    execution_ms: float = 0.0
    confidence: float = 1.0
    critical: bool = False
    metric_name: str | None = None
    metric_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class PhaseReport:
    """Aggregate result for one evaluation phase / feature module."""

    phase_id: str
    feature: str
    status: Status
    checks: list[CheckResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    execution_ms: float = 0.0
    failed_stage: str | None = None
    dataset: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == Status.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status in (Status.FAIL, Status.ERROR))

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.critical and c.status in (Status.FAIL, Status.ERROR)]

    def finalize(self) -> PhaseReport:
        if any(c.status == Status.ERROR for c in self.checks):
            self.status = Status.ERROR
        elif self.critical_failures:
            self.status = Status.FAIL
        elif self.checks and all(c.status == Status.SKIP for c in self.checks):
            self.status = Status.SKIP
        elif any(c.status == Status.FAIL for c in self.checks):
            # Soft (non-critical) failures → WARN so pipeline can continue
            self.status = Status.WARN
        elif any(c.status == Status.WARN for c in self.checks) and not any(
            c.status == Status.FAIL for c in self.checks
        ):
            self.status = Status.PASS if self.passed else Status.WARN
        else:
            self.status = Status.PASS
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "feature": self.feature,
            "status": self.status.value,
            "passed": self.passed,
            "failed": self.failed,
            "execution_ms": self.execution_ms,
            "failed_stage": self.failed_stage,
            "dataset": self.dataset,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "meta": self.meta,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class RunReport:
    """Full evaluation run across phases and datasets."""

    run_id: str
    started_at: float
    finished_at: float = 0.0
    phases: list[PhaseReport] = field(default_factory=list)
    overall_status: Status = Status.PASS
    overall_score: float = 0.0
    regression: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> RunReport:
        self.finished_at = time.time()
        if any(p.status == Status.ERROR for p in self.phases):
            self.overall_status = Status.ERROR
        elif any(p.status == Status.FAIL for p in self.phases):
            self.overall_status = Status.FAIL
        else:
            self.overall_status = Status.PASS
        # Score only decisive checks (ignore SKIP)
        decisive = [
            c
            for p in self.phases
            for c in p.checks
            if c.status in (Status.PASS, Status.FAIL, Status.ERROR, Status.WARN)
        ]
        if decisive:
            ok = sum(1 for c in decisive if c.status in (Status.PASS, Status.WARN))
            self.overall_score = ok / len(decisive)
        else:
            self.overall_score = 1.0 if self.overall_status == Status.PASS else 0.0
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": (self.finished_at - self.started_at) * 1000,
            "overall_status": self.overall_status.value,
            "overall_score": self.overall_score,
            "regression": self.regression,
            "config_snapshot": self.config_snapshot,
            "phases": [p.to_dict() for p in self.phases],
        }


def check(
    feature: str,
    check_id: str,
    *,
    ok: bool,
    reason: str,
    expected: Any = None,
    actual: Any = None,
    evidence: dict[str, Any] | None = None,
    execution_ms: float = 0.0,
    confidence: float = 1.0,
    critical: bool = False,
    metric_name: str | None = None,
    metric_value: float | None = None,
    skip: bool = False,
    error: bool = False,
) -> CheckResult:
    if skip:
        status = Status.SKIP
    elif error:
        status = Status.ERROR
    elif ok:
        status = Status.PASS
    else:
        status = Status.FAIL
    return CheckResult(
        feature=feature,
        check_id=check_id,
        status=status,
        reason=reason,
        expected=expected,
        actual=actual,
        evidence=evidence or {},
        execution_ms=execution_ms,
        confidence=confidence,
        critical=critical,
        metric_name=metric_name,
        metric_value=metric_value,
    )
