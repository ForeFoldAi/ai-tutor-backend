"""Phase 08 — Topic / section hierarchy detection."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._helpers import add, ensure_chunks, new_report


@register_phase
class TopicsPhase(Phase):
    phase_id = "topics"
    feature = "topics"
    requires_heavy_ml = True
    depends_on = ["chunking"]

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        for ds in ctx.datasets:
            chunks = ensure_chunks(ctx, ds)
            golden = ctx.goldens.load(ds.name, "topics", default={}) or {}
            hints = []
            for c in chunks:
                meta = getattr(c, "metadata", {}) or {}
                h = meta.get("section_hint")
                if h:
                    hints.append(h)
            unique = sorted(set(hints))
            ctx.artifacts.write_json(
                f"{ds.name}/topics/detected.json",
                {"topics": unique, "count": len(unique)},
            )

            expected = golden.get("topics") or []
            missing = [t for t in expected if not any(t.lower() in u.lower() for u in unique)]
            extras_forbidden = golden.get("forbidden_topics") or []
            extras = [t for t in extras_forbidden if any(t.lower() in u.lower() for u in unique)]

            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.topics_detected",
                ok=len(unique) > 0,
                reason="No section/topic hints detected",
                actual={"count": len(unique), "sample": unique[:10]},
            )
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.expected_topics",
                ok=len(missing) == 0,
                critical=bool(expected),
                reason="Missing expected topics",
                expected=expected,
                actual={"missing": missing, "detected": unique},
            )
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.no_forbidden_topics",
                ok=len(extras) == 0,
                reason="Forbidden/extra topics present",
                actual=extras,
            )

            # Hierarchy: numbered headings like 1.2 should exist if golden says so
            if golden.get("require_numbered_hierarchy"):
                numbered = [u for u in unique if __import__("re").match(r"^\d+(\.\d+)*\b", u)]
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.hierarchy",
                    ok=len(numbered) > 0,
                    reason="No numbered topic hierarchy found",
                    actual=numbered[:10],
                )
            report.metrics["topic_count"] = float(len(unique))
            report.metrics["quality_score"] = 1.0 if not missing else max(0.0, 1 - len(missing) / max(1, len(expected)))
        return report.finalize()
