"""Phase 03 — Table extraction evaluation."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.metrics.table_quality import score_table
from evaluation.phases._helpers import add, ensure_pipeline, new_report


@register_phase
class TablesPhase(Phase):
    phase_id = "tables"
    feature = "tables"
    critical = False
    requires_heavy_ml = True
    depends_on = ["pdf_extraction"]

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        thr = ctx.config.threshold("table_quality_min")
        scores = []
        for ds in ctx.datasets:
            result = ensure_pipeline(ctx, ds)
            tables = [a for a in (result.assets or []) if a.asset_type == "table"]
            golden = ctx.goldens.load(ds.name, "tables", default={}) or {}
            min_tables = golden.get("min_tables")
            if min_tables is not None:
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.count",
                    ok=len(tables) >= int(min_tables),
                    critical=True,
                    reason="Missing tables vs golden",
                    expected=min_tables,
                    actual=len(tables),
                )

            golden_by_id = {str(t.get("id") or t.get("number")): t for t in golden.get("tables", [])}

            for i, tbl in enumerate(tables):
                tid = str(tbl.number or f"t{i}")
                g = golden_by_id.get(tid)
                scored = score_table(tbl.structured_text or "", g)
                scores.append(scored["score"])
                ctx.artifacts.write_json(
                    f"{ds.name}/tables/{tid}.json",
                    {
                        "page": tbl.page_no,
                        "caption": tbl.caption,
                        "structured_text": tbl.structured_text,
                        "score": scored,
                    },
                )
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.table.{tid}",
                    ok=scored["score"] >= thr,
                    reason="PASS" if scored["score"] >= thr else f"Table issues: {scored['issues']}",
                    expected={"min_score": thr, "golden": g},
                    actual=scored,
                    evidence={"page": tbl.page_no, "caption": tbl.caption},
                    metric_name="table_quality_score",
                    metric_value=scored["score"],
                )
                if g and "title" in g:
                    title_ok = (g["title"] or "").lower() in (tbl.caption or "").lower() or (
                        g["title"] or ""
                    ).lower() in (tbl.structured_text or "").lower()
                    add(
                        report,
                        feature=self.feature,
                        check_id=f"{ds.name}.table.{tid}.title",
                        ok=title_ok,
                        reason="Table title/caption mismatch",
                        expected=g["title"],
                        actual=tbl.caption,
                    )

            if not tables and min_tables is None:
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.no_tables_optional",
                    ok=True,
                    skip=True,
                    reason="No tables extracted and no golden minimum set",
                )

        if scores:
            mean = sum(scores) / len(scores)
            report.metrics["table_quality_score"] = mean
            report.metrics["quality_score"] = mean
        return report.finalize()
