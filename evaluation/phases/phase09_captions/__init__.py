"""Phase 09 — Caption generation / figure association."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._helpers import add, ensure_pipeline, new_report


@register_phase
class CaptionsPhase(Phase):
    phase_id = "captions"
    feature = "captions"
    requires_heavy_ml = True
    depends_on = ["images"]

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        thr = ctx.config.threshold("caption_accuracy_min")
        scores = []
        for ds in ctx.datasets:
            result = ensure_pipeline(ctx, ds)
            golden = ctx.goldens.load(ds.name, "captions", default={}) or {}
            figures = [a for a in (result.assets or []) if a.asset_type in ("figure", "image", "table")]
            expected = {str(c.get("figure") or c.get("number")): c for c in golden.get("captions", [])}

            hits = 0
            total = max(1, len(expected) or len(figures))
            for i, fig in enumerate(figures):
                fid = str(fig.number or f"idx{i}")
                cap = (fig.caption or "").strip()
                exp = expected.get(fid) or expected.get(str(fig.number))
                if exp:
                    exp_text = (exp.get("text") or exp.get("caption") or "").strip()
                    ok = bool(cap) and (
                        exp_text.lower() in cap.lower()
                        or cap.lower() in exp_text.lower()
                        or _token_overlap(cap, exp_text) >= 0.4
                    )
                    page_ok = int(fig.page_no) == int(exp["page"]) if "page" in exp else True
                    if ok:
                        hits += 1
                    add(
                        report,
                        feature=self.feature,
                        check_id=f"{ds.name}.caption.{fid}",
                        ok=ok and page_ok,
                        reason="Caption/page association mismatch",
                        expected=exp,
                        actual={"caption": cap, "page": fig.page_no},
                    )
                else:
                    # No golden — require non-empty caption for numbered figures
                    if fig.number:
                        add(
                            report,
                            feature=self.feature,
                            check_id=f"{ds.name}.caption_present.{fid}",
                            ok=bool(cap),
                            reason="Numbered figure missing caption",
                            actual=cap,
                        )
                        if cap:
                            hits += 1
                        total = max(total, hits + 1)

            acc = hits / total if expected else (hits / max(1, len([f for f in figures if f.number])))
            scores.append(acc)
            report.metrics["caption_accuracy"] = acc
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.caption_accuracy",
                ok=acc >= thr,
                reason="Caption accuracy below threshold",
                expected=thr,
                actual=acc,
                metric_name="caption_accuracy",
                metric_value=acc,
            )
            ctx.artifacts.write_json(
                f"{ds.name}/captions/extracted.json",
                [
                    {"number": a.number, "page": a.page_no, "caption": a.caption, "type": a.asset_type}
                    for a in figures
                ],
            )
        if scores:
            report.metrics["quality_score"] = sum(scores) / len(scores)
        return report.finalize()


def _token_overlap(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
