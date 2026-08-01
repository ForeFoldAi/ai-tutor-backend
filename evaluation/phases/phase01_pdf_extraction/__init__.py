"""Phase 01 — PDF extraction evaluation."""

from __future__ import annotations

from evaluation.core.adapters.pdf import page_count, pipeline_to_serializable
from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._helpers import add, ensure_pipeline, new_report, timed


@register_phase
class PdfExtractionPhase(Phase):
    phase_id = "pdf_extraction"
    feature = "pdf_extraction"
    critical = True
    requires_heavy_ml = True

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        if not ctx.datasets:
            add(
                report,
                feature=self.feature,
                check_id="datasets_present",
                ok=False,
                critical=True,
                reason="No PDF datasets found under evaluation/datasets",
            )
            return report.finalize()

        page_scores = []
        for ds in ctx.datasets:
            expected_pages = page_count(ds.pdf_path)
            golden = ctx.goldens.load(ds.name, "pdf_extraction", default={}) or {}
            if golden.get("expected_pages") is not None:
                expected_pages = int(golden["expected_pages"])

            try:
                result, ms = timed(lambda: ensure_pipeline(ctx, ds))
            except Exception as exc:  # noqa: BLE001
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.pipeline_runs",
                    ok=False,
                    error=True,
                    critical=True,
                    reason=str(exc),
                    execution_ms=0,
                )
                report.failed_stage = "pdf_extraction"
                continue

            serial = pipeline_to_serializable(result)
            art = ctx.artifacts.write_json(f"{ds.name}/pdf/extraction.json", serial)
            report.artifacts.append(art)

            actual_pages = len(result.pages or [])
            page_nos = [p.page_no for p in (result.pages or [])]
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.all_pages_extracted",
                ok=actual_pages == expected_pages,
                critical=True,
                reason="Every PDF page must be extracted",
                expected=expected_pages,
                actual=actual_pages,
                execution_ms=ms,
                evidence={"page_nos": page_nos},
            )
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.no_duplicate_pages",
                ok=len(page_nos) == len(set(page_nos)),
                critical=True,
                reason="Page numbers must be unique",
                expected="unique page_nos",
                actual=page_nos,
            )
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.reading_order",
                ok=page_nos == sorted(page_nos),
                reason="Pages must be in ascending reading order",
                expected=sorted(page_nos),
                actual=page_nos,
            )

            # Bounding boxes / coordinates
            bad_bbox = 0
            headings = 0
            for p in result.pages or []:
                pw = p.page_info.width if p.page_info else None
                ph = p.page_info.height if p.page_info else None
                for det in p.layout_dets or []:
                    cat = (det.category_type or "").lower()
                    if "title" in cat or "heading" in cat or "section" in cat:
                        headings += 1
                    try:
                        x0, y0, x1, y1 = det.bbox
                    except Exception:
                        bad_bbox += 1
                        continue
                    if x0 > x1 or y0 > y1:
                        bad_bbox += 1
                    if pw and ph and (x0 < -5 or y0 < -5 or x1 > pw + 5 or y1 > ph + 5):
                        bad_bbox += 1

            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.bboxes_valid",
                ok=bad_bbox == 0,
                reason="All layout bounding boxes must be valid",
                expected=0,
                actual=bad_bbox,
            )

            figures = [a for a in (result.assets or []) if a.asset_type == "figure"]
            tables = [a for a in (result.assets or []) if a.asset_type == "table"]
            captions = [a for a in (result.assets or []) if (a.caption or "").strip()]

            if "min_figures" in golden:
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.images_extracted",
                    ok=len(figures) >= int(golden["min_figures"]),
                    critical=True,
                    reason="Figure count below golden minimum",
                    expected=f">={golden['min_figures']}",
                    actual=len(figures),
                )
            else:
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.images_extracted",
                    ok=True,
                    reason=f"Extracted {len(figures)} figures (no golden min)",
                    actual=len(figures),
                )

            if "min_tables" in golden:
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.tables_extracted",
                    ok=len(tables) >= int(golden["min_tables"]),
                    reason="Table count below golden minimum",
                    expected=f">={golden['min_tables']}",
                    actual=len(tables),
                )

            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.captions_present",
                ok=len(captions) > 0 or len(result.assets or []) == 0,
                reason="Assets should carry captions when figures/tables exist",
                actual={"captions": len(captions), "assets": len(result.assets or [])},
            )

            # Full text non-empty
            full_len = len(getattr(result, "full_text", "") or "")
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.text_extracted",
                ok=full_len > 100 or any((p.markdown or "").strip() for p in result.pages or []),
                critical=True,
                reason="Pipeline must produce page text/markdown",
                actual={"full_text_len": full_len, "headings_detected": headings},
            )
            page_scores.append(1.0 if actual_pages == expected_pages else 0.0)

            if ctx.config.get("runtime", "bootstrap_goldens"):
                ctx.goldens.save(
                    ds.name,
                    "pdf_extraction",
                    {
                        "expected_pages": expected_pages,
                        "min_figures": len(figures),
                        "min_tables": len(tables),
                        "asset_count": len(result.assets or []),
                    },
                )

        if page_scores:
            report.metrics["quality_score"] = sum(page_scores) / len(page_scores)
            report.metrics["datasets"] = float(len(page_scores))
        return report.finalize()
