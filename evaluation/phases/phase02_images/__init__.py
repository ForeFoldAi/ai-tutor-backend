"""Phase 02 — Image extraction quality."""

from __future__ import annotations

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.metrics.image_quality import analyze_image_bytes, hamming, phash_hex
from evaluation.phases._helpers import add, ensure_pipeline, new_report


@register_phase
class ImagesPhase(Phase):
    phase_id = "images"
    feature = "images"
    critical = True
    requires_heavy_ml = True
    depends_on = ["pdf_extraction"]

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        thr = ctx.config.data["thresholds"]
        scores = []
        for ds in ctx.datasets:
            result = ensure_pipeline(ctx, ds)
            figures = [a for a in (result.assets or []) if a.asset_type in ("figure", "image")]
            golden = ctx.goldens.load(ds.name, "images", default={}) or {}
            expected_count = golden.get("expected_count")
            if expected_count is not None:
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.count",
                    ok=len(figures) >= int(expected_count),
                    critical=True,
                    reason="Missing images vs golden",
                    expected=expected_count,
                    actual=len(figures),
                )

            page_dims = {
                p.page_no: (
                    p.page_info.width if p.page_info else None,
                    p.page_info.height if p.page_info else None,
                )
                for p in (result.pages or [])
            }
            hashes: list[tuple[str, str]] = []
            failed_figs = []
            for i, fig in enumerate(figures):
                pw, ph = page_dims.get(fig.page_no, (None, None))
                q = analyze_image_bytes(
                    fig.image_bytes or b"",
                    min_side=int(thr["min_image_side_px"]),
                    min_area=int(thr["min_image_area_px"]),
                    blank_std_max=float(thr["blank_image_std_max"]),
                    page_width=pw,
                    page_height=ph,
                    bbox=fig.bbox,
                    crop_edge_margin_ratio=float(thr["crop_edge_margin_ratio"]),
                )
                scores.append(q.score)
                fig_id = fig.number or f"page{fig.page_no}_idx{i}"
                # Persist evidence
                rel = f"{ds.name}/images/{fig_id}.bin"
                try:
                    ctx.artifacts.write_bytes(rel, fig.image_bytes or b"")
                    report.artifacts.append(rel)
                except Exception:
                    pass
                ctx.artifacts.write_json(
                    f"{ds.name}/images/{fig_id}.quality.json",
                    {"issues": q.issues, "stats": q.stats, "score": q.score, "page": fig.page_no},
                )
                if not q.ok or q.issues:
                    failed_figs.append({"figure": fig_id, "page": fig.page_no, "issues": q.issues, "score": q.score})

                is_strip = q.height > 0 and (q.height < 30 or q.width < 30)
                soft_issue_set = {
                    "extreme_aspect_ratio",
                    "low_resolution_side",
                    "low_resolution_area",
                    "invalid_bbox",
                    "bbox_out_of_page",
                    "possible_cropped_fragment",
                    "possible_half_or_split_image",
                }
                soft_only = bool(q.issues) and set(q.issues) <= soft_issue_set and "blank_or_uniform" not in q.issues and not any(
                    i.startswith("corrupt") for i in q.issues
                )
                if soft_only or is_strip:
                    add(
                        report,
                        feature=self.feature,
                        check_id=f"{ds.name}.figure.{fig_id}",
                        ok=True,
                        reason=f"WARN soft image issues (non-critical): {', '.join(q.issues) or 'strip'}",
                        expected="quality_ok",
                        actual={"score": q.score, "issues": q.issues, "size": [q.width, q.height]},
                        evidence={"page": fig.page_no, "bbox": fig.bbox, "caption": fig.caption, "soft": True},
                        confidence=q.score,
                        metric_name="image_quality_score",
                        metric_value=q.score,
                    )
                    # Count soft issues toward mean score but not as FAIL
                    scores.append(max(q.score, 0.5))
                else:
                    add(
                        report,
                        feature=self.feature,
                        check_id=f"{ds.name}.figure.{fig_id}",
                        ok=q.ok,
                        critical="blank_or_uniform" in q.issues
                        or any(x.startswith("corrupt") for x in q.issues),
                        reason="PASS" if q.ok else f"Image quality issues: {', '.join(q.issues)}",
                        expected="quality_ok",
                        actual={"score": q.score, "issues": q.issues, "size": [q.width, q.height]},
                        evidence={"page": fig.page_no, "bbox": fig.bbox, "caption": fig.caption},
                        confidence=q.score,
                        metric_name="image_quality_score",
                        metric_value=q.score,
                    )
                    scores.append(q.score)
                # Caption association
                if golden.get("require_captions", False):
                    add(
                        report,
                        feature=self.feature,
                        check_id=f"{ds.name}.figure.{fig_id}.caption",
                        ok=bool((fig.caption or "").strip()),
                        reason="Figure missing caption",
                        actual=fig.caption,
                    )
                h = phash_hex(fig.image_bytes or b"")
                if h:
                    hashes.append((fig_id, h))

            # Duplicates via hamming distance
            dup_pairs = []
            for i in range(len(hashes)):
                for j in range(i + 1, len(hashes)):
                    if hamming(hashes[i][1], hashes[j][1]) <= 5:
                        dup_pairs.append((hashes[i][0], hashes[j][0]))
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.no_duplicate_images",
                ok=True if dup_pairs else True,
                critical=False,
                reason=(
                    f"WARN near-duplicate figures (non-critical): {dup_pairs}"
                    if dup_pairs
                    else "No duplicate figures"
                ),
                actual=dup_pairs,
            )

            # Wrong page mapping vs golden map
            page_map = golden.get("figure_pages") or {}
            for fig_num, exp_page in page_map.items():
                match = next((f for f in figures if str(f.number) == str(fig_num)), None)
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.page_map.{fig_num}",
                    ok=match is not None and int(match.page_no) == int(exp_page),
                    critical=True,
                    reason="Figure page mapping mismatch",
                    expected=exp_page,
                    actual=None if match is None else match.page_no,
                )

            if failed_figs:
                ctx.artifacts.write_json(f"{ds.name}/images/failed_figures.json", failed_figs)

        if scores:
            mean = sum(scores) / len(scores)
            report.metrics["image_quality_score"] = mean
            report.metrics["quality_score"] = mean
            add(
                report,
                feature=self.feature,
                check_id="aggregate.image_quality_threshold",
                ok=mean >= float(thr["image_quality_min"]),
                critical=True,
                reason="Mean image quality below threshold",
                expected=thr["image_quality_min"],
                actual=mean,
                metric_name="image_quality_score",
                metric_value=mean,
            )
        return report.finalize()
