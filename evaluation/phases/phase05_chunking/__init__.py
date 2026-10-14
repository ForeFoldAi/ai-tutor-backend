"""Phase 05 — Chunking evaluation."""

from __future__ import annotations

from evaluation.core.adapters.document import chunks_to_serializable
from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.metrics.chunk_quality import corpus_chunk_quality
from evaluation.phases._helpers import add, ensure_chunks, new_report, timed


@register_phase
class ChunkingPhase(Phase):
    phase_id = "chunking"
    feature = "chunking"
    critical = True
    requires_heavy_ml = True  # process_document uses PdfExtractionPipeline when enabled
    depends_on = ["pdf_extraction"]

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        thr = ctx.config.data["thresholds"]
        for ds in ctx.datasets:
            try:
                chunks, ms = timed(lambda: ensure_chunks(ctx, ds))
            except Exception as exc:  # noqa: BLE001
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.chunking_runs",
                    ok=False,
                    error=True,
                    critical=True,
                    reason=str(exc),
                )
                report.failed_stage = "chunking"
                continue

            serial = chunks_to_serializable(chunks)
            report.artifacts.append(
                ctx.artifacts.write_json(f"{ds.name}/chunks/chunks.json", serial)
            )
            quality = corpus_chunk_quality(
                chunks,
                min_tokens=int(thr["chunk_size_tokens_min"]),
                max_tokens=int(thr["chunk_size_tokens_max"]),
            )
            ctx.artifacts.write_json(f"{ds.name}/chunks/quality.json", {
                k: v for k, v in quality.items() if k != "per_chunk"
            })

            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.non_empty",
                ok=quality["count"] > 0,
                critical=True,
                reason="No chunks produced",
                actual=quality["count"],
                execution_ms=ms,
            )
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.size_bounds",
                ok=quality["too_large"] == 0,
                reason="Chunks exceed max token size",
                expected={"max_tokens": thr["chunk_size_tokens_max"]},
                actual={"too_large": quality["too_large"]},
            )
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.quality_threshold",
                ok=quality["mean_score"] >= float(thr["chunk_quality_min"]),
                critical=True,
                reason="Chunk quality below threshold",
                expected=thr["chunk_quality_min"],
                actual=quality["mean_score"],
                metric_name="chunk_quality_score",
                metric_value=quality["mean_score"],
            )
            # Heading / section preservation
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.section_metadata",
                ok=quality["with_section_hint"] > 0,
                reason="No section_hint metadata on chunks",
                actual={"with_section_hint": quality["with_section_hint"], "count": quality["count"]},
            )
            # Integrity issues
            broken = sum(
                1
                for s in quality.get("per_chunk", [])
                if any(
                    i in s["issues"]
                    for i in ("broken_paragraph_hyphen", "broken_list", "possible_split_table")
                )
            )
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.paragraph_list_table_integrity",
                ok=broken == 0,
                reason="Broken paragraphs/lists/tables detected in chunks",
                actual={"broken_chunks": broken},
            )
            report.metrics["chunk_quality_score"] = quality["mean_score"]
            report.metrics["quality_score"] = quality["mean_score"]
            report.metrics["chunk_count"] = float(quality["count"])
        return report.finalize()
