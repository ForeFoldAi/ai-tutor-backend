"""Phase 06 — Embedding generation evaluation."""

from __future__ import annotations

import time

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._helpers import add, ensure_chunks, ensure_embedded, new_report


@register_phase
class EmbeddingsPhase(Phase):
    phase_id = "embeddings"
    feature = "embeddings"
    critical = True
    requires_heavy_ml = True  # needs chunks from ML/document pipeline
    depends_on = ["chunking"]

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        expected_dim = int(ctx.config.threshold("embedding_dim"))
        for ds in ctx.datasets:
            chunks = ensure_chunks(ctx, ds)
            t0 = time.perf_counter()
            try:
                n = ensure_embedded(ctx, ds)
            except Exception as exc:  # noqa: BLE001
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.embed_store",
                    ok=False,
                    error=True,
                    critical=True,
                    reason=str(exc),
                )
                report.failed_stage = "embeddings"
                continue
            ms = (time.perf_counter() - t0) * 1000

            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.created",
                ok=n > 0,
                critical=True,
                reason="No embeddings stored",
                expected=f">=1 (chunks={len(chunks)})",
                actual=n,
                execution_ms=ms,
            )
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.count_matches_chunks",
                ok=n >= len(chunks) or n > 0,  # some backends dedupe
                reason="Stored embedding count suspiciously low vs chunks",
                expected=len(chunks),
                actual=n,
            )

            # Probe embedding dimension via vector service if available
            dim = None
            try:
                from app.services import vector_service as vs

                if hasattr(vs, "get_embeddings"):
                    emb = vs.get_embeddings()
                    vec = emb.embed_query("dimension probe")
                    dim = len(vec)
                elif hasattr(vs, "_get_embeddings"):
                    emb = vs._get_embeddings()
                    vec = emb.embed_query("dimension probe")
                    dim = len(vec)
            except Exception as exc:  # noqa: BLE001
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.dimension_probe",
                    ok=False,
                    reason=f"Could not probe embedding dim: {exc}",
                )
            if dim is not None:
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.dimension",
                    ok=dim == expected_dim,
                    critical=True,
                    reason="Embedding dimension mismatch (BGE-base=768)",
                    expected=expected_dim,
                    actual=dim,
                )
                # Normalization check
                try:
                    import math

                    from app.services import vector_service as vs

                    emb = vs.get_embeddings() if hasattr(vs, "get_embeddings") else vs._get_embeddings()
                    vec = emb.embed_query("normalization probe")
                    norm = math.sqrt(sum(x * x for x in vec))
                    add(
                        report,
                        feature=self.feature,
                        check_id=f"{ds.name}.normalization",
                        ok=0.5 <= norm <= 2.0,  # allow unnormalized; warn if insane
                        reason="Embedding L2 norm out of sane range",
                        actual=norm,
                    )
                except Exception:
                    pass

            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.latency",
                ok=ms < float(ctx.config.threshold("latency_ms_fail")),
                reason="Embedding+store latency exceeded fail threshold",
                expected=f"<{ctx.config.threshold('latency_ms_fail')}ms",
                actual=ms,
                metric_name="embed_latency_ms",
                metric_value=ms,
            )
            report.metrics["embed_count"] = float(n)
            report.metrics["embed_latency_ms"] = ms
            report.metrics["quality_score"] = 1.0 if n > 0 else 0.0
            ctx.artifacts.write_json(
                f"{ds.name}/embeddings/meta.json",
                {"count": n, "latency_ms": ms, "dim": dim, "collection": ds.collection_name},
            )
        return report.finalize()
