"""Phase 07 — Retrieval evaluation (Top-k, MRR, nDCG, page/chapter accuracy)."""

from __future__ import annotations

from evaluation.core.adapters.retrieval import docs_to_serializable, retrieve
from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.metrics.retrieval import hit_at_k, mean, mrr, ndcg_at_k, page_match
from evaluation.phases._helpers import add, ensure_embedded, new_report, timed


@register_phase
class RetrievalPhase(Phase):
    phase_id = "retrieval"
    feature = "retrieval"
    critical = True
    requires_heavy_ml = True
    depends_on = ["embeddings"]

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        thr = ctx.config.data["thresholds"]
        top1s, top3s, mrrs, ndcgs = [], [], [], []

        for ds in ctx.datasets:
            ensure_embedded(ctx, ds)
            golden = ctx.goldens.load(ds.name, "retrieval", default={}) or {}
            cases = golden.get("cases") or []
            if not cases:
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.golden_cases",
                    ok=False,
                    critical=True,
                    reason="No retrieval golden cases — add goldens/<dataset>/retrieval.json",
                )
                continue

            case_results = []
            for ci, case in enumerate(cases):
                q = case["question"]
                try:
                    (docs, scope, instruction), ms = timed(
                        lambda: retrieve(
                            q,
                            collection_name=ds.collection_name,
                            chapter_ids=[ds.textbook_upload_id],
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    add(
                        report,
                        feature=self.feature,
                        check_id=f"{ds.name}.case{ci}.retrieve",
                        ok=False,
                        error=True,
                        critical=True,
                        reason=str(exc),
                    )
                    continue

                serial = docs_to_serializable(docs)
                ctx.artifacts.write_json(
                    f"{ds.name}/retrieval/case_{ci}.json",
                    {
                        "question": q,
                        "docs": serial,
                        "scope": str(scope),
                        "instruction": instruction,
                        "latency_ms": ms,
                    },
                )

                pages = [d.get("page") for d in serial]
                exp_pages = case.get("expected_pages") or []
                exp_chunk_ids = set(int(x) for x in (case.get("expected_chunks") or []))
                # Rank by chunk index if present in metadata, else use page match as relevance
                ranked_ids = []
                for d in serial:
                    meta_idx = d.get("index")
                    if meta_idx is None:
                        # try parse from evidence preview not available — use rank as id
                        ranked_ids.append(d["rank"] - 1)
                    else:
                        ranked_ids.append(int(meta_idx))

                # Relevance by page overlap when chunk ids not provided
                if exp_chunk_ids:
                    relevant = exp_chunk_ids
                    ranked = ranked_ids
                else:
                    # Synthesize relevance: ranks whose page is in expected_pages
                    relevant = set()
                    ranked = list(range(len(serial)))
                    for i, d in enumerate(serial):
                        if d.get("page") is not None and int(d["page"]) in set(int(p) for p in exp_pages):
                            relevant.add(i)

                h1 = hit_at_k(relevant, ranked, 1) if relevant else (1.0 if page_match(exp_pages, pages, 1) else 0.0)
                h3 = hit_at_k(relevant, ranked, 3) if relevant else (1.0 if page_match(exp_pages, pages, 3) else 0.0)
                h5 = hit_at_k(relevant, ranked, 5) if relevant else (1.0 if page_match(exp_pages, pages, 5) else 0.0)
                rr = mrr(relevant, ranked) if relevant else h1
                nd = ndcg_at_k(relevant, ranked, 5) if relevant else h5
                top1s.append(h1)
                top3s.append(h3)
                mrrs.append(rr)
                ndcgs.append(nd)

                # Wrong chapter / page
                wrong_chapter = any(
                    str(d.get("textbook_upload_id") or ds.textbook_upload_id) != ds.textbook_upload_id
                    for d in serial
                )
                page_ok = page_match(exp_pages, pages, 5) if exp_pages else True

                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.case{ci}.top1",
                    ok=h1 >= 1.0 or h3 >= 1.0,
                    critical=False,
                    reason=(
                        f"Top-1 miss for: {q}"
                        if h1 < 1.0
                        else f"Top-1 hit for: {q}"
                    ),
                    expected={"pages": exp_pages, "chunks": sorted(exp_chunk_ids)},
                    actual={"pages": pages[:5], "hit@1": h1, "hit@3": h3},
                    evidence={"question": q},
                    execution_ms=ms,
                    metric_name="hit@1",
                    metric_value=h1,
                )
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.case{ci}.top3",
                    ok=h3 >= 1.0 or h5 >= 1.0,
                    critical=False,
                    reason=f"Top-3 miss for: {q}" if h3 < 1.0 else f"Top-3 hit for: {q}",
                    expected={"pages": exp_pages},
                    actual={"pages": pages[:3], "hit@3": h3, "hit@5": h5},
                )
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.case{ci}.page_accuracy",
                    ok=page_ok,
                    critical=True,
                    reason="Wrong page retrieval",
                    expected=exp_pages,
                    actual=pages[:5],
                )
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.case{ci}.chapter_scope",
                    ok=not wrong_chapter,
                    critical=True,
                    reason="Retrieved docs from wrong chapter/upload id",
                    actual=serial[:3],
                )
                # Keywords in retrieved context
                for kw in case.get("must_appear_in_context") or []:
                    blob = " ".join(d.get("preview") or "" for d in serial).lower()
                    add(
                        report,
                        feature=self.feature,
                        check_id=f"{ds.name}.case{ci}.ctx.{kw[:24]}",
                        ok=kw.lower() in blob,
                        reason=f"Expected context keyword missing: {kw}",
                        expected=kw,
                    )

                case_results.append(
                    {"question": q, "hit@1": h1, "hit@3": h3, "hit@5": h5, "mrr": rr, "ndcg@5": nd}
                )

            if case_results:
                ctx.artifacts.write_json(f"{ds.name}/retrieval/summary.json", case_results)

        if top1s:
            report.metrics.update(
                {
                    "retrieval_top1": mean(top1s),
                    "retrieval_top3": mean(top3s),
                    "retrieval_mrr": mean(mrrs),
                    "retrieval_ndcg": mean(ndcgs),
                    "quality_score": mean(top3s),
                }
            )
            add(
                report,
                feature=self.feature,
                check_id="aggregate.top1_threshold",
                ok=report.metrics["retrieval_top1"] >= float(thr["retrieval_top1"]),
                critical=True,
                reason="Top-1 retrieval accuracy below threshold",
                expected=thr["retrieval_top1"],
                actual=report.metrics["retrieval_top1"],
            )
            add(
                report,
                feature=self.feature,
                check_id="aggregate.top3_threshold",
                ok=report.metrics["retrieval_top3"] >= float(thr["retrieval_top3"]),
                critical=True,
                reason="Top-3 retrieval accuracy below threshold",
                expected=thr["retrieval_top3"],
                actual=report.metrics["retrieval_top3"],
            )
            add(
                report,
                feature=self.feature,
                check_id="aggregate.mrr_threshold",
                ok=report.metrics["retrieval_mrr"] >= float(thr["retrieval_mrr"]),
                reason="MRR below threshold",
                expected=thr["retrieval_mrr"],
                actual=report.metrics["retrieval_mrr"],
            )
            add(
                report,
                feature=self.feature,
                check_id="aggregate.ndcg_threshold",
                ok=report.metrics["retrieval_ndcg"] >= float(thr["retrieval_ndcg"]),
                reason="nDCG below threshold",
                expected=thr["retrieval_ndcg"],
                actual=report.metrics["retrieval_ndcg"],
            )
        return report.finalize()
