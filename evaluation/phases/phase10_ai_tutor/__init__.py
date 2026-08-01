"""Phase 10 — AI Tutor response quality."""

from __future__ import annotations

from evaluation.core.adapters.retrieval import retrieve
from evaluation.core.adapters.tutor import ask_tutor
from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.metrics.llm_quality import llm_judge_answer, score_tutor_response
from evaluation.phases._helpers import add, ensure_embedded, new_report, timed


@register_phase
class AiTutorPhase(Phase):
    phase_id = "ai_tutor"
    feature = "ai_tutor"
    critical = True
    requires_llm = True
    requires_heavy_ml = True
    depends_on = ["retrieval"]

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        thr = ctx.config.data["thresholds"]
        judge_on = bool(ctx.config.get("llm_judge", "enabled", default=True))
        qualities = []
        halls = []

        for ds in ctx.datasets:
            ensure_embedded(ctx, ds)
            golden = ctx.goldens.load(ds.name, "ai_answers", default={}) or {}
            cases = golden.get("cases") or []
            if not cases:
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.golden_cases",
                    ok=False,
                    critical=True,
                    reason="No ai_answers golden cases",
                )
                continue

            for i, case in enumerate(cases):
                q = case["question"]
                # Context for groundedness
                try:
                    docs, _, _ = retrieve(
                        q,
                        collection_name=ds.collection_name,
                        chapter_ids=[ds.textbook_upload_id],
                    )
                    contexts = [getattr(d, "page_content", "") or "" for d in (docs or [])]
                except Exception:
                    contexts = []

                try:
                    out, ms = timed(
                        lambda: ask_tutor(
                            q,
                            collection_name=ds.collection_name,
                            chapter_ids=[ds.textbook_upload_id],
                            board=ds.board,
                            class_level=ds.class_level,
                            subject_name=ds.subject_name,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    add(
                        report,
                        feature=self.feature,
                        check_id=f"{ds.name}.case{i}.tutor_call",
                        ok=False,
                        error=True,
                        critical=True,
                        reason=str(exc),
                    )
                    continue

                answer = out.get("answer") or ""
                scored = score_tutor_response(
                    answer,
                    contexts=contexts,
                    expected_keywords=case.get("expected_keywords") or [],
                    must_include=case.get("must_include") or [],
                    forbidden=case.get("forbidden") or [],
                )
                qualities.append(scored["tutor_quality_score"])
                halls.append(scored["hallucination_rate"])

                judge = llm_judge_answer(q, answer, contexts, enabled=judge_on)
                payload = {
                    "question": q,
                    "answer": answer,
                    "images": out.get("images"),
                    "metrics": scored,
                    "judge": judge,
                    "latency_ms": ms,
                }
                ctx.artifacts.write_json(f"{ds.name}/ai_tutor/case_{i}.json", payload)

                g_min = float(thr["groundedness_min"])
                g_score = scored["groundedness"]
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.case{i}.groundedness",
                    ok=g_score >= g_min or g_score >= 0.45,
                    critical=g_score < 0.4,
                    reason=(
                        "Answer not grounded in retrieved context"
                        if g_score < g_min
                        else "Groundedness OK"
                    ),
                    expected=g_min,
                    actual=g_score,
                    execution_ms=ms,
                )
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.case{i}.faithfulness",
                    ok=scored["faithfulness"] >= float(thr["faithfulness_min"]),
                    reason="Faithfulness below threshold",
                    expected=thr["faithfulness_min"],
                    actual=scored["faithfulness"],
                )
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.case{i}.hallucination",
                    ok=not scored["hallucination_flags"],
                    critical=True,
                    reason="Potential hallucination flags",
                    actual=scored["hallucination_flags"],
                )
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.case{i}.keywords",
                    ok=scored["curriculum_alignment"] >= 0.5,
                    reason="Curriculum keywords missing from answer",
                    expected=case.get("expected_keywords"),
                    actual=scored["curriculum_alignment"],
                )
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.case{i}.safety_length",
                    ok=scored["safety_ok"] and scored["length_ok"],
                    critical=not scored["safety_ok"],
                    reason="Safety or length check failed",
                    actual={"safety_ok": scored["safety_ok"], "length_ok": scored["length_ok"]},
                )
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.case{i}.quality",
                    ok=scored["tutor_quality_score"] >= float(thr["tutor_quality_min"]),
                    critical=True,
                    reason="Tutor quality score below threshold",
                    expected=thr["tutor_quality_min"],
                    actual=scored["tutor_quality_score"],
                    metric_name="tutor_quality_score",
                    metric_value=scored["tutor_quality_score"],
                )
                # Optional contains_any for golden answers
                contains = case.get("answer_contains_any") or []
                if contains:
                    ok = any(x.lower() in answer.lower() for x in contains)
                    add(
                        report,
                        feature=self.feature,
                        check_id=f"{ds.name}.case{i}.answer_contains",
                        ok=ok,
                        critical=True,
                        reason="Answer missing required content",
                        expected=contains,
                        actual=answer[:500],
                    )

        if qualities:
            report.metrics["tutor_quality_score"] = sum(qualities) / len(qualities)
            report.metrics["hallucination_rate"] = sum(halls) / len(halls)
            report.metrics["quality_score"] = report.metrics["tutor_quality_score"]
            add(
                report,
                feature=self.feature,
                check_id="aggregate.hallucination_rate",
                ok=report.metrics["hallucination_rate"] <= float(thr["hallucination_rate_max"]),
                critical=True,
                reason="Hallucination rate exceeds threshold",
                expected=f"<={thr['hallucination_rate_max']}",
                actual=report.metrics["hallucination_rate"],
            )
        return report.finalize()
