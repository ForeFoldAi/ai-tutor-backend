"""Shared helpers for phase modules."""

from __future__ import annotations

import time
from typing import Callable

from evaluation.core.context import DatasetBundle, EvalContext
from evaluation.core.types import CheckResult, PhaseReport, Status, check


def new_report(phase_id: str, feature: str, dataset: str | None = None) -> PhaseReport:
    return PhaseReport(
        phase_id=phase_id,
        feature=feature,
        status=Status.PASS,
        dataset=dataset,
    )


def timed(fn: Callable):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000


def ensure_pipeline(ctx: EvalContext, ds: DatasetBundle):
    """Cache PipelineResult per dataset for downstream phases."""
    key = f"pipeline:{ds.name}"

    def _load():
        from evaluation.core.adapters.pdf import extract_pdf

        return extract_pdf(ds.pdf_path, use_cache=True)

    return ctx.get_cached(key, _load)


def ensure_chunks(ctx: EvalContext, ds: DatasetBundle):
    key = f"chunks:{ds.name}"

    def _load():
        from evaluation.core.adapters.document import chunk_document

        return chunk_document(
            ds.pdf_path,
            board=ds.board,
            class_level=ds.class_level,
            subject_name=ds.subject_name,
            textbook_upload_id=ds.textbook_upload_id,
        )

    return ctx.get_cached(key, _load)


def ensure_embedded(ctx: EvalContext, ds: DatasetBundle) -> int:
    key = f"embedded:{ds.name}"

    def _load():
        from evaluation.core.adapters.retrieval import embed_and_store

        chunks = ensure_chunks(ctx, ds)
        return embed_and_store(chunks, collection_name=ds.collection_name)

    return ctx.get_cached(key, _load)


def add(
    report: PhaseReport,
    *,
    feature: str,
    check_id: str,
    ok: bool,
    reason: str,
    **kwargs,
) -> CheckResult:
    c = check(feature, check_id, ok=ok, reason=reason, **kwargs)
    report.checks.append(c)
    return c
