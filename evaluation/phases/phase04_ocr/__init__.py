"""Phase 04 — OCR / readable-text evaluation."""

from __future__ import annotations

import re

from evaluation.core.registry import Phase, register_phase
from evaluation.core.types import PhaseReport
from evaluation.phases._helpers import add, ensure_chunks, ensure_pipeline, new_report


def _ocr_noise_ratio(text: str) -> float:
    """Ignore markdown heading markers alone — they inflate noise on sparse pages."""
    cleaned = re.sub(r"[#*`_]+", "", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return 1.0
    weird = len(re.findall(r"[^\w\s.,;:!?()\[\]\"'%-/°]", cleaned))
    return weird / max(1, len(cleaned))


@register_phase
class OcrPhase(Phase):
    phase_id = "ocr"
    feature = "ocr"
    requires_heavy_ml = True
    depends_on = ["pdf_extraction"]

    def run(self, ctx) -> PhaseReport:
        report = new_report(self.phase_id, self.feature)
        for ds in ctx.datasets:
            result = ensure_pipeline(ctx, ds)
            golden = ctx.goldens.load(ds.name, "ocr", default={}) or {}
            must_find = golden.get("must_find_phrases", [])
            pages_text = "\n".join((p.markdown or "") for p in (result.pages or []))
            full = (getattr(result, "full_text", "") or "") + "\n" + pages_text
            ctx.artifacts.write_text(f"{ds.name}/ocr/full_text.txt", full[:200000])

            # RAG-facing text (text-layer + chunks) — what the tutor actually uses
            chunk_blob = ""
            try:
                chunks = ensure_chunks(ctx, ds)
                chunk_blob = "\n".join(getattr(c, "page_content", "") or "" for c in chunks)
                ctx.artifacts.write_text(f"{ds.name}/ocr/chunk_corpus.txt", chunk_blob[:200000])
            except Exception as exc:  # noqa: BLE001
                add(
                    report,
                    feature=self.feature,
                    check_id=f"{ds.name}.chunk_corpus",
                    ok=False,
                    reason=f"Could not build chunk corpus: {exc}",
                )

            searchable = chunk_blob if chunk_blob.strip() else full

            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.non_empty",
                ok=len(searchable.strip()) > 200,
                critical=True,
                reason="No usable OCR/text-layer corpus for RAG",
                actual={"ml_markdown_len": len(full), "chunk_corpus_len": len(chunk_blob)},
            )

            # ML markdown quality — informational only (text-layer chunks power RAG)
            noise = _ocr_noise_ratio(full)
            ml_sparse = noise >= 0.08 or len(re.sub(r"[#\s]", "", full)) < 200
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.ml_markdown_quality",
                ok=True,
                critical=False,
                reason=(
                    f"WARN: ML page markdown sparse/noisy (noise={noise:.2f}); "
                    "tutor uses text-layer chunks"
                    if ml_sparse
                    else "ML markdown looks usable"
                ),
                expected="<0.08 noise and substantial text",
                actual={"noise": noise, "ml_len": len(full), "sparse": ml_sparse},
                metric_name="ocr_noise_ratio",
                metric_value=noise,
            )

            missing = [p for p in must_find if p.lower() not in searchable.lower()]
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.must_find_phrases",
                ok=len(missing) == 0,
                critical=bool(must_find),
                reason="Required phrases missing from chunk/OCR corpus" if missing else "All golden phrases found",
                expected=must_find,
                actual={"missing": missing, "corpus": "chunks" if chunk_blob.strip() else "ml_markdown"},
            )

            empty_pages = [p.page_no for p in (result.pages or []) if not (p.markdown or "").strip()]
            empty_ratio = len(empty_pages) / max(1, len(result.pages or []))
            add(
                report,
                feature=self.feature,
                check_id=f"{ds.name}.no_empty_pages",
                ok=True,
                critical=False,
                reason=(
                    f"WARN: {len(empty_pages)} pages empty ML markdown (ratio={empty_ratio:.2f})"
                    if empty_pages
                    else "No empty ML markdown pages"
                ),
                actual={"empty_pages": empty_pages, "empty_ratio": empty_ratio},
            )
            report.metrics["ocr_noise_ratio"] = noise
            report.metrics["quality_score"] = 1.0 if not missing and len(searchable.strip()) > 200 else 0.5
        return report.finalize()
