"""Time each step of single-chapter processing on this machine (no DB / vector store writes).

Usage: .venv/bin/python scripts/profile_chapter.py /path/to/chapter.pdf
Run the same file locally and inside the production worker container to compare.
"""

from __future__ import annotations

import os
import sys
import time


def step(label: str, t0: float) -> float:
    now = time.perf_counter()
    print(f"{label:<34} {now - t0:7.1f}s", flush=True)
    return now


def main() -> None:
    pdf = sys.argv[1]
    import torch

    print(f"cpu_count={os.cpu_count()} torch_threads={torch.get_num_threads()} "
          f"cuda={torch.cuda.is_available()} mps={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
    import fitz

    print(f"pages={fitz.open(pdf).page_count} size={os.path.getsize(pdf) / 1e6:.1f} MB")

    t = time.perf_counter()
    from app.services.pdf_extract_pipeline.pipeline import get_pipeline

    pipe = get_pipeline()
    _ = pipe.layout  # force model load so it's timed separately
    t = step("load layout model", t)

    result = pipe.process_pdf(pdf)
    t = step(f"ML pipeline ({len(result.pages)} pages)", t)

    from app.services.document_service import process_document

    chunks = process_document(pdf, extra_metadata={})  # reuses the cached pipeline result
    t = step(f"chunking ({len(chunks)} chunks)", t)

    from app.services.vector_service import _get_embedding_model

    model = _get_embedding_model()
    t = step("load embedding model", t)
    model.embed_documents([c.page_content for c in chunks])
    step("embed chunks", t)


if __name__ == "__main__":
    main()
