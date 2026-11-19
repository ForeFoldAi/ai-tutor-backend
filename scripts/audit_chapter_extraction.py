#!/usr/bin/env python3
"""Run chapter detection over every textbook PDF and report one line per book.

  cd ai-tutor-backend && PYTHONPATH=. .venv/bin/python scripts/audit_chapter_extraction.py uploads [-v]

Columns: pages, chapters found (vs. files in a sibling chapters/ folder when present), numbering gaps,
longest chapter, titles that still carry table cells (page numbers, months), seconds.
Exits 1 when any book has a numbering gap, a dirty title, or a count that differs from chapters/.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.CRITICAL)

from app.services.textbook_structure import detect_textbook_structure  # noqa: E402
from app.services.textbook_structure.toc_detector import _is_month  # noqa: E402

DIRTY = re.compile(r"(?<![-–])\s\d{1,3}(?:-\d{1,3})?$|\.pdf$|_")  # 'Picture - 1' is a real title


def numbering_gaps(chapters: list[dict]) -> list[int]:
    nums = sorted({int(m.group()) for c in chapters if (m := re.match(r"\d+", str(c["chapter_number"])))})
    return sorted(set(range(nums[0], nums[-1] + 1)) - set(nums)) if nums else []


def audit(pdf: Path, verbose: bool) -> bool:
    t = time.time()
    res = detect_textbook_structure(str(pdf))
    secs = time.time() - t
    chs = res["chapters"]
    # Unit-level split files only ('ch_3_12.pdf', not reading files 'ch_3.A_13.pdf'); folders can be partial.
    expected = sum(1 for p in (pdf.parent / "chapters").glob("ch_*.pdf") if "." not in p.stem)
    gaps = numbering_gaps(chs)
    dirty = [c["chapter_title"] for c in chs if DIRTY.search(c["chapter_title"]) or any(_is_month(w) for w in c["chapter_title"].split()[-1:])]
    longest = max((c["end_pdf_page"] - c["start_pdf_page"] + 1 for c in chs), default=0)
    count_off = len(chs) < expected
    ok = not gaps and not dirty and not count_off
    print(
        f"{'OK ' if ok else 'BAD'} {str(pdf)[-58:]:58} pages={res['total_pages']:4} found={len(chs):3}"
        f"{f' exp={expected}' if expected else '':8} gaps={gaps or '-'} longest={longest:3} review={res['needs_manual_review']!s:5}"
        f" {secs:5.1f}s" + (f" dirty={dirty}" if dirty else "")
    )
    if verbose:
        for c in chs:
            print(f"      {c['chapter_number']:>5} | {c['chapter_title'][:70]:70} | {c['start_pdf_page'] + 1}-{c['end_pdf_page'] + 1}")
        for w in res["review_warnings"]:
            print(f"      ! {w}")
    return ok


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    root = Path(args[0] if args else "uploads")
    pdfs = sorted(p for p in root.rglob("*.pdf") if "chapters" not in p.parts)
    results = [audit(p, "-v" in sys.argv) for p in pdfs]
    print(f"\n{sum(results)}/{len(results)} books clean")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
