#!/usr/bin/env python3
"""
Run the production conversation scenario matrix and print a report.

Usage:
  cd ai-tutor-backend && .venv/bin/python scripts/run_conversation_scenario_matrix.py
  cd ai-tutor-backend && .venv/bin/python scripts/run_conversation_scenario_matrix.py --with-bge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.conversation_context import resolve_conversation_context  # noqa: E402

# Import scenario matrix from test module via importlib
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "conversation_scenarios",
    ROOT / "tests" / "test_conversation_scenarios.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
SCENARIOS = _mod.SCENARIOS


def main() -> int:
    parser = argparse.ArgumentParser(description="Conversation scenario matrix runner")
    parser.add_argument(
        "--with-bge",
        action="store_true",
        help="Warm up BGE embeddings before running (loads model if available)",
    )
    args = parser.parse_args()

    if args.with_bge:
        from app.services.vector_service import _get_embedding_model

        model = _get_embedding_model()
        print(f"BGE model loaded: {model is not None}")

    passed = 0
    failed = 0
    print(f"\n{'ID':<35} {'INTENT':<22} {'METHOD':<8} {'CONF':>5}  STATUS")
    print("-" * 85)

    for scenario in SCENARIOS:
        ctx = resolve_conversation_context(
            scenario["query"],
            conversation_history=scenario.get("history"),
        )
        ok = ctx.followup_type == scenario["followup"].value
        for token in scenario.get("retrieval_contains", []):
            if token.lower() not in ctx.retrieval_query.lower():
                ok = False
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(
            f"{scenario['id']:<35} {ctx.followup_type:<22} {ctx.intent_method:<8} "
            f"{ctx.intent_confidence:>5.2f}  {status}"
        )

    print("-" * 85)
    print(f"Total: {len(SCENARIOS)}  Passed: {passed}  Failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
