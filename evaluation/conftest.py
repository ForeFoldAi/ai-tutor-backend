"""pytest configuration for the evaluation package."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "eval: AI evaluation check")
    config.addinivalue_line("markers", "critical: critical AI quality gate")
    config.addinivalue_line("markers", "heavy_ml: requires PDF ML pipeline / GPU weights")
    config.addinivalue_line("markers", "llm: requires LLM API key")
    config.addinivalue_line("markers", "pipeline: end-to-end pipeline")
    config.addinivalue_line("markers", "retrieval: retrieval quality")
    config.addinivalue_line("markers", "images: image extraction quality")


@pytest.fixture(scope="session")
def eval_config():
    from evaluation.core.config import EvalConfig

    return EvalConfig.load(preset="smoke")


@pytest.fixture(scope="session")
def eval_ctx(eval_config):
    from evaluation.core.context import build_context

    return build_context(eval_config)
