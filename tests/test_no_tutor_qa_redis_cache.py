"""Assert tutor Q&A is not cached in Redis (plan: no tutor answers in Redis)."""

from __future__ import annotations

import ast
from pathlib import Path


def test_cache_module_has_no_tutor_qa_helpers():
    import app.core.cache as cache

    assert hasattr(cache, "get_redis")
    for name in (
        "get_cached_answer",
        "set_cached_answer",
        "deserialize_tutor_cache",
        "serialize_tutor_cache",
        "invalidate_collection",
    ):
        assert not hasattr(cache, name), name


def test_chat_service_does_not_import_tutor_qa_cache():
    path = Path(__file__).resolve().parents[1] / "app" / "services" / "chat_service.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "app.core.cache":
            names = {a.name for a in node.names}
            assert not names & {
                "get_cached_answer",
                "set_cached_answer",
                "deserialize_tutor_cache",
            }, names


def test_config_has_no_tutor_answer_cache_flag():
    import app.config as cfg

    assert not hasattr(cfg, "TUTOR_ANSWER_CACHE_ENABLED")
