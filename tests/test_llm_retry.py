"""LLM client 429 retry helpers."""

import httpx

from app.services.llm_client import _retry_wait_sec


def test_retry_after_header():
    assert _retry_wait_sec(httpx.Response(429, headers={"Retry-After": "5"}), 0) == 5.0


def test_retry_backoff_without_header():
    w = _retry_wait_sec(httpx.Response(429), 0)
    assert 2.0 <= w < 3.0


def test_image_select_no_retry():
    from app.services.llm_client import _max_attempts

    assert _max_attempts("image_select") == 1
    assert _max_attempts("chat") > 1
