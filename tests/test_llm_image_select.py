"""Unit checks for LLM chapter-image selection (fail-closed)."""

import asyncio
from unittest.mock import AsyncMock, patch

from app.services.image_service.llm_image_select import (
    apply_select_result,
    _parse_select_json,
    select_images_for_qa,
)


_CANDS = [
    {"figure_number": "2.3", "caption": "Political map", "page": 14, "url": "/a"},
    {"figure_number": "2.12", "caption": "Later map", "page": 22, "url": "/b"},
    {"figure_number": "1.1", "caption": "Wildlife", "page": 3, "url": "/c"},
]


def test_parse_select_json_ok():
    ids, conf = _parse_select_json('{"ids":["0","2"],"confidence":0.9}')
    assert ids == ["0", "2"]
    assert conf == 0.9


def test_parse_select_json_garbage():
    assert _parse_select_json("not json") is None
    assert _parse_select_json("") is None


def test_apply_select_result_maps_and_drops_bad():
    out = apply_select_result(_CANDS, ["0", "9", "0", "1", "x"], max_images=5)
    assert [x["figure_number"] for x in out] == ["2.3", "2.12"]


def test_apply_select_result_respects_max():
    out = apply_select_result(_CANDS, ["0", "1", "2"], max_images=1)
    assert len(out) == 1
    assert out[0]["figure_number"] == "2.3"


def test_select_images_low_confidence_returns_empty():
    async def run():
        with patch(
            "app.services.llm_client.complete",
            AsyncMock(return_value='{"ids":["0"],"confidence":0.2}'),
        ):
            return await select_images_for_qa(
                "maps?",
                "India's political map changed.",
                _CANDS,
                min_confidence=0.65,
            )

    assert asyncio.run(run()) == []


def test_select_images_maps_ids():
    async def run():
        with patch(
            "app.services.llm_client.complete",
            AsyncMock(return_value='{"ids":["0","1"],"confidence":0.9}'),
        ):
            return await select_images_for_qa(
                "maps?",
                "Look at the political maps.",
                _CANDS,
                min_confidence=0.65,
            )

    out = asyncio.run(run())
    assert [x["figure_number"] for x in out] == ["2.3", "2.12"]


def test_select_images_parse_fail_returns_empty():
    async def run():
        with patch(
            "app.services.llm_client.complete",
            AsyncMock(return_value="sorry no json"),
        ):
            return await select_images_for_qa("q", "a", _CANDS)

    assert asyncio.run(run()) == []
