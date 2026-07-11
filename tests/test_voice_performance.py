"""Phase 6 — voice performance helpers."""

from unittest.mock import MagicMock, patch

from app.services.voice_chunking import VoicePipelineTiming
from app.services.voice_performance import (
    export_voice_turn_metrics,
    prewarm_rag_sync,
    warm_embeddings_sync,
)


def test_export_voice_turn_metrics_json(caplog):
    timing = VoicePipelineTiming(turn_id="t1")
    timing.mark_llm_first_token()
    with caplog.at_level("INFO"):
        export_voice_turn_metrics(timing, phase="turn_end", channel="ws")
    assert any("voice_metrics" in r.message for r in caplog.records)


def test_prewarm_rag_no_collection():
    assert prewarm_rag_sync("", None) is False


@patch("app.services.voice_performance.VOICE_EMBEDDING_WARMUP", True)
@patch("app.services.vector_service.is_embedding_model_loaded", return_value=True)
def test_warm_embeddings_already_loaded(_mock_loaded):
    assert warm_embeddings_sync() is True


def test_http_tts_prefetcher_synthesize():
    from app.services.voice_http_tts import http_tts_prefetcher

    prefetcher = http_tts_prefetcher()
    prefetcher._voice = "en-IN-NeerjaNeural"  # noqa: SLF001 — test shortcut
    with patch(
        "app.services.voice_http_tts.synthesize_mp3",
        return_value=b"\xff\xfb",
    ) as mock_syn:
        import asyncio

        data = asyncio.run(prefetcher.synthesize("hello"))
    assert data == b"\xff\xfb"
    mock_syn.assert_called_once()
