"""BGE intent classifier tests with mocked embeddings (no model download)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.conversation_intent_classifier import (
    FollowupType,
    _classify_by_bge,
    classify_followup_intent,
)


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(8).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def mock_bge_prototypes():
    """Deterministic prototype vectors so clarification scores highest for confused queries."""
    prototypes = {
        FollowupType.CLARIFICATION: [_unit_vec(1), _unit_vec(2)],
        FollowupType.CONTINUE_EXPLANATION: [_unit_vec(10)],
        FollowupType.NEW_TOPIC: [_unit_vec(20)],
    }
    with patch(
        "app.services.conversation_intent_classifier._load_intent_prototype_embeddings",
        return_value=prototypes,
    ):
        with patch(
            "app.services.vector_service.is_embedding_model_loaded",
            return_value=True,
        ):
            with patch(
                "app.services.image_service.figure_context_bge.embed_query",
                return_value=_unit_vec(1),
            ):
                yield


def test_bge_classifies_ambiguous_clarification(mock_bge_prototypes):
    result = _classify_by_bge(
        "that part was unclear",
        [{"role": "user", "content": "explain shadows"}],
    )
    assert result is not None
    assert result.followup_type == FollowupType.CLARIFICATION
    assert result.method == "bge"
    assert result.confidence > 0.0


def test_hybrid_prefers_bge_for_ambiguous_short_followup(mock_bge_prototypes):
    history = [
        {"role": "user", "content": "what are eclipses"},
        {"role": "assistant", "content": "An eclipse happens when the Moon blocks sunlight."},
    ]
    with patch(
        "app.services.conversation_intent_classifier.classify_followup_regex",
        return_value=FollowupType.NEW_TOPIC,
    ):
        result = classify_followup_intent("that bit", history)
    assert result.followup_type == FollowupType.CLARIFICATION
    assert result.method in ("bge", "hybrid")


def test_bge_returns_none_when_model_unavailable():
    with patch(
        "app.services.vector_service.is_embedding_model_loaded",
        return_value=False,
    ):
        assert _classify_by_bge("confused", []) is None
