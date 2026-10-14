"""Unit checks for Qdrant ensure_collection error classification."""

from app.services.vector_backend.qdrant_common import (
    _is_already_exists,
    _is_connectivity,
    _is_not_found,
)


def test_not_found():
    assert _is_not_found(Exception("Collection `foo` doesn't exist!"))
    assert _is_not_found(Exception("Not found: collection"))


def test_connectivity():
    assert _is_connectivity(Exception("timed out"))
    assert _is_connectivity(Exception("ConnectTimeout"))
    assert not _is_connectivity(Exception("Not found: collection"))


def test_already_exists():
    assert _is_already_exists(Exception("Wrong input: collection already exists"))
