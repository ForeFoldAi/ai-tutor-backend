from unittest.mock import MagicMock

from app.services.image_service.textbook_image_retrieval import _load_uploads
from app.services.vector_service import resolve_chroma_upload_ids


def test_resolve_legacy_uuid_via_chapter_name():
    resolved = resolve_chroma_upload_ids(
        ["8"],
        ["Chapter 2 - Understanding the Weather"],
        collection_name="CBSE_CLASS_9_Social",
    )
    assert resolved == ["97ce23be-1c71-45f2-942c-34ea82cc205c"]


def test_resolve_passthrough_when_no_label_map(monkeypatch):
    monkeypatch.setattr(
        "app.services.vector_service._chroma_upload_id_by_content_label",
        lambda _name: {},
    )
    assert resolve_chroma_upload_ids(["8"], ["Chapter 2"], collection_name="X") == ["8"]


def test_load_uploads_rejects_uuid_chapter_ids():
    """Image DB lookup needs integer upload ids, not Chroma legacy UUIDs."""
    db = MagicMock()
    db.scalars.return_value.all.return_value = []
    assert _load_uploads(db, ["97ce23be-1c71-45f2-942c-34ea82cc205c"]) == {}
    db.scalars.assert_not_called()
