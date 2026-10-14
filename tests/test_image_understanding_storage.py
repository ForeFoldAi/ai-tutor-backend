"""Ownership and path-traversal guards for ephemeral tutor images."""

from __future__ import annotations

import pytest
from PIL import Image
from io import BytesIO

from app.modules.image_understanding.errors import ImageNotFoundError
from app.modules.image_understanding.service import (
    delete_image,
    load_image_for_user,
    store_validated_image,
)
from app.modules.image_understanding.validators import validate_image_bytes


def _png() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (40, 40), "red").save(buf, format="PNG")
    return buf.getvalue()


def test_cross_user_image_id_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.modules.image_understanding.service.IMAGE_TEMP_DIR",
        str(tmp_path),
    )
    monkeypatch.setattr(
        "app.modules.image_understanding.service._sync_redis",
        lambda: None,
    )
    validated = validate_image_bytes(_png(), filename="a.png")
    meta = store_validated_image(validated, user_id=1)
    with pytest.raises(ImageNotFoundError):
        load_image_for_user(meta.image_id, user_id=2)
    data, loaded = load_image_for_user(meta.image_id, user_id=1)
    assert loaded.user_id == 1
    assert len(data) > 0
    delete_image(meta.image_id)


def test_path_traversal_image_id_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.modules.image_understanding.service.IMAGE_TEMP_DIR",
        str(tmp_path),
    )
    monkeypatch.setattr(
        "app.modules.image_understanding.service._sync_redis",
        lambda: None,
    )
    with pytest.raises(ImageNotFoundError):
        load_image_for_user("../etc/passwd", user_id=1)
    with pytest.raises(ImageNotFoundError):
        load_image_for_user("not-hex!", user_id=1)


def test_redis_roundtrip(monkeypatch):
    """Simulate multi-worker: store via Redis, load without local disk."""
    store: dict[str, bytes | str] = {}

    class FakeRedis:
        def pipeline(self):
            return self

        def setex(self, key, _ttl, value):
            store[key] = value
            return self

        def execute(self):
            return []

        def get(self, key):
            return store.get(key)

        def delete(self, *keys):
            for k in keys:
                store.pop(k, None)

        def ping(self):
            return True

        def close(self):
            pass

    monkeypatch.setattr(
        "app.modules.image_understanding.service._sync_redis",
        lambda: FakeRedis(),
    )
    monkeypatch.setattr(
        "app.modules.image_understanding.service.IMAGE_TEMP_DIR",
        "/tmp/tutor_img_test_unused",
    )
    validated = validate_image_bytes(_png(), filename="a.png")
    meta = store_validated_image(validated, user_id=42)
    # Pretend this worker has no local file (other instance)
    monkeypatch.setattr(
        "app.modules.image_understanding.service._load_from_disk",
        lambda _id: (_ for _ in ()).throw(
            __import__(
                "app.modules.image_understanding.errors", fromlist=["ImageNotFoundError"]
            ).ImageNotFoundError()
        ),
    )
    data, loaded = load_image_for_user(meta.image_id, user_id=42)
    assert loaded.user_id == 42
    assert len(data) > 0
    delete_image(meta.image_id)

