"""Subject cover image: upload stores a re-encoded image, re-upload replaces and deletes the old object."""

import asyncio
from datetime import datetime
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image

from app.modules.catalog import router as catalog_router
from app.modules.catalog.models import BoardEnum, ClassEnum, SyllabusSubject
from app.services.image_service import storage_backend


class FakeStore:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def save(self, key, data):
        self.objects[key] = data
        return key, ""

    def delete(self, key):
        self.objects.pop(key, None)

    def get_bytes(self, key):
        return self.objects.get(key)


class FakeDb:
    def __init__(self, row):
        self.row = row

    def get(self, _model, _id):
        return self.row

    def commit(self):
        pass

    def refresh(self, _row):
        pass


def _png(color) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (40, 30), color).save(buf, format="PNG")
    return buf.getvalue()


def _upload(data: bytes):
    async def read(_n=-1):
        return data

    return SimpleNamespace(read=read)


def test_upload_replace_and_serve(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(storage_backend, "get_storage_backend", lambda: store)
    row = SyllabusSubject(id=7, board=BoardEnum.CBSE, class_level=ClassEnum.CLASS_8, subject_name="Science")
    row.created_at = row.updated_at = datetime.now()
    db = FakeDb(row)

    res = asyncio.run(catalog_router.upload_syllabus_image(7, db, None, _upload(_png("red"))))
    first = row.image_path
    assert first.startswith("subject_images/7/") and first in store.objects
    assert res.image_url.startswith("/auth/catalog/subject-images/7?v=")

    asyncio.run(catalog_router.upload_syllabus_image(7, db, None, _upload(_png("blue"))))
    assert row.image_path != first and first not in store.objects and row.image_path in store.objects

    served = catalog_router.get_subject_image(7, db, None)
    assert served.body == store.objects[row.image_path]

    with pytest.raises(HTTPException) as exc:
        asyncio.run(catalog_router.upload_syllabus_image(7, db, None, _upload(b"<script>not an image")))
    assert exc.value.status_code == 400
