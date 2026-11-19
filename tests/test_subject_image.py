"""Subject image pool: bulk upload (bad files skipped), pick for a subject, serve, delete blocked while in use."""

import asyncio
from datetime import datetime
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image

from app.modules.catalog import router as catalog_router
from app.modules.catalog.models import BoardEnum, ClassEnum, SubjectImage, SyllabusSubject
from app.modules.catalog.schemas import SyllabusUpdateRequest
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
    """Just enough Session for the catalog handlers: get/add/delete/commit and scalar(s) by model."""

    def __init__(self, *rows):
        self.rows = list(rows)
        self._next_id = 100

    def get(self, model, id_):
        return next((r for r in self.rows if isinstance(r, model) and r.id == id_), None)

    def add(self, row):
        self.rows.append(row)

    def delete(self, row):
        self.rows.remove(row)

    def commit(self):
        for r in self.rows:
            if getattr(r, "id", None) is None:
                r.id = self._next_id
                self._next_id += 1
            if isinstance(r, SubjectImage) and r.created_at is None:
                r.created_at = datetime.now()

    def refresh(self, _row):
        pass

    def scalars(self, stmt):
        model = stmt.column_descriptions[0]["entity"]
        rows = [r for r in self.rows if isinstance(r, model)]
        crit = str(stmt.whereclause) if stmt.whereclause is not None else ""
        if "image_id IS NOT NULL" in crit:
            rows = [r for r in rows if r.image_id is not None]
        elif "image_id =" in crit:
            target = stmt.whereclause.right.value
            rows = [r for r in rows if r.image_id == target]
        return rows

    def scalar(self, stmt):
        path = stmt.whereclause.right.value
        return next((r.id for r in self.rows if isinstance(r, SubjectImage) and r.image_path == path), None)


def _png(color) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (40, 30), color).save(buf, format="PNG")
    return buf.getvalue()


def _upload(name: str, data: bytes):
    async def read(_n=-1):
        return data

    return SimpleNamespace(filename=name, read=read)


def test_pool_upload_pick_serve_delete(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(storage_backend, "get_storage_backend", lambda: store)
    subject = SyllabusSubject(id=7, board=BoardEnum.CBSE, class_level=ClassEnum.CLASS_8, subject_name="Science")
    subject.created_at = subject.updated_at = datetime.now()
    db = FakeDb(subject)

    res = asyncio.run(
        catalog_router.upload_subject_images(
            db, None, [_upload("ok.png", _png("red")), _upload("bad.png", b"<script>not an image")]
        )
    )
    assert len(res.created) == 1 and [s.file_name for s in res.skipped] == ["bad.png"]
    img = res.created[0]
    assert img.image_url == f"/auth/catalog/subject-images/{img.id}"

    out = catalog_router.update_syllabus(7, SyllabusUpdateRequest(image_id=img.id), db, None)
    assert out.image_id == img.id and out.image_url == img.image_url

    served = catalog_router.get_subject_image(img.id, db, None)
    assert served.body in store.objects.values()

    with pytest.raises(HTTPException) as exc:
        catalog_router.delete_subject_image(img.id, db, None)
    assert exc.value.status_code == 409 and "CBSE Class 8 Science" in exc.value.detail

    catalog_router.update_syllabus(7, SyllabusUpdateRequest(image_id=None), db, None)
    assert subject.image_id is None

    catalog_router.delete_subject_image(img.id, db, None)
    assert db.get(SubjectImage, img.id) is None and store.objects == {}
