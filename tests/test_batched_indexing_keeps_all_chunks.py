"""Batched indexing of one upload must keep every batch.

Regression: add_documents deleted all of the upload's vectors before each upsert,
so a chapter indexed in 50-chunk batches kept only its last batch (the exercises
and teacher transcripts at the end of the PDF), and the tutor explained those
instead of the lesson.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.services.vector_backend import qdrant_text


class _FakeQdrant:
    def __init__(self):
        self.points: dict = {}
        self.deletes = 0

    def delete(self, collection_name, points_selector):
        self.deletes += 1
        self.points.clear()

    def upsert(self, collection_name, points):
        for p in points:
            self.points[p.id] = p


def _docs(start: int, n: int) -> list[Document]:
    return [
        Document(page_content=f"chunk {i}", metadata={"textbook_upload_id": "752", "page": i})
        for i in range(start, start + n)
    ]


def _run(batches: list[list[Document]], **kw) -> _FakeQdrant:
    fake = _FakeQdrant()
    with patch.object(qdrant_text, "_embed_docs", side_effect=lambda d: [[0.0] * 4 for _ in d]), \
         patch.object(qdrant_text, "ensure_collection", MagicMock()), \
         patch.object(qdrant_text, "get_qdrant_client", return_value=fake), \
         patch.object(qdrant_text, "invalidate_collection_cache", MagicMock()):
        for b in batches:
            qdrant_text.add_documents(b, collection_name="c", **kw)
    return fake


def test_batches_without_replace_keep_every_chunk():
    fake = _run([_docs(0, 50), _docs(50, 11)], replace_existing=False)
    assert len(fake.points) == 61
    assert fake.deletes == 0


def test_default_replace_still_dedupes_single_shot_reindex():
    fake = _run([_docs(0, 5), _docs(0, 5)])
    assert len(fake.points) == 5
    assert fake.deletes == 2


def test_upload_handler_does_not_replace_per_batch():
    import inspect

    from app.services.catalog_pipeline.handlers import upload_handler

    src = inspect.getsource(upload_handler)
    assert "replace_existing=False" in src
