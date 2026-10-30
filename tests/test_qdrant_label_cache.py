"""Label map / chapter chunks are cached per collection and dropped when that collection is written."""

from unittest.mock import patch

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.services.vector_backend import qdrant_text

COLL = "CBSE_CLASS_8_Science"


def _client_with_chapter() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(COLL, vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE))
    client.upsert(COLL, points=[
        qm.PointStruct(id=i, vector=[1.0, 0.0, 0.0, float(i)],
                       payload={"textbook_upload_id": "10", "content_label": "Light", "page": i, "page_content": f"p{i}"})
        for i in range(3)
    ])
    return client


def test_label_map_cached_then_invalidated_by_add_documents():
    client = _client_with_chapter()
    qdrant_text.invalidate_collection_cache(COLL)
    with patch.object(qdrant_text, "get_qdrant_client", return_value=client), \
         patch("app.services.vector_backend.qdrant_common.get_qdrant_client", return_value=client), \
         patch.object(qdrant_text, "_embed_docs", side_effect=lambda docs: [[0.0, 1.0, 0.0, 0.0] for _ in docs]):
        assert qdrant_text._label_to_upload_id(COLL) == {"Light": "10"}

        with patch.object(client, "scroll", side_effect=AssertionError("label map should be cached")):
            assert qdrant_text._label_to_upload_id(COLL) == {"Light": "10"}

        qdrant_text.add_documents(
            [Document(page_content="sound", metadata={"textbook_upload_id": "11", "content_label": "Sound"})],
            collection_name=COLL,
        )
        assert qdrant_text._label_to_upload_id(COLL) == {"Light": "10", "Sound": "11"}


def test_chapter_chunks_cached_as_copies_and_invalidated_by_delete():
    client = _client_with_chapter()
    qdrant_text.invalidate_collection_cache(COLL)
    with patch.object(qdrant_text, "get_qdrant_client", return_value=client):
        first = qdrant_text.fetch_chapter_chunks(COLL, ["10"])
        assert [d.page_content for d in first] == ["p0", "p1", "p2"]
        first[0].metadata["section_title"] = "mutated by caller"

        with patch.object(client, "scroll", side_effect=AssertionError("chunks should be cached")):
            again = qdrant_text.fetch_chapter_chunks(COLL, ["10"])
        assert "section_title" not in again[0].metadata

        qdrant_text.delete_docs(COLL, where_filter={"textbook_upload_id": "10"})
        assert qdrant_text.fetch_chapter_chunks(COLL, ["10"]) == []
