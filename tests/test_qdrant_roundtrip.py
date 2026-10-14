"""Self-check: Qdrant text adapter add/search/delete with sequential BIGINT ids.

Run (Qdrant must be up, e.g. docker compose up -d qdrant):

  PYTHONPATH=. python tests/test_qdrant_roundtrip.py
"""

from __future__ import annotations

import os
import sys

# Force qdrant for this check without mutating process defaults permanently for other imports.
os.environ["VECTOR_BACKEND"] = "qdrant"
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")


def main() -> None:
    from langchain_core.documents import Document

    from app.services.vector_backend import qdrant_text

    coll = "_qdrant_roundtrip_test"
    upload_id = "42"  # sequential BIGINT string
    docs = [
        Document(
            page_content="The water cycle includes evaporation and condensation.",
            metadata={
                "textbook_upload_id": upload_id,
                "content_label": "Roundtrip Chapter",
                "page": 1,
            },
        )
    ]
    # Wipe prior run
    try:
        qdrant_text.delete_docs(coll)
    except Exception:
        pass

    n = qdrant_text.add_documents(docs, collection_name=coll)
    assert n == 1, n

    hits = qdrant_text.similarity_search(
        "evaporation",
        collection_name=coll,
        chapter_ids=[upload_id],
        k=3,
    )
    assert hits, "expected at least one hit"
    assert str(hits[0].metadata.get("textbook_upload_id")) == upload_id

    fetched = qdrant_text.fetch_chapter_chunks(coll, [upload_id], limit=10)
    assert len(fetched) >= 1

    qdrant_text.delete_docs(coll, where_filter={"textbook_upload_id": upload_id})
    after = qdrant_text.fetch_chapter_chunks(coll, [upload_id], limit=10)
    assert after == [], after

    # Cleanup collection
    qdrant_text.delete_docs(coll)
    print("ok: qdrant roundtrip with sequential id", upload_id)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
