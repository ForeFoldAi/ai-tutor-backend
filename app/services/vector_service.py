import logging
import os
import shutil
import sys
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import CHROMA_PATH
from app.services.query_match import document_page, keyword_match_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Python 3.14 compat: pydantic v1 cannot infer types from deferred
# annotations (PEP 649). ChromaDB 1.x still uses pydantic.v1.BaseSettings
# which triggers ConfigError("unable to infer type …") on Py 3.14.
# We patch the single failing method so the fallback uses Any for fields
# whose type can't be resolved.  The patch is idempotent and only active
# on 3.14+.
# ---------------------------------------------------------------------------
if sys.version_info >= (3, 14):
    try:
        import pydantic.v1.fields as _pv1_fields
        import pydantic.v1.errors as _pv1_errors

        _orig_set_default_and_type = _pv1_fields.ModelField._set_default_and_type

        if not getattr(_orig_set_default_and_type, "_py314_patched", False):
            def _safe_set_default_and_type(self):  # type: ignore[override]
                try:
                    _orig_set_default_and_type(self)
                except _pv1_errors.ConfigError:
                    from typing import Any, Optional
                    self.type_ = type(self.default) if self.default is not None else Any
                    self.outer_type_ = Optional[self.type_]
                    self.required = self.default is ...
                    self.allow_none = True

            _safe_set_default_and_type._py314_patched = True  # type: ignore[attr-defined]
            _pv1_fields.ModelField._set_default_and_type = _safe_set_default_and_type  # type: ignore[assignment]
    except Exception:
        pass
# ---------------------------------------------------------------------------

_embedding_model = None


def is_embedding_model_loaded() -> bool:
    """True when HuggingFace embeddings are already in memory (no load triggered)."""
    return _embedding_model is not None


def _get_embedding_model():
    """
    Lazily create embeddings to avoid crashing server startup when the
    HF Hub is blocked/unavailable.
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        _embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        return _embedding_model
    except Exception:
        # If embeddings can't be created (e.g. HF download blocked),
        # fall back to keyword-only retrieval in-memory.
        _embedding_model = None
        return None


class KeywordRetriever(BaseRetriever):
    docs: List[Document]
    k: int = 2

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        if not (query or "").strip():
            return self.docs[: self.k]

        scored = []
        for d in self.docs:
            text = d.page_content or ""
            score = keyword_match_score(query, text)
            scored.append((score, document_page(d), d))

        # Higher score first; stable tie-break by page (deterministic across similar queries).
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = [d for s, _p, d in scored if s > 0][: self.k]
        if top:
            return top
        return self.docs[: self.k]


class InMemoryDocVectorStore:
    def __init__(self, docs: List[Document]):
        self.docs = docs

    def as_retriever(self, search_kwargs=None):
        k = 2
        if isinstance(search_kwargs, dict):
            try:
                k = int(search_kwargs.get("k", 2))
            except Exception:
                k = 2
        return KeywordRetriever(docs=self.docs, k=k)


def _sanitize_chroma_env():
    # Some environments inject CHROMA_SERVER_NOFILE with values that break
    # chromadb settings parsing ("unable to infer type ...").
    os.environ.pop("CHROMA_SERVER_NOFILE", None)
    os.environ.pop("chroma_server_nofile", None)
    # Force embedded/local behavior and avoid server-mode flags leaking in.
    os.environ.setdefault("IS_PERSISTENT", "TRUE")
    os.environ.setdefault("ALLOW_RESET", "TRUE")


def create_vector_store(docs):
    _sanitize_chroma_env()
    from langchain_community.vectorstores import Chroma

    embedding_model = _get_embedding_model()
    if embedding_model is None:
        return InMemoryDocVectorStore(docs)

    try:
        if os.path.isdir(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        vectorstore = Chroma.from_documents(
            docs,
            embedding_model,
            persist_directory=CHROMA_PATH
        )
        return vectorstore
    except Exception:
        logger.exception("Failed to create Chroma vector store, falling back to in-memory")
        return InMemoryDocVectorStore(docs)

def load_vector_store():
    _sanitize_chroma_env()
    from langchain_community.vectorstores import Chroma

    embedding_model = _get_embedding_model()
    if embedding_model is None:
        return InMemoryDocVectorStore([])

    try:
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model
        )
    except Exception:
        logger.exception("Failed to load Chroma vector store, falling back to in-memory")
        return InMemoryDocVectorStore([])


def add_documents_to_store(docs, *, collection_name: str = "textbooks") -> int:
    """Add documents to an existing ChromaDB collection (or create it).

    Before inserting, any existing vectors that share the same
    ``textbook_upload_id`` metadata are deleted so that re-processing a
    chapter never creates duplicate vectors.

    Returns the number of chunks added.
    """
    _sanitize_chroma_env()
    from langchain_community.vectorstores import Chroma

    embedding_model = _get_embedding_model()
    if embedding_model is None:
        logger.error("Embedding model unavailable — cannot add documents to store")
        return 0

    # Dedup: remove existing vectors for this upload before re-inserting.
    if docs:
        upload_id = (docs[0].metadata or {}).get("textbook_upload_id")
        if upload_id:
            try:
                import chromadb as _chromadb
                _client = _chromadb.PersistentClient(path=CHROMA_PATH)
                _existing = _client.get_collection(collection_name)
                _existing.delete(where={"textbook_upload_id": upload_id})
                logger.info(
                    "Dedup: removed existing vectors for upload_id=%s in collection '%s'",
                    upload_id, collection_name,
                )
            except Exception:
                pass  # Collection may not exist yet on first insert; that's fine.

    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        vs = Chroma(
            collection_name=collection_name,
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model,
        )
        vs.add_documents(docs)
        logger.info("Added %d documents to collection '%s'", len(docs), collection_name)
        return len(docs)
    except Exception:
        logger.exception("Failed to add %d documents to collection '%s'", len(docs), collection_name)
        return 0


def get_collection_stats() -> dict:
    """Return per-collection document counts from the local ChromaDB store."""
    _sanitize_chroma_env()
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = client.list_collections()
        stats: dict = {}
        for coll in collections:
            stats[coll.name] = coll.count()
        return stats
    except Exception:
        logger.exception("Failed to get ChromaDB collection stats")
        return {}


def retrieve_from_collection(
    query: str,
    *,
    collection_name: str,
    chapter_ids: list[str] | None = None,
    k: int = 5,
) -> list:
    """Retrieve relevant chunks from a specific ChromaDB collection.

    When *chapter_ids* is given, only chunks whose ``textbook_upload_id``
    metadata matches one of the IDs are returned.
    """
    _sanitize_chroma_env()
    from langchain_community.vectorstores import Chroma

    embedding_model = _get_embedding_model()
    if embedding_model is None:
        print(f"[RETRIEVE] Embedding model unavailable — cannot retrieve documents")
        return []

    try:
        print(f"[RETRIEVE] collection={collection_name!r}, chapter_ids={chapter_ids}, k={k}")
        vs = Chroma(
            collection_name=collection_name,
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model,
        )

        where_filter = None
        if chapter_ids:
            if len(chapter_ids) == 1:
                where_filter = {"textbook_upload_id": chapter_ids[0]}
            else:
                where_filter = {"textbook_upload_id": {"$in": chapter_ids}}

        print(f"[RETRIEVE] where_filter={where_filter}")

        if where_filter:
            docs = vs.similarity_search(query, k=k, filter=where_filter)
        else:
            docs = vs.similarity_search(query, k=k)

        print(f"[RETRIEVE] got {len(docs)} docs")
        if docs:
            print(f"[RETRIEVE] first doc metadata: {docs[0].metadata}")
        return docs
    except Exception as exc:
        print(f"[RETRIEVE] EXCEPTION: {exc}")
        logger.exception(
            "Failed to retrieve from collection '%s' (chapter_ids=%s)",
            collection_name,
            chapter_ids,
        )
        return []


def delete_collection_docs(collection_name: str, *, where_filter: dict | None = None) -> None:
    """Delete documents from a collection, optionally matching a metadata filter."""
    _sanitize_chroma_env()
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        coll = client.get_collection(collection_name)
        if where_filter:
            coll.delete(where=where_filter)
        else:
            ids = coll.get()["ids"]
            if ids:
                coll.delete(ids=ids)
    except Exception:
        logger.exception("Failed to delete docs from collection '%s'", collection_name)


def vectorstore_has_documents(vs) -> bool:
    """True if the current store likely has chunks to retrieve (voice vs Jarvis routing)."""
    if vs is None:
        return False
    if isinstance(vs, InMemoryDocVectorStore):
        return bool(vs.docs)
    try:
        coll = getattr(vs, "_collection", None)
        if coll is not None:
            return int(coll.count()) > 0
    except Exception:
        pass
    # Unknown store shape — assume indexed so we never silently switch to non-RAG voice.
    return True
