import os
import re
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import CHROMA_PATH

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)


class KeywordRetriever(BaseRetriever):
    docs: List[Document]
    k: int = 2

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        q_terms = set(re.findall(r"\w+", query.lower()))
        if not q_terms:
            return self.docs[: self.k]

        scored = []
        for d in self.docs:
            text = (d.page_content or "").lower()
            score = sum(1 for t in q_terms if t in text)
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [d for s, d in scored if s > 0][: self.k]
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
    # Import AFTER sanitizing env, otherwise chromadb may parse bad values
    # during module import.
    from langchain_community.vectorstores import Chroma
    try:
        vectorstore = Chroma.from_documents(
            docs,
            embedding_model,
            persist_directory=CHROMA_PATH
        )
        vectorstore.persist()
        return vectorstore
    except Exception:
        # Robust fallback that avoids Chroma entirely on problematic runtimes.
        return InMemoryDocVectorStore(docs)

def load_vector_store():
    _sanitize_chroma_env()
    from langchain_community.vectorstores import Chroma
    try:
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model
        )
    except Exception:
        # No persisted Chroma available; return empty fallback retriever.
        return InMemoryDocVectorStore([])
