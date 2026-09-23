"""Breadth classification and weak-hit filtering for textbook RAG."""

from langchain_core.documents import Document

from app.services.section_retrieval import (
    filter_weak_semantic_hits,
    query_breadth,
    voice_context_budget,
)


def test_query_breadth_narrow_vs_broad_vs_chapter():
    assert query_breadth("What is the Delhi Sultanate?") == "narrow"
    assert query_breadth("What was the First Battle of Panipat?") == "narrow"
    assert query_breadth("Tell me about India's political map.") == "broad"
    assert query_breadth("Tell me about the Delhi Sultanate.") == "broad"
    assert query_breadth("Explain this chapter.") == "chapter"
    assert query_breadth("What is this chapter about?") == "chapter"


def test_filter_keeps_relative_neighbors():
    docs = [
        Document(page_content="a", metadata={"_retrieval_score": 0.72}),
        Document(page_content="b", metadata={"_retrieval_score": 0.60}),
        Document(page_content="c", metadata={"_retrieval_score": 0.20}),
    ]
    kept = filter_weak_semantic_hits(docs)
    texts = [d.page_content for d in kept]
    assert texts == ["a", "b"]


def test_filter_drops_all_when_top_is_weak():
    docs = [
        Document(page_content="noise", metadata={"_retrieval_score": 0.12}),
        Document(page_content="more", metadata={"_retrieval_score": 0.10}),
    ]
    assert filter_weak_semantic_hits(docs) == []


def test_filter_keeps_unscored_docs():
    docs = [Document(page_content="heading-selected", metadata={"page": 1})]
    assert filter_weak_semantic_hits(docs) == docs


def test_voice_context_budget_grows_only_for_broad():
    assert voice_context_budget(6000, "What is a thermometer?") == 6000
    assert voice_context_budget(6000, "Tell me about India's political map.") >= 11000
