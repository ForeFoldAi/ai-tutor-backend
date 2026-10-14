"""Unit tests for metrics — always runnable, no ML/LLM."""

from evaluation.metrics.chunk_quality import score_chunk
from evaluation.metrics.image_quality import analyze_image_bytes
from evaluation.metrics.llm_quality import groundedness, score_tutor_response
from evaluation.metrics.retrieval import hit_at_k, mrr, ndcg_at_k
from evaluation.metrics.table_quality import parse_markdown_table, score_table


def test_retrieval_metrics():
    relevant = {2, 5}
    ranked = [1, 2, 3, 5]
    assert hit_at_k(relevant, ranked, 1) == 0.0
    assert hit_at_k(relevant, ranked, 2) == 1.0
    assert mrr(relevant, ranked) == 0.5
    assert ndcg_at_k(relevant, ranked, 4) > 0.5


def test_chunk_quality_flags_broken_hyphen():
    out = score_chunk("This paragraph ends badly-", {"section_hint": "1.1 Intro"})
    assert "broken_paragraph_hyphen" in out["issues"]


def test_table_parse():
    md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    parsed = parse_markdown_table(md)
    assert parsed["rows"] == 2
    assert parsed["cols"] == 2
    scored = score_table(md, {"rows": 2, "cols": 2, "headers": ["A", "B"]})
    assert scored["score"] >= 0.75


def test_groundedness_and_tutor_score():
    ctx = ["Weather is the day-to-day condition of the atmosphere."]
    ans = "Weather is the day-to-day condition of the atmosphere around us."
    assert groundedness(ans, ctx) > 0.5
    scored = score_tutor_response(
        ans,
        contexts=ctx,
        expected_keywords=["weather", "atmosphere"],
        must_include=["weather"],
    )
    assert scored["tutor_quality_score"] > 0.5
    assert scored["safety_ok"]


def test_blank_image_detection():
    from PIL import Image
    import io

    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    q = analyze_image_bytes(buf.getvalue())
    assert "blank_or_uniform" in q.issues or q.score < 0.9
