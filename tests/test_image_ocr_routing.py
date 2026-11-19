"""Tesseract OCR layer + routing between local OCR and the vision model (vision always mocked)."""

from __future__ import annotations

import asyncio
import time
from io import BytesIO

import httpx
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.modules.image_understanding import cache as iu_cache
from app.modules.image_understanding import service as iu_service
from app.modules.image_understanding import vision as iu_vision
from app.modules.image_understanding.errors import ImageNotFoundError, VisionProviderError
from app.modules.image_understanding.ocr import service as ocr_service
from app.modules.image_understanding.ocr.schemas import OCRResult, OCRWord
from app.modules.image_understanding.ocr.tesseract import (
    TesseractOCRService,
    ink_text_coverage,
    parse_image_to_data,
)
from app.modules.image_understanding.prompts import image_should_override_chapter_rag
from app.modules.image_understanding.quality import assess_quality
from app.modules.image_understanding.routing import Task, decide
from app.modules.image_understanding.schemas import (
    ImageQualityResult,
    ImageUnderstandingResult,
    MathematicalContent,
    sanitize_vision_payload,
)
from app.modules.image_understanding.validators import validate_image_bytes

# ---------------------------------------------------------------------------
# Synthetic fixtures (no copyrighted content)
# ---------------------------------------------------------------------------


def _png(im: Image.Image) -> bytes:
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _page(lines: int = 12) -> Image.Image:
    im = Image.new("RGB", (1200, 1600), "white")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=36)
    for i in range(lines):
        d.text((60, 80 + i * 110), "The water cycle moves water between land and sky.", fill="black", font=font)
    return im


def _ocr(**kw) -> OCRResult:
    base = dict(
        text="Q1. What is evaporation?\nThe water cycle moves water between land and sky.",
        average_confidence=0.95,
        text_coverage=0.8,
        word_count=20,
        has_text=True,
        language="eng",
    )
    base.update(kw)
    return OCRResult(**base)


GOOD = ImageQualityResult()

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image,flag",
    [
        (Image.new("RGB", (800, 600), "white"), "mostly_blank"),
        (_page().point(lambda p: p * 0.12), "too_dark"),
        (_page().point(lambda p: 215 + p * 0.15), "too_bright"),
        (_page().filter(ImageFilter.GaussianBlur(4)), "blurry"),
        (Image.new("RGB", (20, 20), "gray"), "too_small"),
    ],
)
def test_quality_flags(image, flag):
    q = assess_quality(_png(image))
    assert getattr(q, flag) is True
    assert flag in q.reason


def test_quality_clean_page_and_single_line_ok():
    for im in (_page(), _page(1)):
        q = assess_quality(_png(im))
        assert q.is_usable and q.reason == "" and q.score == 1.0


def test_blank_and_tiny_are_unusable_but_dark_is_usable():
    assert not assess_quality(_png(Image.new("RGB", (800, 600), "white"))).is_usable
    assert not assess_quality(_png(Image.new("RGB", (20, 20), "gray"))).is_usable
    assert assess_quality(_png(_page().point(lambda p: p * 0.12))).is_usable


# ---------------------------------------------------------------------------
# OCR schema + parsing (no Tesseract needed)
# ---------------------------------------------------------------------------


def _tess_dict(rows):
    keys = ["level", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf", "text"]
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


def test_parse_image_to_data_lines_and_noise():
    data = _tess_dict(
        [
            (1, 0, 0, 0, 0, 0, 0, 100, 100, -1, ""),
            (2, 1, 0, 0, 0, 0, 0, 100, 50, -1, ""),
            (5, 1, 1, 1, 1, 0, 0, 10, 10, 96, "Solve"),
            (5, 1, 1, 1, 2, 12, 0, 10, 10, 90, "2x"),
            (5, 1, 1, 1, 3, 24, 0, 5, 10, 88, "="),  # math symbol kept
            (5, 1, 1, 1, 4, 30, 0, 5, 10, 91, "8"),
            (5, 1, 1, 2, 1, 0, 20, 10, 10, 40, "|"),  # noise token dropped
            (5, 1, 1, 2, 2, 12, 20, 10, 10, 94, "  next\tline "),
            (5, 1, 1, 2, 3, 30, 20, 10, 10, "bad", "junk"),  # unparseable row skipped
        ]
    )
    r = parse_image_to_data(data, language="eng")
    assert r.text == "Solve 2x = 8\nnext line"
    assert r.word_count == 5
    assert r.has_text
    assert r.average_confidence == pytest.approx((0.96 + 0.90 + 0.88 + 0.91 + 0.94) / 5)
    assert r.low_confidence_ratio == 0.0


def test_parse_empty_output_has_no_text():
    r = parse_image_to_data(_tess_dict([(1, 0, 0, 0, 0, 0, 0, 10, 10, -1, "")]), language="eng")
    assert not r.has_text and r.word_count == 0 and r.text == "" and r.average_confidence == 0.0


def test_ink_coverage_separates_text_from_drawings():
    im = Image.new("RGB", (600, 400), "white")
    d = ImageDraw.Draw(im)
    d.text((20, 20), "Label", fill="black", font=ImageFont.load_default(size=30))
    label = OCRWord(text="Label", confidence=0.95, left=18, top=18, width=90, height=40)
    assert ink_text_coverage(im, [label]) > 0.9  # only the label has ink
    d.line((50, 350, 550, 100), fill="black", width=4)
    d.ellipse((250, 150, 450, 350), outline="black", width=4)
    assert ink_text_coverage(im, [label]) < 0.5  # drawing ink is not covered by any word
    assert ink_text_coverage(im, [label.model_copy(update={"confidence": 0.2})]) == 0.0  # unsure words ignored
    assert ink_text_coverage(Image.new("RGB", (100, 100), "white"), []) == 0.0


def test_ocr_schema_clamps_untrusted_values():
    w = OCRWord(text=" a\n b ", confidence=5)
    assert w.text == "a b" and w.confidence == 1.0
    r = OCRResult(average_confidence="bad", text_coverage=3)
    assert r.average_confidence == 0.0 and r.text_coverage == 1.0
    with pytest.raises(Exception):
        OCRWord(text="x", left=-1)


def test_tesseract_missing_binary_returns_structured_error():
    r = TesseractOCRService(cmd="/nonexistent/tesseract").run(_png(_page(1)))
    assert r.error == "tesseract_not_found"
    assert not r.has_text and r.text == ""


def test_tesseract_timeout_returns_structured_error(monkeypatch):
    import pytesseract

    def boom(*_a, **_k):
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(pytesseract, "image_to_data", boom)
    assert TesseractOCRService().run(_png(_page(1))).error == "tesseract_timeout"


def test_run_ocr_disabled(monkeypatch):
    monkeypatch.setattr(ocr_service, "TESSERACT_ENABLED", False)
    assert asyncio.run(ocr_service.run_ocr(b"x", "h")) == (None, False)
    assert ocr_service.ocr_status() == {"enabled": False, "status": "disabled"}


def test_ocr_status_degraded_when_binary_missing(monkeypatch):
    monkeypatch.setattr(ocr_service, "TESSERACT_ENABLED", True)
    monkeypatch.setattr(ocr_service, "_service", TesseractOCRService(cmd="/nonexistent/tesseract"))
    st = ocr_service.ocr_status()
    assert st["status"] == "degraded" and st["available"] is False and st["version"] is None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ocr,quality,message,task,use_vision",
    [
        (_ocr(), GOOD, "", Task.TEXT_ONLY, False),
        (_ocr(), GOOD, "explain this in simple words", Task.TEXT_ONLY, False),
        (_ocr(average_confidence=0.55), GOOD, "", Task.UNKNOWN, True),
        # Real phone photo: decent average but 8% of words garbled → OCR text is partial, use vision.
        (_ocr(average_confidence=0.86, low_confidence_ratio=0.08), GOOD, "answer this", Task.UNKNOWN, True),
        (_ocr(low_confidence_ratio=0.02), GOOD, "", Task.TEXT_ONLY, False),
        (_ocr(text="", word_count=0, has_text=False, average_confidence=0.0), GOOD, "", Task.UNKNOWN, True),
        (_ocr(), GOOD, "can you check my answer", Task.HANDWRITING, True),
        (_ocr(text="Solve 2x + 3 = 7"), GOOD, "", Task.MATH, True),
        (_ocr(), GOOD, "solve this", Task.MATH, True),
        (_ocr(), GOOD, "label the parts of this diagram", Task.DIAGRAM, True),
        # High-confidence OCR of "Find the slope" must still go to vision: Tesseract cannot see the graph.
        (_ocr(text="Find the slope of the line"), GOOD, "find the slope of the graph", Task.GRAPH, True),
        (_ocr(), GOOD, "mark these states on the map", Task.MAP, True),
        (_ocr(), GOOD, "explain this table", Task.TABLE, True),
        (_ocr(text_coverage=0.2), GOOD, "", Task.MIXED, True),
        (_ocr(), ImageQualityResult(blurry=True, reason="blurry", score=0.6), "", Task.UNKNOWN, True),
        (None, GOOD, "", Task.UNKNOWN, True),
        (_ocr(error="tesseract_not_found"), GOOD, "", Task.UNKNOWN, True),
        (
            _ocr(),
            ImageQualityResult(is_usable=False, mostly_blank=True, reason="mostly_blank", score=0.0),
            "",
            Task.UNCLEAR,
            False,
        ),
    ],
)
def test_router(ocr, quality, message, task, use_vision):
    d = decide(ocr, quality, message)
    assert (d.task, d.use_vision) == (task, use_vision), d.reason


def test_router_year_ranges_are_not_math():
    d = decide(_ocr(text="The revolt of 1857-58 began in Meerut.\nIt spread quickly across north India."), GOOD, "")
    assert d.task is Task.TEXT_ONLY


# ---------------------------------------------------------------------------
# Vision provider bounded retry (llm_client mocked)
# ---------------------------------------------------------------------------


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://vision.test")
    return httpx.HTTPStatusError("err", request=req, response=httpx.Response(code, request=req))


def _run_provider(monkeypatch, outcomes, retries=1):
    calls = {"n": 0}

    async def fake_complete(*_a, **_k):
        out = outcomes[calls["n"]]
        calls["n"] += 1
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr("app.services.llm_client.complete", fake_complete)
    monkeypatch.setattr(iu_vision, "VISION_MAX_RETRIES", retries)
    monkeypatch.setattr(iu_vision, "_RETRY_WAIT_SEC", 0)
    provider = iu_vision.OpenAICompatibleVisionProvider()
    result = asyncio.run(provider.analyze_image(b"img", "image/png"))
    return result, calls["n"], provider.attempts


GOOD_JSON = '{"image_type": "graph", "confidence": 0.9, "content_summary": "A line graph"}'


def test_vision_retries_transient_503_once(monkeypatch):
    result, calls, attempts = _run_provider(monkeypatch, [_status_error(503), GOOD_JSON])
    assert result.image_type == "graph" and calls == 2 and attempts == 2


def test_vision_retries_timeout_once(monkeypatch):
    result, calls, _ = _run_provider(monkeypatch, [httpx.ReadTimeout("slow"), GOOD_JSON])
    assert result.confidence == 0.9 and calls == 2


def test_vision_gives_up_after_budget(monkeypatch):
    with pytest.raises(VisionProviderError):
        _run_provider(monkeypatch, [_status_error(503), _status_error(503), GOOD_JSON])


def test_vision_does_not_retry_auth_error(monkeypatch):
    calls = {"n": 0}

    async def fake_complete(*_a, **_k):
        calls["n"] += 1
        raise _status_error(401)

    monkeypatch.setattr("app.services.llm_client.complete", fake_complete)
    monkeypatch.setattr(iu_vision, "_RETRY_WAIT_SEC", 0)
    with pytest.raises(VisionProviderError):
        asyncio.run(iu_vision.OpenAICompatibleVisionProvider().analyze_image(b"img", "image/png"))
    assert calls["n"] == 1


def test_vision_malformed_json_retried_then_ok(monkeypatch):
    result, calls, _ = _run_provider(monkeypatch, ["not json at all", GOOD_JSON])
    assert result.image_type == "graph" and calls == 2


def test_vision_malformed_json_twice_returns_marked_result(monkeypatch):
    result, calls, _ = _run_provider(monkeypatch, ["nope", "still nope", GOOD_JSON])
    assert iu_vision.MALFORMED_OUTPUT in result.unclear_regions and calls == 2


def test_vision_json_with_raw_newlines_in_strings_parses_first_try(monkeypatch):
    fenced = '```json\n{"image_type": "question_paper", "ocr_text": "\n D. Match\n Column A\n", "confidence": 0.95}\n```'
    result, calls, _ = _run_provider(monkeypatch, [fenced])
    assert "Match" in result.ocr_text and result.confidence == 0.95 and calls == 1


def test_vision_total_deadline_caps_all_attempts(monkeypatch):
    async def hang(*_a, **_k):
        await asyncio.sleep(10)

    monkeypatch.setattr("app.services.llm_client.complete", hang)
    monkeypatch.setattr(iu_vision, "IMAGE_ANALYSIS_TIMEOUT", 0.2)
    t0 = time.monotonic()
    with pytest.raises(VisionProviderError):
        asyncio.run(iu_vision.OpenAICompatibleVisionProvider().analyze_image(b"img", "image/png"))
    assert time.monotonic() - t0 < 2


def test_vision_no_retry_when_disabled(monkeypatch):
    with pytest.raises(VisionProviderError):
        _run_provider(monkeypatch, [_status_error(503), GOOD_JSON], retries=0)


def test_llm_client_does_not_nest_vision_retries():
    from app.services.llm_client import _max_attempts

    assert _max_attempts("vision") == 1


def test_sanitize_strips_pipeline_fields_from_model_json():
    r = sanitize_vision_payload({"source": "tesseract", "status": "unclear", "requires_clearer_image": True, "confidence": 0.8})
    assert r.source == "vision" and r.status == "ok" and not r.requires_clearer_image and r.confidence == 0.8


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_cache(monkeypatch):
    monkeypatch.setattr(iu_cache, "IMAGE_CACHE_ENABLED", True)
    monkeypatch.setattr(iu_cache, "_redis_down_until", float("inf"))
    iu_cache._MEMORY.clear()
    yield
    iu_cache._MEMORY.clear()


def test_memory_cache_is_bounded_and_ocr_cache_drops_word_boxes(memory_cache, monkeypatch):
    monkeypatch.setattr(iu_cache, "_MEMORY_MAX", 3)
    for i in range(5):
        iu_cache.set_cached_ocr(f"h{i}", "tesseract-5.5.3", "eng", _ocr(words=[OCRWord(text="x", confidence=0.9)]))
    assert len(iu_cache._MEMORY) == 3
    assert iu_cache.get_cached_ocr("h0", "tesseract-5.5.3", "eng") is None
    hit = iu_cache.get_cached_ocr("h4", "tesseract-5.5.3", "eng")
    assert hit.words == [] and hit.word_count == 20


def test_redis_failure_backs_off(monkeypatch):
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise ConnectionError("no redis")

    monkeypatch.setattr("redis.from_url", boom)
    monkeypatch.setattr(iu_cache, "_redis_down_until", 0.0)
    assert iu_cache._redis_call(lambda r: r.get("k")) is None
    assert iu_cache._redis_call(lambda r: r.get("k")) is None
    assert calls["n"] == 1


def test_ocr_and_vision_cache_keys_are_separate(memory_cache):
    assert iu_cache.cache_key("abc", "pixtral-12b-2409") == "imgund:1:pixtral-12b-2409:abc"
    assert iu_cache.ocr_cache_key("abc", "tesseract-5.5.3", "eng") == "imgocr:1:tesseract-5.5.3:eng:abc"
    assert iu_cache.get_cached_ocr("abc", "tesseract-5.5.3", "eng") is None
    iu_cache.set_cached_ocr("abc", "tesseract-5.5.3", "eng", _ocr())
    hit = iu_cache.get_cached_ocr("abc", "tesseract-5.5.3", "eng")
    assert hit is not None and hit.word_count == 20
    assert iu_cache.get_cached_ocr("abc", "tesseract-5.5.3", "hin") is None
    assert iu_cache.get_cached_analysis("abc", "pixtral-12b-2409") is None


def test_run_ocr_uses_cache_and_skips_errors(memory_cache, monkeypatch):
    runs = {"n": 0}

    class FakeService:
        lang = "eng"

        def version(self):
            return "5.5.3"

        def run(self, _data):
            runs["n"] += 1
            return _ocr()

    monkeypatch.setattr(ocr_service, "TESSERACT_ENABLED", True)
    monkeypatch.setattr(ocr_service, "_service", FakeService())
    first, hit1 = asyncio.run(ocr_service.run_ocr(b"img", "hash1"))
    second, hit2 = asyncio.run(ocr_service.run_ocr(b"img", "hash1"))
    assert (hit1, hit2, runs["n"]) == (False, True, 1)
    assert first.engine == "tesseract-5.5.3" and second.text == first.text


def test_run_ocr_busy_skips_to_vision_instead_of_queueing(memory_cache, monkeypatch):
    monkeypatch.setattr(ocr_service, "TESSERACT_ENABLED", True)
    monkeypatch.setattr(ocr_service, "TESSERACT_TIMEOUT_SEC", 0.05)

    async def scenario():
        monkeypatch.setattr(ocr_service, "_slots", asyncio.Semaphore(0))
        return await ocr_service.run_ocr(b"img", "busy-hash")

    result, hit = asyncio.run(scenario())
    assert result.error == ocr_service.OCR_BUSY and not hit
    assert decide(result, GOOD, "").use_vision


def test_version_cache_follows_binary_changes(monkeypatch):
    from app.modules.image_understanding.ocr import tesseract as tess

    stamp = {"v": 0}
    monkeypatch.setattr(tess, "_binary_stamp", lambda _cmd: stamp["v"])
    monkeypatch.setattr(tess, "_tesseract_version_at", lambda _cmd, s: None if s == 0 else f"5.5.{s}")
    assert tess._tesseract_version("") is None  # not installed yet
    stamp["v"] = 3
    assert tess._tesseract_version("") == "5.5.3"  # installed later: picked up without restart


@pytest.mark.parametrize(
    "lang,text,expected",
    [
        ("eng", "Water cycle", "English"),
        ("eng+hin", "जल चक्र क्या है? जल वाष्प", "Hindi"),
        ("eng+hin", "What is the water cycle?", "English"),
        ("hin+eng", "", "Hindi"),
    ],
)
def test_ocr_language_follows_script(lang, text, expected):
    assert iu_service._ocr_language(_ocr(language=lang, text=text)) == expected


# ---------------------------------------------------------------------------
# understand() orchestration (OCR + vision mocked)
# ---------------------------------------------------------------------------


class FakeProvider:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0
        self.attempts = 0

    async def analyze_image(self, *_a, **_k):
        self.calls += 1
        self.attempts = 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def pipeline(tmp_path, monkeypatch, memory_cache):
    monkeypatch.setattr(iu_service, "IMAGE_TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(iu_service, "_sync_redis", lambda: None)
    state: dict = {"ocr": _ocr(), "ocr_calls": 0, "provider": FakeProvider(ImageUnderstandingResult(confidence=0.9))}

    async def fake_run_ocr(_data, _hash):
        state["ocr_calls"] += 1
        return state["ocr"], False

    monkeypatch.setattr(iu_service, "run_ocr", fake_run_ocr)
    monkeypatch.setattr(iu_service, "get_vision_provider", lambda: state["provider"])

    def store(im: Image.Image = None, user_id: int = 7) -> str:
        validated = validate_image_bytes(_png(im or _page()), filename="a.png")
        return iu_service.store_validated_image(validated, user_id=user_id).image_id

    def understand(image_id, query="", user_id=7, delete_after=True):
        return asyncio.run(
            iu_service.understand([image_id], query, user_id=user_id, class_level="7", delete_after=delete_after)
        )

    state["store"], state["understand"] = store, understand
    return state


def test_clean_printed_text_uses_tesseract_only(pipeline):
    bundle = pipeline["understand"](pipeline["store"]())
    assert pipeline["provider"].calls == 0
    r = bundle.result
    assert r.source == "tesseract" and r.status == "ok"
    assert r.image_type == "textbook_question" and r.questions[0].text.startswith("Q1.")
    assert r.ocr_confidence == pytest.approx(0.95) and r.image_quality is not None
    assert bundle.routing_task == "text_only"
    assert "local OCR" in bundle.tutor_prompt_block
    assert image_should_override_chapter_rag(r) is True  # text-heavy upload stays image-only
    assert bundle.retrieval_query and len(bundle.retrieval_query) < 400  # concise, not full OCR


def test_low_confidence_ocr_falls_back_to_vision(pipeline):
    pipeline["ocr"] = _ocr(average_confidence=0.4)
    bundle = pipeline["understand"](pipeline["store"]())
    assert pipeline["provider"].calls == 1
    assert bundle.result.source == "vision" and bundle.routing_task == "unknown"


def test_garbled_ocr_is_never_hybrid_text(pipeline):
    pipeline["ocr"] = _ocr(low_confidence_ratio=0.08)
    bundle = pipeline["understand"](pipeline["store"](), query="explain this diagram")
    assert bundle.result.source == "vision" and bundle.result.ocr_text == ""


def test_vision_cache_is_per_student_message(pipeline):
    pipeline["ocr"] = _ocr(average_confidence=0.4)
    image_id = pipeline["store"]()
    pipeline["understand"](image_id, query="answer this", delete_after=False)
    pipeline["understand"](image_id, query="  Answer   THIS ", delete_after=False)
    assert pipeline["provider"].calls == 1  # normalised message → same key
    pipeline["understand"](image_id, query="what is the capital?")
    assert pipeline["provider"].calls == 2


def test_visual_question_with_good_ocr_is_hybrid(pipeline):
    pipeline["provider"] = FakeProvider(ImageUnderstandingResult(image_type="diagram", confidence=0.88))
    bundle = pipeline["understand"](pipeline["store"](), query="explain this diagram")
    r = bundle.result
    assert pipeline["provider"].calls == 1
    assert r.source == "hybrid" and r.image_type == "diagram"
    assert "evaporation" in r.ocr_text  # vision gave no text; Tesseract's high-confidence text fills it


def test_vision_failure_with_usable_ocr_degrades_to_ocr(pipeline):
    pipeline["ocr"] = _ocr(average_confidence=0.7)
    pipeline["provider"] = FakeProvider(VisionProviderError())
    bundle = pipeline["understand"](pipeline["store"]())
    r = bundle.result
    assert r.source == "tesseract" and r.status == "ok"
    assert bundle.low_confidence and r.confidence < 0.45 and r.unclear_regions


def test_both_ocr_and_vision_fail_returns_unclear(pipeline):
    pipeline["ocr"] = _ocr(text="", word_count=0, has_text=False, average_confidence=0.0)
    pipeline["provider"] = FakeProvider(VisionProviderError())
    bundle = pipeline["understand"](pipeline["store"]())
    r = bundle.result
    assert r.status == "unclear" and r.requires_clearer_image and r.confidence == 0.0 and r.source == "none"
    assert r.ocr_text == "" and not r.questions  # nothing fabricated
    assert bundle.low_confidence
    assert "clearer" in bundle.tutor_prompt_block
    assert image_should_override_chapter_rag(r) is True


def test_malformed_vision_output_is_a_failure_and_not_cached(pipeline):
    pipeline["ocr"] = _ocr(error="tesseract_not_found")
    pipeline["provider"] = FakeProvider(
        ImageUnderstandingResult(confidence=0.2, unclear_regions=[iu_vision.MALFORMED_OUTPUT])
    )
    image_id = pipeline["store"]()
    bundle = pipeline["understand"](image_id, delete_after=False)
    assert bundle.result.status == "unclear"
    pipeline["understand"](image_id)
    assert pipeline["provider"].calls == 2  # malformed result was not cached


def test_blank_image_skips_ocr_and_vision(pipeline):
    bundle = pipeline["understand"](pipeline["store"](Image.new("RGB", (800, 600), "white")))
    assert pipeline["ocr_calls"] == 0 and pipeline["provider"].calls == 0
    assert bundle.result.status == "unclear" and bundle.routing_task == "unclear"


def test_tesseract_disabled_keeps_vision_only_behaviour(pipeline):
    pipeline["ocr"] = None
    bundle = pipeline["understand"](pipeline["store"]())
    assert pipeline["provider"].calls == 1
    assert bundle.result.source == "vision" and bundle.result.ocr_confidence is None


def test_vision_cache_hit_avoids_second_call(pipeline):
    pipeline["ocr"] = _ocr(average_confidence=0.4)
    image_id = pipeline["store"]()
    pipeline["understand"](image_id, delete_after=False)
    pipeline["understand"](image_id)
    assert pipeline["provider"].calls == 1


def test_math_from_vision_goes_through_sympy(pipeline):
    pipeline["ocr"] = _ocr(text="Solve 2x + 3 = 7")
    pipeline["provider"] = FakeProvider(
        ImageUnderstandingResult(
            image_type="math_problem",
            confidence=0.9,
            mathematical_content=MathematicalContent(detected=True, expressions=["2x + 3 = 7"]),
        )
    )
    bundle = pipeline["understand"](pipeline["store"](), query="solve")
    assert bundle.routing_task == "math" and pipeline["provider"].calls == 1
    assert bundle.math_prompt_block.strip()
    assert bundle.intent == "solve"


def test_abandoned_disk_uploads_are_swept(pipeline, tmp_path, monkeypatch):
    import os

    old = tmp_path / "deadbeef.png"
    old.write_bytes(b"x")
    os.utime(old, (0, 0))
    monkeypatch.setattr(iu_service, "_last_sweep", 0.0)
    fresh = pipeline["store"]()
    assert not old.exists()
    assert (tmp_path / f"{fresh}.meta.json").exists()


def test_other_user_cannot_analyze_image(pipeline):
    image_id = pipeline["store"](user_id=7)
    with pytest.raises(ImageNotFoundError):
        pipeline["understand"](image_id, user_id=8)
    assert pipeline["ocr_calls"] == 0 and pipeline["provider"].calls == 0
