"""Tests for student image understanding validators, intent, schemas, RAG query."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from app.modules.image_understanding.errors import ImageTooLargeError, InvalidImageError
from app.modules.image_understanding.intent import detect_intent
from app.modules.image_understanding.prompts import build_retrieval_query
from app.modules.image_understanding.schemas import ImageUnderstandingResult, sanitize_vision_payload
from app.modules.image_understanding.validators import validate_image_bytes


def _png_bytes(w: int = 64, h: int = 64) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (48, 48), "blue").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_validate_png_ok():
    v = validate_image_bytes(_png_bytes(), filename="a.png", declared_mime="image/png")
    assert v.mime_type == "image/png"
    assert v.width == 64


def test_validate_jpeg_ok():
    v = validate_image_bytes(_jpeg_bytes(), filename="a.jpg", declared_mime="image/jpeg")
    assert v.mime_type == "image/jpeg"


def test_reject_empty():
    with pytest.raises(InvalidImageError):
        validate_image_bytes(b"", filename="a.png")


def test_reject_corrupt():
    with pytest.raises(InvalidImageError):
        validate_image_bytes(b"not-an-image", filename="a.png", declared_mime="image/png")


def test_reject_bad_extension():
    with pytest.raises(InvalidImageError):
        validate_image_bytes(_png_bytes(), filename="a.exe", declared_mime="image/png")


def test_reject_oversized(monkeypatch):
    monkeypatch.setattr("app.modules.image_understanding.validators.IMAGE_MAX_SIZE_MB", 0.0001)
    with pytest.raises(ImageTooLargeError):
        validate_image_bytes(_png_bytes(200, 200), filename="big.png")


def test_sanitize_malformed_json():
    r = sanitize_vision_payload({"image_type": "not_a_real_type", "confidence": "nope"})
    assert r.image_type == "unknown"
    assert r.confidence == 0.0


def test_sanitize_questions_strings():
    r = sanitize_vision_payload({"questions": ["Find x", {"text": "Area?", "intent": "solve"}]})
    assert len(r.questions) == 2
    assert r.questions[0].text == "Find x"


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("Solve this", "solve"),
        ("Explain in simple words", "explain"),
        ("Is my answer correct?", "check_answer"),
        ("Summarize this page", "summarize"),
        ("What does this diagram mean?", "explain_diagram"),
    ],
)
def test_intent_from_message(msg, expected):
    result = ImageUnderstandingResult(image_type="textbook_question", confidence=0.8)
    assert detect_intent(msg, result) == expected


def test_intent_from_math_image_when_no_message():
    result = ImageUnderstandingResult(
        image_type="math_problem",
        mathematical_content={"detected": True, "expressions": ["2x=4"]},
        confidence=0.9,
    )
    assert detect_intent("", result) == "solve"


def test_retrieval_query_prefers_topic_not_full_ocr():
    result = sanitize_vision_payload(
        {
            "educational_topic": "Photosynthesis",
            "content_summary": "Leaf process",
            "ocr_text": "A" * 2000,
            "questions": [{"text": "Why are leaves green?", "intent": "explain"}],
        }
    )
    q = build_retrieval_query(result, "Explain this")
    assert "Photosynthesis" in q
    assert "Why are leaves green?" in q
    assert len(q) < 800


def _jpeg_with_exif(w: int = 40, h: int = 20) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° CW
    exif[0x010F] = "PhoneMaker"
    exif[0x8825] = {1: "N", 2: (12.0, 58.0, 0.0)}  # GPS IFD
    buf = BytesIO()
    Image.new("RGB", (w, h), "green").save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_preprocess_applies_exif_orientation_and_strips_metadata():
    from app.modules.image_understanding.preprocessing import preprocess_image

    raw = _jpeg_with_exif()
    assert Image.open(BytesIO(raw)).getexif().get(0x0112) == 6
    out, mime, w, h = preprocess_image(raw, "image/jpeg")
    assert (mime, w, h) == ("image/jpeg", 20, 40)  # rotated upright
    with Image.open(BytesIO(out)) as im:
        exif = im.getexif()
        assert 0x0112 not in exif and 0x010F not in exif and 0x8825 not in exif
        assert "exif" not in im.info


def test_upload_pipeline_strips_gps_before_storage(tmp_path, monkeypatch):
    from app.modules.image_understanding import service as iu_service

    monkeypatch.setattr(iu_service, "IMAGE_TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(iu_service, "_sync_redis", lambda: None)
    meta = iu_service.store_validated_image(validate_image_bytes(_jpeg_with_exif(), filename="p.jpg"), user_id=3)
    data, _ = iu_service.load_image_for_user(meta.image_id, user_id=3)
    assert 0x8825 not in Image.open(BytesIO(data)).getexif()
    iu_service.delete_image(meta.image_id)
