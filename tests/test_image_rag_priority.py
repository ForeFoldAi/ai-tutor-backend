"""Unit checks for image-vs-chapter RAG priority."""

from app.modules.image_understanding.prompts import image_should_override_chapter_rag
from app.modules.image_understanding.schemas import ImageUnderstandingResult, sanitize_vision_payload


def test_math_true_false_image_overrides_chapter():
    r = sanitize_vision_payload(
        {
            "image_type": "math_problem",
            "educational_topic": "Cubes",
            "ocr_text": "State true or false. (i) The cube of any odd number is even.",
            "question_detected": True,
            "questions": [{"text": "The cube of any odd number is even.", "intent": "answer"}],
            "mathematical_content": {"detected": True, "expressions": []},
            "confidence": 0.85,
        }
    )
    assert image_should_override_chapter_rag(r) is True


def test_diagram_without_text_keeps_chapter_rag():
    r = ImageUnderstandingResult(
        image_type="diagram",
        content_summary="",
        ocr_text="",
        confidence=0.7,
    )
    assert image_should_override_chapter_rag(r) is False


def test_textbook_page_ocr_overrides():
    r = sanitize_vision_payload(
        {
            "image_type": "textbook_page",
            "ocr_text": "3. State true or false. Explain your reasoning.",
            "confidence": 0.6,
        }
    )
    assert image_should_override_chapter_rag(r) is True


def test_graph_and_map_without_text_keep_chapter_rag():
    for image_type in ("graph", "map", "science_diagram"):
        assert image_should_override_chapter_rag(ImageUnderstandingResult(image_type=image_type)) is False


def test_unclear_image_never_answers_from_random_chapter():
    r = ImageUnderstandingResult(status="unclear", requires_clearer_image=True, source="none")
    assert image_should_override_chapter_rag(r) is True


def test_tesseract_text_result_is_image_only():
    r = ImageUnderstandingResult(
        image_type="textbook_page",
        ocr_text="Read the passage and answer the questions that follow.",
        source="tesseract",
        confidence=0.93,
    )
    assert image_should_override_chapter_rag(r) is True
