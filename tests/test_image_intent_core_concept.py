import pytest

from app.services.image_service.image_intent_extractor import extract_image_intent
from app.services.image_service.symbolic_image_filters import phrase_in_text


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Who was Shivaji?", "Shivaji"),
        ("What is a balanced diet?", "balanced diet"),
        ("What are the types of leaves?", "leaves"),
        ("What are the parts of a flower?", "flower"),
        ("How can we make a magnet?", "make a magnet"),
        ("what is map", "map"),
        ("Explain the Battle of Haldighati", "Battle of Haldighati"),
        ("Describe the water cycle", "water cycle"),
    ],
)
def test_core_concept_strips_question_frame(question, expected):
    assert extract_image_intent(question).core_concept == expected


@pytest.mark.parametrize(
    "phrase,text,expected",
    [
        ("read map", "reading a map of the city", True),
        ("battle haldighati", "the battle of haldighati", True),
        ("weather station", "an automated weather station", True),
        ("weather station", "cloudy weather at the bus station", False),
        ("balanced diet", "some sources of fats", False),
    ],
)
def test_phrase_in_text_tolerates_inflection(phrase, text, expected):
    assert phrase_in_text(phrase, text) is expected
