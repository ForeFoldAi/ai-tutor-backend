"""Tests for math textbook extraction helpers."""

from app.services.image_service.formula_bbox import expand_formula_bbox


def test_expand_formula_bbox_adds_padding():
    box = (100, 100, 200, 130)
    expanded = expand_formula_bbox(box, (1000, 1000))
    assert expanded[0] < box[0]
    assert expanded[2] > box[2]
    assert expanded[3] > box[3]


def test_merge_formula_prefers_complete_pdf_text():
    pdf = "(a + b)^2 = a^2 + 2ab + b^2"
    latex = "a^2"
    assert merge_formula_structured_text(pdf, latex) == pdf


def test_merge_formula_keeps_latex_when_longer():
    pdf = "x = 5"
    latex = "x = \\frac{10}{2}"
    assert merge_formula_structured_text(pdf, latex) == latex


def test_looks_like_math_line():
    assert looks_like_math_line("= 200 × 1825")
    assert looks_like_math_line("(a + b)^2 = a(a + b) + b(a + b)")
    assert not looks_like_math_line("Exploring further, let us multiply")


def test_decorative_speech_bubble():
    assert is_decorative_math_figure(
        caption="How did they measure the distance between the Earth and the Sun?",
        figure_context="",
        figure_number=None,
        width=400,
        height=300,
    )


def test_decorative_keeps_numbered_figures():
    assert not is_decorative_math_figure(
        caption="Algebraic identity for cube",
        figure_context="",
        figure_number="4.10",
        width=400,
        height=300,
    )


def test_oversized_prose_formula_rejected():
    prose = " ".join(["word"] * 50)
    assert is_oversized_formula_prose_crop(prose, width=700, height=280)


def test_compact_formula_allowed():
    assert not is_oversized_formula_prose_crop(
        "= 116 × 5 = 580",
        width=360,
        height=70,
    )
