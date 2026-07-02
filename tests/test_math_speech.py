"""Tests for mathematics → spoken English conversion (voice TTS)."""

from app.services.math_speech import latex_to_speech, math_to_speech, plain_math_to_speech
from app.services.tts_sanitize import sanitize_for_tts


def test_squared_cubed_caret():
    assert "a squared" in plain_math_to_speech("a^2")
    assert "x cubed" in plain_math_to_speech("x^3")
    assert "n to the power 4" in plain_math_to_speech("n^4")
    assert "(a + b) squared" in plain_math_to_speech("(a + b)^2")


def test_unicode_superscript():
    assert "a squared" in plain_math_to_speech("a²")
    assert "x cubed" in plain_math_to_speech("x³")


def test_latex_powers_and_frac():
    out = latex_to_speech(r"(a+b)^2 = a^2 + 2ab + b^2")
    assert "squared" in out
    assert "2ab" in out or "2 a b" in out.replace("  ", " ")

    out = latex_to_speech(r"\frac{1}{2}")
    assert "1 over 2" in out


def test_latex_integration_geometry():
    out = latex_to_speech(r"\int_0^1 x^2 \, dx")
    assert "integral" in out
    assert "squared" in out

    out = latex_to_speech(r"\angle ABC")
    assert "angle" in out

    out = latex_to_speech(r"\triangle PQR")
    assert "triangle" in out

    out = latex_to_speech(r"\sqrt{16}")
    assert "square root of 16" in out


def test_plain_geometry_symbols():
    out = plain_math_to_speech("∠ABC and △PQR are related.")
    assert "angle" in out
    assert "integral" in plain_math_to_speech("∫ f(x) dx")


def test_full_sanitize_math_voice():
    src = (
        "**The Formula**\n"
        "$$(a+b)^2 = a^2 + 2ab + b^2$$\n"
        "Area scales as x² and volume as x³."
    )
    out = sanitize_for_tts(src)
    assert "squared" in out
    assert "cubed" in out
    assert "$$" not in out
    assert "Formula" in out


if __name__ == "__main__":
    test_squared_cubed_caret()
    test_unicode_superscript()
    test_latex_powers_and_frac()
    test_latex_integration_geometry()
    test_plain_geometry_symbols()
    test_full_sanitize_math_voice()
    print("ok")
