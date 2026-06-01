"""Tests for spoken-text sanitization before Edge-TTS."""

from app.services.tts_sanitize import sanitize_for_tts


def test_bold_and_italic():
    assert sanitize_for_tts("**Photosynthesis** is the process.") == (
        "Photosynthesis is the process."
    )
    assert sanitize_for_tts("It uses *light* energy.") == "It uses light energy."


def test_heading():
    assert sanitize_for_tts("# Chapter 1") == "Chapter 1"
    assert sanitize_for_tts("## Cell Structure") == "Cell Structure"


def test_list_to_sentences():
    out = sanitize_for_tts("- Plants need sunlight\n- Water is important")
    assert "Plants need sunlight" in out
    assert "Water is important" in out
    assert "-" not in out


def test_code_block():
    src = 'Before.\n```python\nprint("hello")\n```\nAfter.'
    out = sanitize_for_tts(src)
    assert "print" not in out
    assert "code example" in out.lower()
    assert "Before" in out
    assert "After" in out


def test_urls():
    out = sanitize_for_tts("See https://example.com/path for details.")
    assert "example.com" not in out
    assert "provided link" in out.lower()


def test_markdown_link():
    out = sanitize_for_tts("Read [NCERT notes](https://ncert.nic.in/page) today.")
    assert "NCERT notes" in out
    assert "https://" not in out
    assert ".nic.in" not in out


def test_table_row():
    out = sanitize_for_tts("| Col A | Col B |\n| --- | --- |\n| x | y |")
    assert "|" not in out
    assert "Col A" in out or "x" in out


def test_preserve_science():
    out = sanitize_for_tts("Water boils at 100°C and H2O is polar. Growth is ~25%.")
    assert "100" in out
    assert "H2O" in out
    assert "25" in out


def test_html_and_template_noise():
    out = sanitize_for_tts("Hello <div>student</div> {token} [x]")
    assert "<div>" not in out
    assert "student" in out


def test_punctuation_normalize():
    out = sanitize_for_tts("Hello!!!   Student...")
    assert "!!!" not in out
    assert "Hello" in out
    assert "Student" in out


def test_horizontal_rule():
    out = sanitize_for_tts("Intro\n---\nBody text here.")
    assert "---" not in out
    assert "Body text" in out


def test_blockquote():
    out = sanitize_for_tts("> Important: review the chapter.")
    assert "Important" in out
    assert ">" not in out
