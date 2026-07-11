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


def test_latex_math_stripped_for_speech():
    out = sanitize_for_tts(
        "**Solution** Substituting: $$= 5 \\times 365$$ and $$\\frac{10}{2}$$"
    )
    assert "$$" not in out
    assert "\\times" not in out
    assert "times" in out
    assert "10 over 2" in out
    assert "Solution" in out


def test_orphan_dollars_and_bare_frac():
    out = sanitize_for_tts("The answer is $$\\frac{10}{2} and more")
    assert "$$" not in out
    assert "10 over 2" in out

    out = sanitize_for_tts("Use \\frac{a}{b} for ratios")
    assert "a over b" in out
    assert "\\" not in out


def test_unclosed_math_delimiter_hold():
    from app.services.voice_chunking import has_unclosed_math_delimiters

    assert has_unclosed_math_delimiters("Formula: $$\\frac{1}{2}")
    assert not has_unclosed_math_delimiters("Formula: $$\\frac{1}{2}$$")


def test_voice_live_teaching_vs_full_format():
    from app.services.chat_service import _voice_should_use_text_format

    assert _voice_should_use_text_format("Mathematics") is True
    assert _voice_should_use_text_format("Science", voice_mode=True) is False
    assert _voice_should_use_text_format("Mathematics", voice_mode=True) is False
    assert _voice_should_use_text_format(
        "Mathematics",
        voice_mode=True,
        understanding_scores={"wants_expansion": True},
    ) is True
    assert _voice_should_use_text_format(
        "Mathematics",
        voice_mode=True,
        query="show me the full step by step solution",
    ) is True

    from app.services.section_heading import HeadingInfo, HeadingScope

    main_scope = HeadingScope(
        kind="main_section",
        matched=HeadingInfo(
            raw_hint="Weather Instruments",
            section_number="2.6",
            title="Weather Instruments",
            level=1,
            page=6,
        ),
        page_start=6,
        page_end=12,
    )
    assert _voice_should_use_text_format(
        "Social",
        voice_mode=True,
        query="what are weather instruments",
        heading_scope=main_scope,
    ) is True
