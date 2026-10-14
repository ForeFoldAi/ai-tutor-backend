"""Self-check: parse structured PPT markdown and render a themed PPTX."""

from __future__ import annotations

import tempfile
from pathlib import Path

from app.services.lesson_planner.export.deck_schema import ppt_slides_from_markdown
from app.services.lesson_planner.export.figure_assets import attach_figures_to_slides, normalize_icon
from app.services.lesson_planner.export.pptx_builder import build_classroom_pptx

SAMPLE = """
### Slide 1: Title Slide
**Layout:** title
**Side Heading:** Welcome
**Icon:** engage
**Slide Content:**
- Class 9 Mathematics
**Speaker Notes:** Greet the class.
**Callout:** ** Accurate weather data saves lives!

### Slide 2: What is a Fraction?
**Layout:** bullets
**Side Heading:** Concept
**Icon:** diagram
**Slide Content:**
- A fraction names equal parts of a whole
- Numerator counts the parts we take
- Denominator counts how many equal parts make one whole
**Callout:** Parts must be equal.
**Speaker Notes:** Use a pizza diagram.

### Slide 3: Concept vs Example
**Layout:** two_column
**Side Heading:** Compare
**Icon:** example
**Left:**
- Proper fraction is less than 1
- Improper fraction is 1 or more
**Right:**
- 3/4 of a chocolate bar
- 5/4 means one whole and one quarter

### Slide 4: How to Simplify
**Layout:** steps
**Side Heading:** Method
**Icon:** try
**Slide Content:**
- Find a common factor
- Divide numerator and denominator
- Check if it can simplify further

### Slide 5: Look at the Diagram
**Layout:** figure
**Side Heading:** Diagram
**Icon:** diagram
**Figure:** pizza.png — Equal parts of a pizza
**Slide Content:**
- Each slice is one equal part
- Shade 3 of 8 slices for 3/8

### Slide 6: Extra filler A
**Layout:** bullets
**Side Heading:** Extra
**Slide Content:**
- Extra point one
- Extra point two

### Slide 7: Extra filler B
**Layout:** bullets
**Side Heading:** Extra
**Slide Content:**
- Extra point three
- Extra point four

### Slide 8: Extra filler C
**Layout:** section
**Side Heading:** Divider

### Slide 9: Broken Steps
**Layout:** steps
**Side Heading:** Bad
**Slide Content:**
- Only one step

### Slide 10: How a Thermometer Works
**Layout:** steps
**Side Heading:** Try this
**Slide Content:**

### Slide 11: Practice – Match the Instrument
**Layout:** steps
**Side Heading:** Your turn
**Interactive Element:** Match thermometer, rain gauge, and barometer to what they measure
**Slide Content:**

### Slide 12: Summary Checklist
**Layout:** summary
**Side Heading:** Exit ticket
**Icon:** remember
**Slide Content:**
- Summary Checklist

### Slide 13: Thank You!
**Layout:** title
**Side Heading:** Close
"""


def main() -> None:
    assert normalize_icon("Try This") == "try"
    slides = ppt_slides_from_markdown(
        SAMPLE,
        chapter="Fractions",
        subject="Mathematics",
        max_slides=8,
    )
    assert len(slides) <= 8, len(slides)
    assert slides[0]["title"] == "Fractions", slides[0]["title"]
    assert "Title Slide" not in slides[0]["title"]
    assert all("**" not in (s.get("callout") or "") for s in slides), slides

    # Empty steps/summary must be repaired — never title-only cards.
    thermo = next((s for s in slides if "Thermometer" in s["title"]), None)
    practice = next((s for s in slides if "Match" in s["title"] or "Practice" in s["title"]), None)
    summary = next((s for s in slides if s["layout"] == "summary" and "Thank" not in s["title"]), None)
    if thermo:
        assert len(thermo["bullets"]) >= 2, thermo
        assert not any("thermometer works" in b.lower() and len(b.split()) <= 5 for b in thermo["bullets"])
    if practice:
        assert len(practice["bullets"]) >= 2, practice
    if summary:
        assert all("summary checklist" not in b.lower() for b in summary["bullets"]), summary

    # Empty section / broken steps should not dominate the deck.
    assert not any(s["title"] == "Extra filler C" for s in slides)

    enriched = attach_figures_to_slides(
        slides,
        [{"file_name": "pizza.png", "caption": "Equal parts of a pizza", "textbook_upload_id": "1"}],
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "deck.pptx"
        build_classroom_pptx(
            out,
            slides=enriched,
            theme_id="stem_focus",
            meta={
                "chapter": "Fractions",
                "subject": "Mathematics",
                "grade": "Grade 9",
                "eyebrow": "Mathematics · Classroom Deck",
            },
        )
        assert out.exists() and out.stat().st_size > 5000, out.stat().st_size

    print("deck_pptx check: ok")


if __name__ == "__main__":
    main()
