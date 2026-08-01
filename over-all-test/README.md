# over-all-test

Local PDF fixtures and extraction checks used while debugging textbook image quality.

## Files

| File | Purpose |
|------|---------|
| `chapter-1.pdf` / `chapter-2.pdf` | Source PDFs |
| `verify_chapter2_figures.py` | Same check used in chat: extract figures, list sizes/captions, assert 2.1–2.13 |

## Run chapter-2 figure verification

```bash
cd ai-tutor-backend
PYTHONPATH=. .venv/bin/python over-all-test/verify_chapter2_figures.py
PYTHONPATH=. .venv/bin/python over-all-test/verify_chapter2_figures.py --write-crops
```

Report: `over-all-test/chapter-2-figures/report.json`

## How captions are produced

1. **At extraction (PDF pairing)** — `figure_pairing` reads the `Fig. N` label and nearby PDF text / layout caption (`_caption_for_numbered_figure`). That becomes the initial `caption` on the figure asset.
2. **At enrich (after save)** — if the caption is missing, bare (`Fig. 2.1`), or OCR junk:
   - **Contextual** (`caption_generator.generate_contextual_caption`): no LLM; picks the best nearby sentence + section heading + image-type prefix.
   - **Vision** (`vision_caption_service`, BLIP): only when the text caption is still weak; describes pixels, then wraps with subject/chapter.
3. **At display** — `resolve_display_caption` chooses what the student sees: prefer clean generated/vision over corrupt OCR; fall back to PDF caption line / nearby / subtopic.

S3 path after re-extract (example for upload 16):

`textbook_images/CBSE/CLASS_9/Social/Chapter_2_-_Understanding_the_Weather/figures/fig_2_6.jpg`
