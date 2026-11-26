# Figure quality goldens

Hand-checked labels for extracted textbook figures, scored with
`scripts/figure_quality.py score <file>`.

One JSON object per line:

```json
{"image_id": 16135, "junk": false, "crop_ok": true, "caption_ok": true, "note": ""}
```

- `junk`: not a teaching figure (QR code, chapter banner, mascot, page decoration).
- `crop_ok`: the crop shows the whole figure and nothing else (null = not judged).
- `caption_ok`: what the student sees is correct and clear for that class (null = not judged).

`image_id` is `textbook_images.id`, so a file is only valid until those rows are
re-extracted. After re-extraction, regenerate a sample with
`scripts/figure_quality.py sample` and relabel it.

## Files

- `cbse10_seed.jsonl` (196 labels, CBSE Class 10, all subjects). Contains:
  - a 40-figure stratified sample, 5 per book, judged for junk, crop and caption;
  - the 23 QR codes and QR banners that decode;
  - 145 decorations repeated across chapters, each checked visually.

  This is a starting point, not the full ~300-figure multi-board set. Add other boards
  and classes, and scanned PDFs, by sampling and labelling with `review.html`.
