# Multimodal Image Understanding

Students can attach a study image (camera, gallery, or file) in:

- Chapter text chat (`/auth/chat`, `/auth/chat/stream`)
- Ask AI Tutor (`/auth/student/assistant/chat/stream`)
- Voice typed turns (Nest WS `text` + `image_ids` → FastAPI)

## Flow

1. `POST /auth/tutor/images` — validate, preprocess, ephemeral store → `{ image_id }`
2. Chat/stream with `image_ids: [...]`
3. `image_understanding.understand()`:
   1. `quality.assess_quality` (Pillow): blank / too small → "unclear" (no OCR, no vision call)
   2. `ocr.run_ocr` — one local Tesseract `image_to_data` pass (words, boxes, confidences, coverage), cached
   3. `routing.decide` — pure rules, no LLM:
      - clean printed text (confidence ≥ `TESSERACT_MIN_CONFIDENCE`, coverage ≥ `TESSERACT_MIN_TEXT_COVERAGE`,
        ≥ `TESSERACT_MIN_WORDS`, no math, no visual question, good quality) → Tesseract result (`source=tesseract`)
      - math, handwriting / low confidence, diagram, graph, map, table, mixed, poor quality → vision LLM
   4. Vision (`LLM_VISION_MODEL`, default Pixtral): 1 call + at most `VISION_MAX_RETRIES` retries on
      timeout / 429 / 5xx / malformed JSON. Success → `source=vision` (or `hybrid` when OCR text was also good).
   5. Vision failed → usable OCR text becomes a low-confidence answer; otherwise `status=unclear`,
      `requires_clearer_image=true` and the tutor asks for a clearer photo. Never loops back to OCR.
   6. Unchanged: intent → optional SymPy → concise retrieval query
4. Existing RAG + tutor LLM (chapter or assistant) with an `IMAGE UNDERSTANDING` prompt block

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_VISION_MODEL` / `IMAGE_VISION_MODEL` | `pixtral-12b-2409` | Vision-capable model (**required** — do not use text-only chat models like `ministral-8b`) |
| `LLM_VISION_API_KEY` / `LLM_VISION_BASE_URL` | LLM_* | Optional overrides |
| `IMAGE_MAX_SIZE_MB` | 8 | Upload size |
| `IMAGE_MAX_WIDTH` / `IMAGE_MAX_HEIGHT` | 4096 | Soft resize ceiling |
| `IMAGE_ALLOWED_MIME` | jpeg,png,webp | Whitelist |
| `IMAGE_ANALYSIS_TIMEOUT` | 45 | Hard wall-clock budget (s) for all vision attempts together, retries included |
| `IMAGE_CACHE_ENABLED` / `IMAGE_CACHE_TTL` | true / 3600 | Analysis-only cache |
| `IMAGE_MAX_IMAGES_PER_REQUEST` | 1 | Cap per turn |
| `IMAGE_TEMP_TTL_SEC` | 600 | Ephemeral file TTL |
| `IMAGE_TEMP_DIR` | `uploads/_tutor_user_images` | Temp storage |
| `IMAGE_ANALYSIS_VERSION` | 1 | Cache key version (vision and OCR) |
| `VISION_MAX_RETRIES` | 1 | Extra vision attempts on transient failure (total calls = 1 + this) |
| `TESSERACT_ENABLED` | true | Local OCR before vision; `false` = previous vision-only behaviour |
| `TESSERACT_CMD` | (PATH) | Absolute path to the `tesseract` binary if not on PATH |
| `TESSERACT_LANG` | `OCR_LANG` (eng) | Tesseract language(s), e.g. `eng` or `eng+hin` (traineddata must be installed) |
| `TESSERACT_MIN_CONFIDENCE` | 0.85 | Routing heuristic: mean word confidence for OCR-only answers |
| `TESSERACT_MIN_TEXT_COVERAGE` | 0.60 | Routing heuristic: share of the image's edge "ink" inside confidently-read word boxes (text pages ≈ 0.7–1.0; graphs, diagrams, tables, photos ≈ 0.1–0.45 on synthetic fixtures) |
| `TESSERACT_MIN_WORDS` | 3 | Routing heuristic: minimum words for OCR-only answers |
| `TESSERACT_MAX_LOW_CONF_RATIO` | 0.05 | Routing heuristic: max share of words below 0.6 confidence for OCR-only answers. Catches partial/garbled reads with a decent average (a real phone photo: average 0.84, ratio 0.08, 3 of 5 answers missing; clean pages ≈ 0–0.02) |
| `TESSERACT_TIMEOUT_SEC` | 10 | Per-image Tesseract timeout; also the max wait for a free OCR slot |
| `TESSERACT_MAX_CONCURRENCY` | CPUs / 2 | Concurrent tesseract processes per worker. When all slots are busy the image skips OCR (`ocr_error:ocr_busy`) and goes to vision instead of queueing |

Use a **vision-capable** model (e.g. Mistral Pixtral / GPT-4o-compatible). Text-only models will fail analysis.

The OCR thresholds are routing heuristics, not claims that Tesseract is correct. High OCR confidence never
bypasses vision when the image needs visual reasoning (graph, diagram, map, handwriting, math layout).

## Tesseract installation

| Environment | How |
|-------------|-----|
| Docker (`Dockerfile`, `Dockerfile.nuitka`) | Builds **Tesseract 5.5.3** (latest upstream, 2026-07-24; fixes CVE-2026-73066/73067) from the official tag tarball in a builder stage, SHA-256 pinned. English data: `tessdata_fast` tag 4.1.0 `eng.traineddata`, SHA-256 pinned. Only binary, `libtesseract` and `tessdata` are copied into the runtime image (`TESSDATA_PREFIX=/usr/local/share/tessdata`). The build fails if the version/language check fails. |
| macOS dev | `brew install tesseract` (check `tesseract --version` reports 5.5.x) |
| Python | `pytesseract==0.3.13` (in `requirements.txt`) |

Add languages by downloading more `*.traineddata` into `tessdata` and setting `TESSERACT_LANG`.
Do not install `tesseract-ocr-all`.

`GET /health/ai-models` → `models.image_understanding.ocr` reports `enabled`, `available`, `version`,
`lang_available` and `status` (`ok` / `degraded` / `disabled`). A missing binary never blocks startup; images
simply route to vision and the status shows `degraded`.

## Observability

One structured `image_understanding {...}` INFO log per analyzed image: image id prefix, quality flags,
OCR attempted / cache hit / latency / confidence / low-confidence word ratio / coverage / word count, route + reason, vision attempted / cache hit /
latency / attempts, final source, status, confidence and failure reason. No image bytes or OCR text are logged.

## Privacy

- Images are ephemeral (TTL) and deleted after analysis.
- Never log raw image bytes or full OCR dumps.
- `image_id` is bound to the uploader; cross-user reuse is rejected.

## Production (multi-instance)

Ephemeral uploads are stored in **Redis** (`REDIS_URL`) so upload and chat can hit different workers.
If Redis is unavailable, images fall back to local disk and multi-worker deploys will return
`"That image is no longer available"`. Ensure Redis is reachable in production.

## API examples

Upload:

```http
POST /auth/tutor/images
Authorization: Bearer <token>
Content-Type: multipart/form-data

file=<image>
```

Chapter chat:

```json
{
  "query": "Explain this in simple words",
  "board": "CBSE",
  "class_level": "7",
  "subject_name": "Science",
  "chapter_ids": ["..."],
  "image_ids": ["<hex id>"]
}
```

## Known v1 limits

- One image per turn in the UI
- Attach applies to typed/send turns during voice (not the next spoken utterance)
- Local OCR is Tesseract only (printed text); handwriting / math / visuals always use the vision model
- Routing never labels an image "handwriting" from OCR confidence alone (low confidence routes as `unknown`);
  `handwriting_detected` in the result comes from the vision model
- The vision cache key includes a hash of the normalised student message (the message steers the vision prompt);
  the OCR cache is per image and stores no word boxes
- Tesseract version is re-read when the binary changes (mtime), so upgrades need no worker restart
