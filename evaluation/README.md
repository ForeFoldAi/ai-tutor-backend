# AI Evaluation & Validation Framework

Production-grade QA for the AI Tutor backend. This is **not** HTTP smoke testing — it validates extraction quality, retrieval accuracy, educational response quality, lesson artifacts, and regressions after code changes.

## Quick start

```bash
cd ai-tutor-backend
source .venv/bin/activate
pip install -r evaluation/requirements-eval.txt

# List phases
python evaluation/run_evaluation.py --list-phases

# Smoke (no heavy ML / LLM)
python evaluation/run_evaluation.py --preset smoke --skip-llm --skip-heavy-ml

# Extraction only
python evaluation/run_evaluation.py --preset extraction

# Full suite
python evaluation/run_evaluation.py --preset full

# Feature subset
python evaluation/run_evaluation.py --phases retrieval,ai_tutor,pipeline,reporting
```

Pytest:

```bash
cd evaluation
pytest tests/test_metrics_unit.py -q
pytest tests/retrieval -m retrieval
pytest tests/images -m images
pytest tests/pipeline -m pipeline
pytest -n auto   # parallel (pytest-xdist)
```

## Architecture (phase-wise)

| Phase | Folder | What it validates |
|------|--------|-------------------|
| 00 | `phases/phase00_core` | Framework docs / extension guide |
| 01 | `phase01_pdf_extraction` | Pages, order, bboxes, figures/tables/text |
| 02 | `phase02_images` | Half/crop/blank/dup/corrupt/page-map |
| 03 | `phase03_tables` | Rows/cols/headers/structure |
| 04 | `phase04_ocr` | Text presence, noise, golden phrases |
| 05 | `phase05_chunking` | Size, overlap integrity, headings |
| 06 | `phase06_embeddings` | Create, dim (768), latency, store |
| 07 | `phase07_retrieval` | Top-1/3/5, MRR, nDCG, page/chapter |
| 08 | `phase08_topics` | Topics/hierarchy vs golden |
| 09 | `phase09_captions` | Caption–figure–page association |
| 10 | `phase10_ai_tutor` | Groundedness, hallucination, curriculum |
| 11 | `phase11_voice_tutor` | Voice module/prompt/TTS chunk contracts |
| 12 | `phase12_lesson_planner` | Objectives, Bloom, schema, time |
| 13 | `phase13_worksheet` | Coverage, variety, schema |
| 14 | `phase14_quiz` | MCQ quality, distractors, duplicates |
| 15 | `phase15_homework` | Tasks, outcomes, schema |
| 16 | `phase16_science_experiments` | Experiment match + schema |
| 17 | `phase17_monitoring` | LIA guidance, memory, prompt builder |
| 18 | `phase18_pipeline` | E2E stage chain; pinpoints failed stage |
| 19 | `phase19_regression` | Current vs baseline quality/latency |
| 20 | `phase20_reporting` | HTML / Markdown / JSON / console |

Each phase calls **real** `app.services.*` adapters under `evaluation/core/adapters/`.

## Datasets & goldens

- PDFs: `evaluation/datasets/` (symlinks to `over-all-test/chapter-*.pdf`)
- Goldens: `evaluation/goldens/<dataset>/*.json`
  - `pdf_extraction.json`, `images.json`, `tables.json`, `ocr.json`
  - `retrieval.json`, `ai_answers.json`, `topics.json`, `captions.json`
  - `lesson_plan.json`, `quiz.json`, `worksheet.json`, `homework.json`

Bootstrap goldens from a trusted run:

```bash
python evaluation/run_evaluation.py --phases pdf_extraction --bootstrap-goldens
```

Then hand-edit expected pages / phrases / retrieval cases.

## Reports & artifacts

After every run:

- `evaluation/reports/<run_id>/report.{json,md,html}` + `summary.txt`
- `evaluation/reports/latest.{json,md}`
- `evaluation/artifacts/<run_id>/…` — extracted images, chunks, retrieved context, tutor answers, lesson payloads
- `evaluation/baselines/latest.json` — regression baseline

## Metrics

Retrieval: Accuracy@k, Precision/Recall proxies, MRR, nDCG  
Images: blank / low-res / half-crop / duplicate / bbox  
Chunks: size, broken paragraph/list/table  
Tutor: groundedness, faithfulness, hallucination rate, curriculum alignment, safety  
Aggregate: **Overall AI Quality Score** (`evaluation/metrics/aggregate.py`)

## CI

GitHub Actions: `.github/workflows/ai-evaluation.yml`

Fails the job when critical checks fail, regressions fire, or overall score drops below `ci.fail_on_overall_below` (see `configs/thresholds.yaml`).

Env knobs:

| Env | Effect |
|-----|--------|
| `EVAL_SKIP_LLM=true` | Skip tutor/lesson LLM phases |
| `EVAL_SKIP_HEAVY_ML=true` | Skip PDF ML extraction |
| `EVAL_PRESET=smoke` | Phase preset |
| `EVAL_PHASES=a,b,c` | Explicit phase list |
| `EVAL_PLUGINS=evaluation.plugins.example_prompt_builder` | Extra phases |

## Extending (plugin)

1. Copy `plugins/example_prompt_builder.py`
2. `@register_phase` a new `Phase`
3. Load via `EVAL_PLUGINS=...` or add import in `discover_phases()`
4. Add goldens + pytest under `tests/`

## Exit codes (`run_evaluation.py`)

| Code | Meaning |
|------|---------|
| 0 | PASS |
| 1 | FAIL/ERROR |
| 2 | Critical check failed |
| 3 | Regression detected |
| 4 | Overall score below CI threshold |
