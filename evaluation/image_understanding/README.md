# Image Understanding Evaluation Fixtures

Soft-match expectations for multimodal tutor image analysis (not exact OCR strings).

| # | Fixture | Expected image_type | Topic hints | Default intent | Expected route |
|---|---------|---------------------|-------------|----------------|----------------|
| 1 | printed_textbook_question | textbook_question | any subject topic | answer / explain | tesseract |
| 2 | handwritten_question | handwritten_notes or handwritten_homework | — | answer | vision |
| 3 | handwritten_math | math_problem | algebra / arithmetic | solve | vision (+ SymPy) |
| 4 | geometry_diagram | diagram or math_problem | triangle / geometry | explain_diagram / solve | vision |
| 5 | science_diagram | science_diagram or diagram | biology / physics | explain_diagram | vision |
| 6 | graph | graph or chart | axes / plot | explain_graph | vision |
| 7 | table | mixed or textbook_page | table | explain_table | vision |
| 8 | textbook_paragraph | textbook_page | chapter topic | explain / summarize | tesseract |
| 9 | question_paper | question_paper | exam | answer | tesseract (vision if math present) |
| 10 | student_answer | answer_sheet or handwritten_homework | — | check_answer | vision |
| 11 | map | map | geography | explain | vision |
| 12 | poor_quality (blurred / dark) | any | — | — | vision; blank / tiny → unclear |
| 13 | mixed printed + handwritten | mixed_educational_content | — | answer | vision |

Place sample images under this folder when running live vision evals. Prefer synthetic or self-made
images (no copyrighted textbook pages).

Automated tests (no vision API calls):

- `tests/test_image_understanding.py`: validators, intent, schemas, RAG query, EXIF.
- `tests/test_image_ocr_routing.py`: quality, OCR parsing, router, orchestration, retries, cache, fallbacks.
  It uses synthetic Pillow fixtures and hand-built Tesseract output.
- `tests/test_image_ocr_tesseract_integration.py`: real Tesseract on synthetic Pillow images. It is skipped
  when the `tesseract` binary is not installed.
