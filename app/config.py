import os

from dotenv import load_dotenv

# Get the project root directory (parent of 'app' directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
# NOTE: This backend used to rely on a local GGUF model via llama.cpp.
# We now use Mistral via the Mistral API for chat/RAG.
# Keep MODEL_PATH for backwards-compatibility / optional local fallback.
MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "mistral.gguf"))

# Mistral API settings (used by the /chat endpoint and voice fallback).
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_TEMPERATURE = float(os.environ.get("MISTRAL_TEMPERATURE", "0.35"))
MISTRAL_MAX_TOKENS = int(os.environ.get("MISTRAL_MAX_TOKENS", "1024"))
CHROMA_PATH = "chroma_db"
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")

# RAG chunking: ~512-token targets with ~10% overlap (tiktoken cl100k_base in pdf_service).
CHUNK_SIZE_TOKENS = int(os.environ.get("CHUNK_SIZE_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "51"))

# Top-k chunks sent to the LLM (typical practice: 3–5).
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "5"))

# Voice mode: smaller retrieval for conversational answers.
VOICE_RETRIEVAL_K = int(os.environ.get("VOICE_RETRIEVAL_K", "3"))
VOICE_CONTEXT_CHAR_BUDGET = int(os.environ.get("VOICE_CONTEXT_CHAR_BUDGET", "6000"))

# Rough cap on retrieved context size (~4k tokens; avoids huge prompts / latency).
CONTEXT_CHAR_BUDGET = int(os.environ.get("CONTEXT_CHAR_BUDGET", "14000"))

# Related textbook diagrams returned with chapter-aware answers (ranked).
TOP_RELATED_IMAGES = int(os.environ.get("TOP_RELATED_IMAGES", "3"))
# Minimum fused relevance score (weighted CLIP+page+keywords; typical good hit ≈ 22–32).
MIN_IMAGE_RELEVANCE_SCORE = float(os.environ.get("MIN_IMAGE_RELEVANCE_SCORE", "22"))
# Drop candidates scoring below this fraction of the top hit (dynamic count).
MIN_IMAGE_RELATIVE_TO_TOP = float(os.environ.get("MIN_IMAGE_RELATIVE_TO_TOP", "0.62"))
# CLIP cosine similarity floor when multimodal retrieval is enabled.
MIN_CLIP_IMAGE_SIMILARITY = float(os.environ.get("MIN_CLIP_IMAGE_SIMILARITY", "0.18"))
# Auto-named assets (Im0.jp2, p1_2.png) need stronger keyword overlap.
MIN_TOPIC_SCORE_GENERIC_ASSET = float(os.environ.get("MIN_TOPIC_SCORE_GENERIC_ASSET", "4"))
# Page proximity required for fallback when semantic gates return nothing.
MIN_PAGE_PROXIMITY_FALLBACK = float(os.environ.get("MIN_PAGE_PROXIMITY_FALLBACK", "10"))
# Start image search once the streamed answer reaches this length with a sentence end.
EARLY_IMAGE_MIN_CHARS = int(os.environ.get("EARLY_IMAGE_MIN_CHARS", "100"))

# Max seconds to wait for ranked images while the LLM answer is generated.
IMAGE_RETRIEVAL_TIMEOUT_SEC = float(os.environ.get("IMAGE_RETRIEVAL_TIMEOUT_SEC", "60"))

# Multimodal image index + retrieval (CLIP in Chroma, scoped by chapter upload IDs).
USE_MULTIMODAL_IMAGE_RETRIEVAL = os.environ.get("USE_MULTIMODAL_IMAGE_RETRIEVAL", "true").lower() in (
    "1",
    "true",
    "yes",
)
MULTIMODAL_IMAGE_MODEL = os.environ.get(
    "MULTIMODAL_IMAGE_MODEL",
    "openai/clip-vit-base-patch32",
)
MULTIMODAL_RETRIEVAL_K = int(os.environ.get("MULTIMODAL_RETRIEVAL_K", "24"))
IMAGE_COLLECTION_SUFFIX = os.environ.get("IMAGE_COLLECTION_SUFFIX", "_images")

# Hugging Face token — required to download/load CLIP (and optional BGE) weights.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# Pre-load CLIP in a background thread on startup (avoids first-request timeout).
WARM_MULTIMODAL_ON_STARTUP = os.environ.get("WARM_MULTIMODAL_ON_STARTUP", "true").lower() in (
    "1",
    "true",
    "yes",
)

# ---------------------------------------------------------------------------
# Precision pedagogy-centric retrieval thresholds
# ---------------------------------------------------------------------------

# Minimum final pedagogy score (0-100) to include a figure in the response.
# Symbolic-first formula: 0.35*concept + 0.25*section + 0.20*bge + 0.10*type + ...
MIN_FINAL_SCORE = float(os.environ.get("MIN_FINAL_SCORE", "40.0"))

# Multimodal path uses same floor (no longer 30% of heuristic threshold).
MIN_FINAL_SCORE_MULTIMODAL = float(os.environ.get("MIN_FINAL_SCORE_MULTIMODAL", "40.0"))

# Hard reject concept_specificity when combined with failed symbolic gates.
HARD_REJECT_SPECIFICITY = float(os.environ.get("HARD_REJECT_SPECIFICITY", "3.0"))

# Minimum topic_purity (0-1) for multi-word concepts when specificity is low.
MIN_TOPIC_PURITY = float(os.environ.get("MIN_TOPIC_PURITY", "0.25"))

# sidebar_example / decorative images need concept_specificity >= this to pass.
SIDEBAR_MIN_SPECIFICITY = float(os.environ.get("SIDEBAR_MIN_SPECIFICITY", "70.0"))

# Maximum number of primary images (always returned if above MIN_FINAL_SCORE).
MAX_PRIMARY_IMAGES = int(os.environ.get("MAX_PRIMARY_IMAGES", "2"))

# Extra images are included ONLY if their score >= top_score * this ratio.
# 0.92 means second image must score at least 92 % of the top image.
EXTRA_IMAGE_SCORE_RATIO = float(os.environ.get("EXTRA_IMAGE_SCORE_RATIO", "0.92"))

# MMR lambda: 1.0 = pure relevance, 0.0 = pure diversity.
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.70"))

# Topic-centric figure grounding (mandatory stage bypasses ranking; no score bonus)
MAX_GROUNDED_IMAGES = int(os.environ.get("MAX_GROUNDED_IMAGES", "3"))
# When true, page-proximity / citation fallbacks cannot rescue symbolically rejected images.
DISABLE_PAGE_PROXIMITY_FALLBACK = os.environ.get(
    "DISABLE_PAGE_PROXIMITY_FALLBACK", "true"
).lower() in ("1", "true", "yes")

# Visual intent gating (conversation_context.should_retrieve_images).
ENABLE_VISUAL_INTENT_DETECTION = os.environ.get(
    "ENABLE_VISUAL_INTENT_DETECTION", "true"
).lower() in ("1", "true", "yes")

# Frontend defers figure blocks until safe markdown boundaries (see stream-safe util).
ENABLE_STREAM_SAFE_INSERTION = os.environ.get(
    "ENABLE_STREAM_SAFE_INSERTION", "true"
).lower() in ("1", "true", "yes")

# Entity geographic conflict penalty magnitude (applied in _pedagogy_score).
ENTITY_CONFLICT_PENALTY = float(os.environ.get("ENTITY_CONFLICT_PENALTY", "25.0"))

# Enforce required-term gate in symbolic filters (see symbolic_image_filters).
REQUIRED_TERM_GATE_ENABLED = os.environ.get(
    "REQUIRED_TERM_GATE_ENABLED", "true"
).lower() in ("1", "true", "yes")

# Legacy aliases (kept for backward compat with existing env-var deployments)
MIN_PEDAGOGY_SCORE = MIN_FINAL_SCORE
MIN_CAPTION_BGE_SCORE = float(os.environ.get("MIN_CAPTION_BGE_SCORE", "0.0"))
MIN_CONCEPT_SCORE = float(os.environ.get("MIN_CONCEPT_SCORE", "0.0"))

# ---------------------------------------------------------------------------
# Storage backend
# ---------------------------------------------------------------------------

# "local" (default) or "s3"
STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "")
S3_REGION = os.environ.get("S3_REGION", "ap-south-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
# CDN prefix for public image URLs (e.g. "https://cdn.example.com/images")
CDN_BASE_URL = os.environ.get("CDN_BASE_URL", "")
# Thumbnail dimensions "WxH", e.g. "320x240". Empty = no thumbnails.
IMAGE_THUMB_SIZE = os.environ.get("IMAGE_THUMB_SIZE", "")

# ---------------------------------------------------------------------------
# OCR service
# ---------------------------------------------------------------------------

# Master on/off switch for OCR (applies to scanned PDF detection and per-image OCR)
OCR_ENABLED = os.environ.get("OCR_ENABLED", "true").lower() in ("1", "true", "yes")
# Tesseract language code(s) for pytesseract / PyMuPDF OCR
OCR_LANG = os.environ.get("OCR_LANG", "eng")
# Pages with fewer than this many characters are treated as scanned
OCR_MIN_CHARS_PER_PAGE = int(os.environ.get("OCR_MIN_CHARS_PER_PAGE", "80"))
# DPI for rasterising scanned pages before OCR
OCR_DPI = int(os.environ.get("OCR_DPI", "200"))

# Figure extraction: save page overlays (caption=red, candidates=green, final=blue)
DEBUG_FIGURE_BBOXES = os.environ.get("DEBUG_FIGURE_BBOXES", "false").lower() in (
    "1",
    "true",
    "yes",
)
