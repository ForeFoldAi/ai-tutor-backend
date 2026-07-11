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

# Edge TTS prosody — near-natural rate (was -6%; slightly faster = less robotic)
VOICE_TTS_RATE = os.environ.get("VOICE_TTS_RATE", "-2%")
VOICE_TTS_PITCH = os.environ.get("VOICE_TTS_PITCH", "+0Hz")

# Voice pipeline tuning (config-driven; override via env)
VOICE_TTS_PREFETCH = os.environ.get("VOICE_TTS_PREFETCH", "true").lower() in ("1", "true", "yes")
VOICE_TTS_WORKERS = int(os.environ.get("VOICE_TTS_WORKERS", "1"))  # prefetch is separate
VOICE_IDLE_FLUSH_SEC = float(os.environ.get("VOICE_IDLE_FLUSH_SEC", "0.12"))
VOICE_IDLE_FLUSH_STEADY_SEC = float(os.environ.get("VOICE_IDLE_FLUSH_STEADY_SEC", "0.22"))
VOICE_SPEECH_FIRST_WORDS = int(os.environ.get("VOICE_SPEECH_FIRST_WORDS", "3"))
VOICE_SPEECH_FIRST_CHARS = int(os.environ.get("VOICE_SPEECH_FIRST_CHARS", "14"))
VOICE_SPEECH_STEADY_WORDS = int(os.environ.get("VOICE_SPEECH_STEADY_WORDS", "10"))
VOICE_SPEECH_STEADY_CHARS = int(os.environ.get("VOICE_SPEECH_STEADY_CHARS", "48"))
VOICE_SPEECH_MAX_WORDS = int(os.environ.get("VOICE_SPEECH_MAX_WORDS", "16"))
VOICE_INTERRUPT_CANCEL_SEC = float(os.environ.get("VOICE_INTERRUPT_CANCEL_SEC", "0.45"))
# Max speech units waiting for TTS; 0 = unbounded (not recommended for production)
VOICE_SPEECH_QUEUE_MAXSIZE = int(os.environ.get("VOICE_SPEECH_QUEUE_MAXSIZE", "6"))

# Server-side Whisper STT (optional — requires faster-whisper)
VOICE_WHISPER_ENABLED = os.environ.get("VOICE_WHISPER_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# ---------------------------------------------------------------------------
# Voice protection (barge-in / false-interrupt hardening)
# Keep Browser STT + Edge TTS + WS; these gates only filter interrupts.
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: str = "true") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


VOICE_PROTECTION_ENABLED = _env_bool("VOICE_PROTECTION_ENABLED", "true")
VAD_ENABLED = _env_bool("VAD_ENABLED", "true")
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.75"))
MIN_SPEECH_MS = int(os.environ.get("MIN_SPEECH_MS", "300"))
POST_PLAYBACK_STT_DELAY_MS = int(os.environ.get("POST_PLAYBACK_STT_DELAY_MS", "1200"))
ECHO_SIMILARITY_THRESHOLD = float(os.environ.get("ECHO_SIMILARITY_THRESHOLD", "0.7"))
SPEAKER_VERIFICATION_ENABLED = _env_bool("SPEAKER_VERIFICATION_ENABLED", "true")
SPEAKER_SIMILARITY_THRESHOLD = float(os.environ.get("SPEAKER_SIMILARITY_THRESHOLD", "0.7"))
NOISE_SUPPRESSION_ENABLED = _env_bool("NOISE_SUPPRESSION_ENABLED", "true")
NOISE_SUPPRESSION_PROVIDER = os.environ.get("NOISE_SUPPRESSION_PROVIDER", "rnnoise").strip().lower()
WAKE_WORD_ENABLED = _env_bool("WAKE_WORD_ENABLED", "false")
WAKE_WORDS = [
    w.strip().lower()
    for w in os.environ.get("WAKE_WORDS", "hey tutor,teacher").split(",")
    if w.strip()
]

# Confidence-based barge-in (weights must sum ~1.0)
INTERRUPT_SCORE_THRESHOLD = float(os.environ.get("INTERRUPT_SCORE_THRESHOLD", "0.55"))
INTERRUPT_WEIGHT_VAD = float(os.environ.get("INTERRUPT_WEIGHT_VAD", "0.4"))
INTERRUPT_WEIGHT_SPEAKER = float(os.environ.get("INTERRUPT_WEIGHT_SPEAKER", "0.4"))
INTERRUPT_WEIGHT_INTENT = float(os.environ.get("INTERRUPT_WEIGHT_INTENT", "0.2"))

# TTS continuity
VOICE_TTS_PREFETCH_DEPTH = int(os.environ.get("VOICE_TTS_PREFETCH_DEPTH", "2"))
VOICE_TTS_LOOKAHEAD_CHARS = int(os.environ.get("VOICE_TTS_LOOKAHEAD_CHARS", "24"))
VOICE_PROTECTION_PRELOAD = _env_bool("VOICE_PROTECTION_PRELOAD", "true")

CHROMA_PATH = "chroma_db"
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")

# RAG chunking: ~512-token targets with ~10% overlap (tiktoken cl100k_base).
CHUNK_SIZE_TOKENS = int(os.environ.get("CHUNK_SIZE_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "51"))

# Redis-backed tutor Q&A cache (LLM answers). Off by default — every question hits the LLM.
TUTOR_ANSWER_CACHE_ENABLED = _env_bool("TUTOR_ANSWER_CACHE_ENABLED", "false")

# Top-k chunks sent to the LLM (typical practice: 3–5).
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "5"))

# Voice mode: smaller retrieval for conversational answers.
VOICE_RETRIEVAL_K = int(os.environ.get("VOICE_RETRIEVAL_K", "3"))
VOICE_CONTEXT_CHAR_BUDGET = int(os.environ.get("VOICE_CONTEXT_CHAR_BUDGET", "6000"))
VOICE_MAX_TOKENS = int(os.environ.get("VOICE_MAX_TOKENS", "120"))
VOICE_EARLY_IMAGE_MIN_CHARS = int(os.environ.get("VOICE_EARLY_IMAGE_MIN_CHARS", "40"))
# Spoken-turn word caps injected into voice prompts (not hard token limits)
VOICE_PROMPT_MAX_WORDS = int(os.environ.get("VOICE_PROMPT_MAX_WORDS", "75"))
VOICE_PROMPT_EXPAND_WORDS = int(os.environ.get("VOICE_PROMPT_EXPAND_WORDS", "100"))
# Phase 6 — cold-start overlap + HTTP fallback TTS prefetch
VOICE_EMBEDDING_WARMUP = os.environ.get("VOICE_EMBEDDING_WARMUP", "true").lower() in (
    "1",
    "true",
    "yes",
)
VOICE_RAG_PREWARM = os.environ.get("VOICE_RAG_PREWARM", "true").lower() in ("1", "true", "yes")
VOICE_HTTP_TTS_PREFETCH = os.environ.get("VOICE_HTTP_TTS_PREFETCH", "true").lower() in (
    "1",
    "true",
    "yes",
)

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

# Vision captioning for figures with weak PDF/OCR captions (BLIP via Transformers).
VISION_CAPTION_ENABLED = os.environ.get("VISION_CAPTION_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
VISION_CAPTION_MODEL = os.environ.get(
    "VISION_CAPTION_MODEL",
    "Salesforce/blip-image-captioning-base",
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

# ---------------------------------------------------------------------------
# Native multi-model PDF extraction (layout → formula → OCR → table VLM)
# ---------------------------------------------------------------------------

PDF_EXTRACTION_MODELS_DIR = os.environ.get(
    "PDF_EXTRACTION_MODELS_DIR",
    os.environ.get(
        "PDF_EXTRACT_KIT_ROOT",
        os.path.join(PROJECT_ROOT, "models"),
    ),
)
PDF_EXTRACTION_PIPELINE_ENABLED = os.environ.get(
    "PDF_EXTRACTION_PIPELINE_ENABLED",
    os.environ.get("PDF_EXTRACT_PIPELINE_ENABLED", "true"),
).lower() in ("1", "true", "yes")
PDF_EXTRACTION_CONFIG_PATH = os.environ.get(
    "PDF_EXTRACTION_CONFIG_PATH",
    os.environ.get(
        "PDF_EXTRACT_CONFIG_PATH",
        os.path.join(PROJECT_ROOT, "configs", "pdf_extraction_pipeline.yaml"),
    ),
)
PDF_EXTRACTION_DPI = int(
    os.environ.get("PDF_EXTRACTION_DPI", os.environ.get("PDF_EXTRACT_DPI", "144"))
)
PDF_EXTRACTION_ENABLE_TABLE_VLM = os.environ.get(
    "PDF_EXTRACTION_ENABLE_TABLE_VLM",
    os.environ.get("PDF_EXTRACT_ENABLE_TABLE_VLM", "false"),
).lower() in ("1", "true", "yes")
