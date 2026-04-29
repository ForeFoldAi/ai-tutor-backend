import os

# Get the project root directory (parent of 'app' directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NOTE: This backend used to rely on a local GGUF model via llama.cpp.
# We now use Mistral via the Mistral API for chat/RAG.
# Keep MODEL_PATH for backwards-compatibility / optional local fallback.
MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "mistral.gguf"))

# Mistral API settings (used by the /chat endpoint and voice fallback).
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_TEMPERATURE = float(os.environ.get("MISTRAL_TEMPERATURE", "0.1"))
MISTRAL_MAX_TOKENS = int(os.environ.get("MISTRAL_MAX_TOKENS", "256"))
CHROMA_PATH = "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_K = 2
