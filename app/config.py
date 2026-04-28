import os

# Get the project root directory (parent of 'app' directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Override with env var if you keep GGUF elsewhere.
MODEL_PATH = os.environ.get(
    "LLAMA_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "mistral.gguf"),
)
CHROMA_PATH = "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_K = 2
