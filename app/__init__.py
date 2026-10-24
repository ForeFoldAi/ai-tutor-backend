import os
import tempfile

# Layout models import matplotlib inside indexing threads; GUI backends (macOS) fail off the main thread.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
