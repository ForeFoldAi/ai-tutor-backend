"""CLIP encoder for textbook figure indexing and retrieval (Transformers)."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from app.config import HF_TOKEN, MULTIMODAL_IMAGE_MODEL

logger = logging.getLogger(__name__)

_clip_model: Any = None
_clip_processor: Any = None
_loaded_name: str | None = None


def clip_model_available() -> bool:
    try:
        return _load_clip() is not None
    except Exception:
        return False


def _load_clip() -> tuple[Any, Any] | None:
    global _clip_model, _clip_processor, _loaded_name
    if _clip_model is not None and _clip_processor is not None:
        return _clip_model, _clip_processor
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        name = MULTIMODAL_IMAGE_MODEL
        token = HF_TOKEN
        if not token:
            logger.warning(
                "HF_TOKEN is not set — CLIP download may fail with 403. "
                "Add HF_TOKEN to ai-tutor-backend/.env (https://huggingface.co/settings/tokens)."
            )
        logger.info("Loading CLIP model: %s", name)
        kwargs = {"token": token} if token else {}
        _clip_processor = CLIPProcessor.from_pretrained(name, **kwargs)
        _clip_model = CLIPModel.from_pretrained(name, **kwargs)
        _clip_model.eval()
        _loaded_name = name
        # Avoid CPU thread oversubscription on small batches
        torch.set_num_threads(min(4, torch.get_num_threads()))
        return _clip_model, _clip_processor
    except Exception as exc:
        logger.warning("CLIP model unavailable (%s): %s", MULTIMODAL_IMAGE_MODEL, exc)
        _clip_model = None
        _clip_processor = None
        return None


def get_clip_model():
    pair = _load_clip()
    return pair[0] if pair else None


def current_model_name() -> str:
    return _loaded_name or MULTIMODAL_IMAGE_MODEL


def _normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v
    return v / n


def _features_tensor(model: Any, raw: Any, *, branch: str) -> Any:
    """Transformers 5.x may return tensors or pooled model outputs from get_*_features."""
    import torch

    if isinstance(raw, torch.Tensor):
        feats = raw
    elif hasattr(raw, "pooler_output") and raw.pooler_output is not None:
        feats = raw.pooler_output
    elif hasattr(raw, "last_hidden_state"):
        feats = raw.last_hidden_state[:, 0, :]
    else:
        raise TypeError(f"Unexpected CLIP {branch} output type: {type(raw)}")

    proj = getattr(model, "text_projection" if branch == "text" else "visual_projection", None)
    # Transformers 5.x get_*_features may already return projected embeddings (e.g. 512-dim).
    if proj is not None and feats.shape[-1] == proj.in_features:
        feats = proj(feats)
    return feats


def encode_image_file(path: str) -> np.ndarray | None:
    if not path or not os.path.isfile(path):
        return None
    loaded = _load_clip()
    if loaded is None:
        return None
    model, processor = loaded
    try:
        import torch
        from PIL import Image

        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            raw = model.get_image_features(**inputs)
            feats = _features_tensor(model, raw, branch="image")
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return _normalize(feats[0].cpu().numpy())
    except Exception as exc:
        logger.debug("encode_image_file failed for %s: %s", path, exc)
        return None


def encode_text(text: str) -> np.ndarray | None:
    t = (text or "").strip()
    if not t:
        return None
    loaded = _load_clip()
    if loaded is None:
        return None
    model, processor = loaded
    try:
        import torch

        inputs = processor(
            text=[t[:4000]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            raw = model.get_text_features(**inputs)
            feats = _features_tensor(model, raw, branch="text")
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return _normalize(feats[0].cpu().numpy())
    except Exception as exc:
        logger.debug("encode_text failed: %s", exc)
        return None
