"""Student multimodal image understanding for the AI Tutor."""

# Lazy re-exports so validators/intent can be imported without loading service deps.

__all__ = [
    "understand",
    "save_upload",
    "delete_image",
    "UnderstandingBundle",
    "ImageUnderstandingResult",
]


def __getattr__(name: str):
    if name in ("understand", "save_upload", "delete_image"):
        from app.modules.image_understanding import service as _service

        return getattr(_service, name)
    if name in ("UnderstandingBundle", "ImageUnderstandingResult"):
        from app.modules.image_understanding import schemas as _schemas

        return getattr(_schemas, name)
    raise AttributeError(name)
