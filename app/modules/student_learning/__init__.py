"""Student learning analytics package."""

__all__ = ["router"]


def __getattr__(name: str):
    if name == "router":
        from app.modules.student_learning.router import router

        return router
    raise AttributeError(name)
