"""Interactive mathematics lesson generation with visualization specs."""

from app.services.math_lesson.schemas import MathLesson, VisualizationSpec
from app.services.math_lesson.fallbacks import query_requests_animation
from app.services.math_lesson.service import (
    clean_tutor_answer_text,
    extract_math_lesson_from_answer,
    finalize_math_answer,
    get_visualization_appendix_prompt,
    should_use_interactive_math_lesson,
    strip_math_lesson_block,
)

__all__ = [
    "MathLesson",
    "VisualizationSpec",
    "extract_math_lesson_from_answer",
    "finalize_math_answer",
    "get_visualization_appendix_prompt",
    "query_requests_animation",
    "should_use_interactive_math_lesson",
    "clean_tutor_answer_text",
    "strip_math_lesson_block",
]
