"""User-safe errors for student image understanding."""

from __future__ import annotations


class ImageUnderstandingError(Exception):
    """Base error with a student-facing message."""

    def __init__(self, user_message: str, *, code: str = "image_error") -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.code = code


class InvalidImageError(ImageUnderstandingError):
    def __init__(self, user_message: str = "Please upload a valid JPG, PNG, or WEBP image.") -> None:
        super().__init__(user_message, code="invalid_image")


class ImageTooLargeError(ImageUnderstandingError):
    def __init__(self, user_message: str = "That image is too large. Please upload a smaller image.") -> None:
        super().__init__(user_message, code="image_too_large")


class ImageNotFoundError(ImageUnderstandingError):
    def __init__(
        self,
        user_message: str = "That image is no longer available. Please upload it again.",
    ) -> None:
        super().__init__(user_message, code="image_not_found")


class VisionProviderError(ImageUnderstandingError):
    def __init__(
        self,
        user_message: str = "I'm having trouble understanding the image right now. Please try again.",
    ) -> None:
        super().__init__(user_message, code="vision_failed")
