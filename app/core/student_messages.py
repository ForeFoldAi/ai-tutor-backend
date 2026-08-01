"""
Plain-language messages for students in API responses, tutor replies, and voice sessions.
"""

from __future__ import annotations

# Tutor / RAG
ANSWER_NOT_IN_CHAPTER = (
    "I couldn't find this in your chapter textbook. "
    "Try asking about a topic from the chapter you're studying, "
    "or open Learning Studio and pick the right chapter."
)

# Voice WebSocket
VOICE_ANSWER_FAILED = (
    "Sorry — I had trouble answering that. Please try again in a moment."
)
VOICE_SERVER_ERROR = (
    "Something went wrong on our side. Please close and reopen the voice session, "
    "or try again later."
)

# Voice REST
EMPTY_VOICE_MESSAGE = "Please say or type a question first."

# Textbook images (student catalog)
IMAGE_NOT_AVAILABLE = "This picture couldn't be loaded."
IMAGE_INVALID_PATH = "This picture isn't available."

# Uploads
PDF_ONLY = "Please upload a PDF file (textbook or notes)."

# Auth — student-facing
INVALID_LOGIN = "Wrong username or password. Please try again."
LOGIN_USE_USERNAME = (
    "More than one account shares that email or phone. "
    "Sign in with your username instead."
)
ACCOUNT_INACTIVE = (
    "Your account is paused. Please ask your teacher or school admin for help."
)
NOT_SIGNED_IN = "Your session ended. Please sign in again."
EMAIL_ALREADY_USED = (
    "This email is already in use. Try signing in or use a different email."
)
USERNAME_TAKEN = "That username is taken. Please pick another one."
WRONG_CURRENT_PASSWORD = "Your current password is wrong. Please try again."
CURRENT_PASSWORD_REQUIRED = (
    "Enter your current password before choosing a new one."
)
INVALID_VERIFY_TOKEN = (
    "This verification link is invalid or expired. Request a new one."
)
INVALID_RESET_TOKEN = (
    "This password reset link is invalid or expired. Request a new one."
)
INVALID_RESET_OTP = (
    "This password reset code is invalid or expired. Request a new one."
)

# Legacy alias — older clients may still receive this string
ANSWER_NOT_IN_DOCUMENT = ANSWER_NOT_IN_CHAPTER
