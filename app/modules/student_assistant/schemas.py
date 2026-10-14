from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ConversationTurn(BaseModel):
    role: str
    content: str


class StudentAssistantChatRequest(BaseModel):
    query: str = Field(default="", max_length=4000)
    conversation_history: list[ConversationTurn] | None = None
    agent_mode: str = "free"
    image_ids: list[str] | None = None

    @model_validator(mode="after")
    def _require_query_or_image(self) -> StudentAssistantChatRequest:
        if not (self.query or "").strip() and not (self.image_ids or []):
            raise ValueError("Please type a question or attach an image.")
        return self


class StudentAssistantChatResponse(BaseModel):
    answer: str
    suggested_prompts: list[str] = Field(default_factory=list)


class StudentAssistantSuggestionsResponse(BaseModel):
    greeting: str
    suggested_prompts: list[str]
