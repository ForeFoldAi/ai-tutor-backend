from __future__ import annotations

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: str
    content: str


class StudentAssistantChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    conversation_history: list[ConversationTurn] | None = None
    agent_mode: str = "free"


class StudentAssistantChatResponse(BaseModel):
    answer: str
    suggested_prompts: list[str] = Field(default_factory=list)


class StudentAssistantSuggestionsResponse(BaseModel):
    greeting: str
    suggested_prompts: list[str]
