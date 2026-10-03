from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.constants import Role
from app.modules.auth.dependencies import require_roles
from app.modules.image_understanding.errors import ImageUnderstandingError
from app.modules.student_assistant import service as assistant_service
from app.modules.student_assistant.schemas import (
    StudentAssistantChatRequest,
    StudentAssistantChatResponse,
    StudentAssistantSuggestionsResponse,
)
from app.modules.users.models import User

router = APIRouter(prefix="/auth/student/assistant", tags=["student-assistant"])


def _history(payload: StudentAssistantChatRequest) -> list[dict[str, str]] | None:
    if not payload.conversation_history:
        return None
    return [{"role": t.role, "content": t.content} for t in payload.conversation_history]


@router.get("/suggestions", response_model=StudentAssistantSuggestionsResponse)
def assistant_suggestions(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.STUDENT))],
    agent_mode: Annotated[str, Query()] = "free",
):
    mode = assistant_service.normalize_agent_mode(agent_mode)
    return StudentAssistantSuggestionsResponse(
        greeting=assistant_service.build_greeting(current_user),
        suggested_prompts=assistant_service.mode_suggested_prompts(db, current_user, mode),
    )


@router.post("/chat", response_model=StudentAssistantChatResponse)
async def assistant_chat(
    payload: StudentAssistantChatRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.STUDENT))],
):
    try:
        answer, prompts = await assistant_service.chat(
            db,
            current_user,
            query=payload.query or "",
            conversation_history=_history(payload),
            agent_mode=payload.agent_mode,
            image_ids=payload.image_ids,
        )
    except ImageUnderstandingError as exc:
        raise HTTPException(status_code=400, detail=exc.user_message) from None
    return StudentAssistantChatResponse(answer=answer, suggested_prompts=prompts)


@router.post("/chat/stream")
async def assistant_chat_stream(
    payload: StudentAssistantChatRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_roles(Role.STUDENT))],
):
    history = _history(payload)

    async def ndjson_generator():
        try:
            async for token in assistant_service.chat_stream(
                db,
                current_user,
                query=payload.query or "",
                conversation_history=history,
                agent_mode=payload.agent_mode,
                image_ids=payload.image_ids,
            ):
                if isinstance(token, str) and token.startswith("\0STATUS:"):
                    rest = token[len("\0STATUS:") :]
                    if rest.startswith("image_processing"):
                        yield (
                            json.dumps({"type": "image_processing"}, ensure_ascii=False) + "\n"
                        ).encode()
                    elif rest.startswith("image_understood:"):
                        parts = rest.split(":", 3)
                        # image_understood:type:intent:confidence
                        payload_out: dict = {"type": "image_understood"}
                        if len(parts) >= 4:
                            payload_out["image_type"] = parts[1]
                            payload_out["intent"] = parts[2]
                            try:
                                payload_out["confidence"] = float(parts[3])
                            except ValueError:
                                pass
                        yield (json.dumps(payload_out, ensure_ascii=False) + "\n").encode()
                    continue
                yield (
                    json.dumps({"type": "token", "content": token}, ensure_ascii=False) + "\n"
                ).encode()
            yield (json.dumps({"type": "done"}, ensure_ascii=False) + "\n").encode()
        except ImageUnderstandingError as exc:
            yield (
                json.dumps({"type": "error", "content": exc.user_message}, ensure_ascii=False) + "\n"
            ).encode()
            yield (
                json.dumps({"type": "token", "content": exc.user_message}, ensure_ascii=False) + "\n"
            ).encode()
            yield (json.dumps({"type": "done"}, ensure_ascii=False) + "\n").encode()

    return StreamingResponse(
        ndjson_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )
