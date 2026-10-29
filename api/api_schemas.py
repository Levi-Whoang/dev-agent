from pydantic import BaseModel, Field

from typing import Any

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=75
    )


class ChatResponse(BaseModel):
    answer: str | None
    session_id: str


class APIResponse(BaseModel):
    code: int
    message: str
    data: Any = None

class CreateSessionRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=75
    )


class SessionTitleRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)
