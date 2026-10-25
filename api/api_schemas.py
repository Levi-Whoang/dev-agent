"""请求 / 响应模型（Pydantic 负责校验，路由里不用手写 if 判断）。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    # 不给就新建一个会话，省得前端先调一次 /sessions
    session_id: str | None = Field(default=None, min_length=1, max_length=100)


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class CreateSessionRequest(BaseModel):
    session_id: str | None = Field(default=None, min_length=1, max_length=100)
    title: str | None = Field(default=None, max_length=200)


class RenameSessionRequest(BaseModel):
    """重命名会话。

    min_length 拦不住 "   " 这种只有空白的标题，所以在校验器里再洗一遍：
    压掉换行和连续空格，洗完还是空就报错。校验器在约束之后执行，
    所以超长的标题会先在 max_length 那里被拒。
    """

    title: str = Field(min_length=1, max_length=60)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("标题不能为空")
        return cleaned


class APIResponse(BaseModel):
    code: int
    message: str
    data: Any = None


class HealthResponse(BaseModel):
    status: str
    database: str
    knowledge_chunks: int
