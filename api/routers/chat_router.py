"""聊天接口：普通返回 + SSE 流式返回。

流式为什么单独开一个数据库会话：
`async with get_session_factory()()` 写在生成器内部，生成器的生命周期
和 HTTP 响应完全对齐。若复用请求级依赖，容易踩到“响应还在写、会话已被关闭”
的时序问题（依赖清理时机随 FastAPI 版本变过）。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from agent.events import DONE, ERROR, AgentEvent
from agent.factory import build_agent
from agent.store import SessionStore
from api.api_response import success_response
from api.api_schemas import APIResponse, ChatRequest
from api.deps import CurrentUser, DbSession
from database.base import get_session_factory

logger = logging.getLogger("dev-agent.chat")

router = APIRouter(tags=["chat"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # 关掉 nginx 之类反代的缓冲，否则流式会被攒成一整块再吐出
    "X-Accel-Buffering": "no",
}


async def _ensure_session(db, user, session_id: str | None) -> str:
    store = SessionStore(db)
    if session_id:
        try:
            await store.get_or_create(session_id, user.id)
        except PermissionError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无权访问该会话",
            )
        return session_id

    new_id = str(uuid4())
    await store.get_or_create(new_id, user.id)
    return new_id


@router.post("/chat", response_model=APIResponse, summary="发送消息（等待完整回答）")
async def chat(payload: ChatRequest, db: DbSession, user: CurrentUser):
    session_id = await _ensure_session(db, user, payload.session_id)
    agent = build_agent(db, session_id)

    try:
        answer = await agent.run(payload.message)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    return success_response(
        data={"answer": answer, "session_id": session_id}
    )


@router.post("/chat/stream", summary="发送消息（SSE 流式返回）")
async def chat_stream(payload: ChatRequest, db: DbSession, user: CurrentUser):
    session_id = await _ensure_session(db, user, payload.session_id)
    await SessionStore(db).maybe_autotitle(session_id, payload.message)
    message = payload.message

    async def event_source() -> AsyncIterator[str]:
        # 响应头已经发出去了，此后的任何异常都只能变成一个 error 事件，
        # 否则客户端会看到一条“正常结束但什么都没收到”的空流
        try:
            async with get_session_factory()() as stream_db:
                agent = build_agent(stream_db, session_id)
                async for event in agent.stream(message):
                    yield event.to_sse()
        except Exception as exc:
            logger.exception("流式响应中断")
            yield AgentEvent(ERROR, {"message": str(exc)}).to_sse()
            yield AgentEvent(
                DONE, {"answer": "", "session_id": session_id, "error": True}
            ).to_sse()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream; charset=utf-8",
        headers={**SSE_HEADERS, "X-Session-Id": session_id},
    )
