"""聊天、会话读取、修改、清理和删除接口。"""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.api_response import success_response
from api.api_schemas import (
    APIResponse,
    ChatRequest,
    CreateSessionRequest,
    SessionTitleRequest,
)
from api.chat_helpers import managed_chat_stream
from agent.control import begin_generation, stop_generation
from api.session_helpers import (
    database_user_id,
    resolve_chat_session,
    session_prefix,
    stored_session_id,
)
from auth.jwt_auth import get_current_user
from logs.logging_config import logger
from agent.runtime import agent

router = APIRouter(tags=["会话"])


# 创建或读取当前用户会话并返回流式聊天响应。
@router.post("/chat")
async def chat(request: ChatRequest, current_user: str = Depends(get_current_user)):
    owner_user_id = await database_user_id(current_user)
    session_id, internal_session_id = await resolve_chat_session(
        agent.session_manager,
        current_user,
        request.session_id,
        owner_user_id,
        request.message,
    )
    try:
        event = begin_generation(internal_session_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return StreamingResponse(
        managed_chat_stream(
            agent, logger, request.message, session_id, internal_session_id,
            owner_user_id, event,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 清空当前用户指定会话中的消息。
@router.post("/sessions/{session_id}/clear", response_model=APIResponse)
async def clear_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    internal_session_id = stored_session_id(current_user, session_id)
    if not ((await agent.session_manager.exists(internal_session_id))):
        raise HTTPException(status_code=404, detail="会话不存在")
    await agent.session_manager.clear(internal_session_id)
    return success_response(
        data={"session_id": session_id}, message="会话已清空"
    )


# 为当前登录用户创建新会话。
@router.post("/sessions", response_model=APIResponse)
async def create_session(
    request: CreateSessionRequest,
    current_user: str = Depends(get_current_user),
):
    session_id = request.session_id or str(uuid4())
    internal_session_id = stored_session_id(current_user, session_id)
    owner_user_id = await database_user_id(current_user)
    try:
        await agent.session_manager.create(internal_session_id, owner_user_id)
    except Exception as error:
        raise HTTPException(
            status_code=409,
            detail="会话创建失败，可能是 session_id 已存在",
        ) from error
    return success_response(
        data={"session_id": session_id}, message="会话创建成功"
    )


# 列出当前登录用户拥有的会话。
@router.get("/sessions", response_model=APIResponse)
async def list_sessions(current_user: str = Depends(get_current_user)):
    prefix = session_prefix(current_user)
    sessions = [
        {
            **session,
            "session_id": session["session_id"][len(prefix):],
        }
        for session in ((await agent.session_manager.list_sessions(
            owner_user_id=((await database_user_id(current_user)))
        )))
        if session["session_id"].startswith(prefix)
    ]
    return success_response(data=sessions)


# 读取当前用户指定会话的消息历史。
@router.get("/sessions/{session_id}/messages", response_model=APIResponse)
async def get_session_messages(
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    internal_session_id = stored_session_id(current_user, session_id)
    if not ((await agent.session_manager.exists(internal_session_id))):
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = await agent.session_manager.get_messages(internal_session_id)
    return success_response(
        data={"session_id": session_id, "messages": messages}
    )


# 修改当前用户指定会话的标题。
@router.patch("/sessions/{session_id}/title", response_model=APIResponse)
async def rename_session(
    session_id: str,
    request: SessionTitleRequest,
    current_user: str = Depends(get_current_user),
):
    title = " ".join(request.title.split())
    if not title:
        raise HTTPException(status_code=422, detail="会话名称不能为空")
    internal_session_id = stored_session_id(current_user, session_id)
    if not ((await agent.session_manager.rename(
        internal_session_id, ((await database_user_id(current_user))), title
    ))):
        raise HTTPException(status_code=404, detail="会话不存在")
    return success_response(
        data={"session_id": session_id, "title": title},
        message="会话名称已更新",
    )


# 删除当前用户指定的会话及其历史。
@router.delete("/sessions/{session_id}", response_model=APIResponse)
async def delete_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    internal_session_id = stored_session_id(current_user, session_id)
    if not ((await agent.session_manager.exists(internal_session_id))):
        raise HTTPException(status_code=404, detail="会话不存在")
    await agent.session_manager.delete(internal_session_id)
    return success_response(
        data={"session_id": session_id}, message="会话删除成功"
    )


# 只能停止当前登录用户自己的会话生成任务。
@router.post("/sessions/{session_id}/stop", response_model=APIResponse)
async def stop_session_generation(session_id: str, current_user: str = Depends(get_current_user)):
    internal_id = stored_session_id(current_user, session_id)
    if not ((await agent.session_manager.exists(internal_id))):
        raise HTTPException(status_code=404, detail="会话不存在")
    stopped = stop_generation(internal_id)
    logger.info("generation_stop session_id=%s active=%s", session_id, stopped)
    return success_response({"stopped": stopped})
