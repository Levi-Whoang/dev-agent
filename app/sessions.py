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
from api.chat_helpers import chat_event_stream
from api.session_helpers import (
    database_user_id,
    resolve_chat_session,
    session_prefix,
    stored_session_id,
)
from auth.jwt_auth import get_current_user
from logs.logging_config import logger
from main import agent

router = APIRouter(tags=["会话"])


@router.post("/chat")
def chat(request: ChatRequest, current_user: str = Depends(get_current_user)):
    session_id, internal_session_id = resolve_chat_session(
        agent.session_manager,
        current_user,
        request.session_id,
        database_user_id(current_user),
        request.message,
    )
    return StreamingResponse(
        chat_event_stream(
            agent, logger, request.message, session_id, internal_session_id
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sessions/{session_id}/clear", response_model=APIResponse)
def clear_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id 不能为空")

    internal_session_id = stored_session_id(current_user, session_id)
    if not agent.session_manager.exists(internal_session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    agent.session_manager.clear(internal_session_id)
    return success_response(
        data={"session_id": session_id}, message="会话已清空"
    )


@router.post("/sessions", response_model=APIResponse)
def create_session(
    request: CreateSessionRequest,
    current_user: str = Depends(get_current_user),
):
    session_id = request.session_id or str(uuid4())
    internal_session_id = stored_session_id(current_user, session_id)
    owner_user_id = database_user_id(current_user)
    try:
        agent.session_manager.create(internal_session_id, owner_user_id)
    except Exception as error:
        raise HTTPException(
            status_code=409,
            detail="会话创建失败，可能是 session_id 已存在",
        ) from error
    return success_response(
        data={"session_id": session_id}, message="会话创建成功"
    )


@router.get("/sessions", response_model=APIResponse)
def list_sessions(current_user: str = Depends(get_current_user)):
    prefix = session_prefix(current_user)
    sessions = [
        {
            **session,
            "session_id": session["session_id"][len(prefix):],
        }
        for session in agent.session_manager.list_sessions(
            owner_user_id=database_user_id(current_user)
        )
        if session["session_id"].startswith(prefix)
    ]
    return success_response(data=sessions)


@router.get("/sessions/{session_id}/messages", response_model=APIResponse)
def get_session_messages(
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    internal_session_id = stored_session_id(current_user, session_id)
    if not agent.session_manager.exists(internal_session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = agent.session_manager.get_messages(internal_session_id)
    return success_response(
        data={"session_id": session_id, "messages": messages}
    )


@router.patch("/sessions/{session_id}/title", response_model=APIResponse)
def rename_session(
    session_id: str,
    request: SessionTitleRequest,
    current_user: str = Depends(get_current_user),
):
    title = " ".join(request.title.split())
    if not title:
        raise HTTPException(status_code=422, detail="会话名称不能为空")
    internal_session_id = stored_session_id(current_user, session_id)
    if not agent.session_manager.rename(
        internal_session_id, database_user_id(current_user), title
    ):
        raise HTTPException(status_code=404, detail="会话不存在")
    return success_response(
        data={"session_id": session_id, "title": title},
        message="会话名称已更新",
    )


@router.delete("/sessions/{session_id}", response_model=APIResponse)
def delete_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    internal_session_id = stored_session_id(current_user, session_id)
    if not agent.session_manager.exists(internal_session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    agent.session_manager.delete(internal_session_id)
    return success_response(
        data={"session_id": session_id}, message="会话删除成功"
    )
