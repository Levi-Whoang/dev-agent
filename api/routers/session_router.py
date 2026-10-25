"""会话管理：列表、创建、重命名、消息、清空、删除。

所有操作都以当前登录用户为边界——查别人的 session_id 只会得到 404，
而不是 403。避免攻击者靠状态码差异探测“这个会话存在”。
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from agent.store import SessionStore
from api.api_response import success_response
from api.api_schemas import APIResponse, CreateSessionRequest, RenameSessionRequest
from api.deps import CurrentUser, DbSession

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=APIResponse, summary="当前用户的会话列表")
async def list_sessions(db: DbSession, user: CurrentUser):
    store = SessionStore(db)
    return success_response(data=await store.list_sessions(user.id))


@router.post("", response_model=APIResponse, summary="新建会话")
async def create_session(
    payload: CreateSessionRequest, db: DbSession, user: CurrentUser
):
    session_id = payload.session_id or str(uuid4())
    store = SessionStore(db)

    try:
        await store.get_or_create(session_id, user.id, payload.title)
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="该 session_id 已被占用",
        )

    return success_response(
        data={"session_id": session_id}, message="会话创建成功"
    )


@router.patch("/{session_id}", response_model=APIResponse, summary="重命名会话")
async def rename_session(
    session_id: str,
    payload: RenameSessionRequest,
    db: DbSession,
    user: CurrentUser,
):
    store = SessionStore(db)
    if not await store.rename(session_id, user.id, payload.title):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在"
        )

    return success_response(
        data={"session_id": session_id, "title": payload.title},
        message="会话已重命名",
    )


@router.get("/{session_id}/messages", response_model=APIResponse)
async def get_messages(session_id: str, db: DbSession, user: CurrentUser):
    store = SessionStore(db)
    if not await store.exists(session_id, user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在"
        )

    return success_response(
        data={
            "session_id": session_id,
            "messages": await store.messages_for_api(session_id),
        }
    )


@router.post("/{session_id}/clear", response_model=APIResponse)
async def clear_session(session_id: str, db: DbSession, user: CurrentUser):
    store = SessionStore(db)
    if not await store.exists(session_id, user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在"
        )

    await store.clear(session_id)
    return success_response(
        data={"session_id": session_id}, message="会话已清空"
    )


@router.delete("/{session_id}", response_model=APIResponse)
async def delete_session(session_id: str, db: DbSession, user: CurrentUser):
    store = SessionStore(db)
    if not await store.delete(session_id, user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在"
        )

    return success_response(
        data={"session_id": session_id}, message="会话删除成功"
    )
