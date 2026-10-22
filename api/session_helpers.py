"""封装用户会话命名空间和会话初始化逻辑。"""

import hashlib
from uuid import uuid4

from fastapi import HTTPException

SESSION_ID_MAX_LENGTH = 75


def session_prefix(user_id: str) -> str:
    """为认证用户生成不暴露身份的会话命名空间前缀。"""
    owner_key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
    return f"{owner_key}:"


def stored_session_id(user_id: str, session_id: str) -> str:
    """校验公开会话 ID，并附加用户命名空间前缀。"""
    if not session_id or len(session_id) > SESSION_ID_MAX_LENGTH:
        raise HTTPException(status_code=422, detail="session_id 长度无效")
    return f"{session_prefix(user_id)}{session_id}"


def database_user_id(username: str) -> int:
    """根据当前认证用户名读取用户表主键，供会话外键使用。"""
    from auth.jwt_auth import auth_repository

    user = auth_repository.find_user(username)
    if user is None:
        raise HTTPException(status_code=401, detail="认证用户不存在")
    return int(user["id"])


def resolve_chat_session(
    session_manager,
    user_id: str,
    requested_session_id: str | None,
    owner_user_id: int,
    initial_title: str,
):
    """选择本次聊天的公开 ID，并确保用户专属会话已创建。"""
    session_id = requested_session_id or uuid4().hex
    internal_id = stored_session_id(user_id, session_id)
    if not session_manager.exists(internal_id):
        title = " ".join(initial_title.split())[:120] or "新对话"
        session_manager.create(internal_id, owner_user_id, title)
    return session_id, internal_id
