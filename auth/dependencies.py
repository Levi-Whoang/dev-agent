"""FastAPI 鉴权依赖：从 Authorization 头里取出当前用户。

`Depends(get_current_user)` 写在路由参数里，路由函数体里就再也看不到 token 解析，
也不需要每个接口自己判断“登录了没有”。
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from auth.service import get_user_by_id
from auth.tokens import TokenError, decode_access_token
from database.base import get_db
from database.models import User

# auto_error=False：自己控制 401 的响应体，保持跟统一响应格式一致
bearer_scheme = HTTPBearer(auto_error=False, description="Bearer <JWT>")


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None or not credentials.credentials:
        raise _unauthorized("请先登录")

    try:
        payload = decode_access_token(credentials.credentials)
    except TokenError as exc:
        raise _unauthorized(str(exc)) from exc

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _unauthorized("无效的登录凭证") from exc

    user = await get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        raise _unauthorized("用户不存在或已禁用")

    return user
