"""JWT 的签发与校验。

为什么用 JWT 而不是“服务端存 session_id 到 cookie”：
- 服务端不用保存登录状态，多实例部署时不需要共享 session 存储；
- 前端把 token 放 Authorization 头里，跨域和移动端都好处理。

代价是：签出去的 token 在过期前无法单独作废（改 JWT_SECRET 会让所有 token 失效），
所以过期时间不能设太长。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from config import get_settings


class TokenError(Exception):
    """token 无效或过期。路由层把它翻译成 401。"""


def create_access_token(
    subject: str,
    extra_claims: dict[str, Any] | None = None,
    expires_minutes: int | None = None,
) -> tuple[str, int]:
    """返回 (token, 有效期秒数)，前端可以据此决定什么时候刷新。"""
    settings = get_settings()
    minutes = expires_minutes or settings.jwt_expire_minutes
    now = datetime.now(timezone.utc)

    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    if extra_claims:
        # 不允许外部覆盖 sub/exp，否则等于让调用方自己延长登录
        payload.update(
            {k: v for k, v in extra_claims.items() if k not in {"sub", "exp", "iat"}}
        )

    token = jwt.encode(
        payload, settings.jwt_secret, algorithm=settings.jwt_algorithm
    )
    return token, minutes * 60


def decode_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("登录已过期，请重新登录") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("无效的登录凭证") from exc
