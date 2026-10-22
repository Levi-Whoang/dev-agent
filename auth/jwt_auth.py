"""负责登录凭据校验、JWT 签发和 FastAPI 用户身份解析。"""

from datetime import datetime, timedelta, timezone
import json
import os
from uuid import uuid4

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.passwords import verify_password

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "dev-agent"
bearer_scheme = HTTPBearer(auto_error=False)


def _load_auth_config() -> tuple[dict[str, str], str, int]:
    raw_users = os.getenv("AUTH_USERS_JSON", "").strip()
    secret = os.getenv("JWT_SECRET_KEY", "")
    try:
        users = json.loads(raw_users)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=503, detail="认证用户配置无效") from error

    if (
        not isinstance(users, dict)
        or not users
        or any(
            not isinstance(username, str)
            or not username.strip()
            or not isinstance(password_hash, str)
            for username, password_hash in users.items()
        )
    ):
        raise HTTPException(status_code=503, detail="认证用户配置无效")
    if len(secret) < 32:
        raise HTTPException(status_code=503, detail="JWT 密钥未正确配置")
    try:
        expires_minutes = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))
    except ValueError as error:
        raise HTTPException(status_code=503, detail="JWT 有效期配置无效") from error
    if not 1 <= expires_minutes <= 1440:
        raise HTTPException(status_code=503, detail="JWT 有效期配置无效")
    return users, secret, expires_minutes


def authenticate_user(username: str, password: str) -> bool:
    users, _, _ = _load_auth_config()
    password_hash = users.get(username)
    return bool(password_hash and verify_password(password, password_hash))


def create_access_token(username: str) -> tuple[str, int]:
    _, secret, expires_minutes = _load_auth_config()
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=expires_minutes)
    token = jwt.encode(
        {
            "sub": username,
            "iss": JWT_ISSUER,
            "iat": now,
            "exp": expires_at,
            "jti": uuid4().hex,
        },
        secret,
        algorithm=JWT_ALGORITHM,
    )
    return token, expires_minutes * 60


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    users, secret, _ = _load_auth_config()
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="需要 Bearer JWT 认证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            secret,
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            options={"require": ["sub", "iss", "iat", "exp"]},
        )
    except jwt.PyJWTError as error:
        raise HTTPException(
            status_code=401,
            detail="JWT 无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error
    username = payload.get("sub")
    if not isinstance(username, str) or username not in users:
        raise HTTPException(
            status_code=401,
            detail="JWT 用户无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username
