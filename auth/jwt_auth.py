"""处理注册、登录、JWT 签发和当前用户解析。"""

from datetime import datetime, timedelta, timezone
import os
from uuid import uuid4

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.passwords import hash_password, verify_password
from auth.repository import DuplicateUsernameError, auth_repository

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "dev-agent"
JWT_EXPIRE_MINUTES = 30
bearer_scheme = HTTPBearer(auto_error=False)


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "")
    if len(secret) < 32:
        raise HTTPException(status_code=503, detail="JWT 密钥未正确配置")
    return secret


def normalize_username(username: str) -> str:
    return username.strip().lower()


def register_user(username: str, password: str) -> dict:
    _jwt_secret()
    normalized_username = normalize_username(username)
    password_hash = hash_password(password)
    try:
        user_id = auth_repository.create_user(normalized_username, password_hash)
    except DuplicateUsernameError as error:
        raise HTTPException(status_code=409, detail="用户名已被注册") from error
    return {"user_id": user_id, "username": normalized_username}


def create_access_token(username: str, user_id: int) -> tuple[str, int]:
    secret = _jwt_secret()
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=JWT_EXPIRE_MINUTES)
    jti = uuid4().hex
    token = jwt.encode(
        {
            "sub": username,
            "iss": JWT_ISSUER,
            "iat": now,
            "exp": expires_at,
            "jti": jti,
        },
        secret,
        algorithm=JWT_ALGORITHM,
    )
    auth_repository.create_token(jti, user_id, expires_at)
    return token, JWT_EXPIRE_MINUTES * 60


def login_user(username: str, password: str) -> tuple[str, int] | None:
    normalized_username = normalize_username(username)
    user = auth_repository.find_user(normalized_username)
    if not user or not verify_password(password, user["password_hash"]):
        return None
    return create_access_token(user["username"], int(user["id"]))


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            _jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            options={"require": ["sub", "iss", "iat", "exp", "jti"]},
        )
    except jwt.PyJWTError as error:
        raise HTTPException(
            status_code=401,
            detail="JWT 无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="需要 Bearer JWT 认证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_token(credentials.credentials)
    username = payload.get("sub")
    jti = payload.get("jti")
    if (
        not isinstance(username, str)
        or not isinstance(jti, str)
        or not auth_repository.is_token_active(jti, username)
    ):
        raise HTTPException(
            status_code=401,
            detail="JWT 无效或已撤销",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username


def revoke_access_token(token: str, username: str) -> None:
    payload = _decode_token(token)
    jti = payload.get("jti")
    if isinstance(jti, str):
        auth_repository.revoke_token(jti, username)
