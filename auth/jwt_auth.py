"""注册、限流登录、JWT 轮换/撤销及当前用户认证。"""

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
from uuid import uuid4

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.passwords import hash_password, verify_password
from auth.repository import DuplicateUsernameError, auth_repository

JWT_ALGORITHM = "HS256"
JWT_ISSUER = "dev-agent"
ACCESS_EXPIRE_MINUTES = 30
REFRESH_EXPIRE_DAYS = 7
LOGIN_FAILURE_LIMIT = 5
LOGIN_WINDOW_MINUTES = 15
bearer_scheme = HTTPBearer(auto_error=False)


def validate_auth_settings() -> None:
    """启动时校验密钥；生产环境使用更严格的长度与占位值检查。"""
    secret = os.getenv("JWT_SECRET_KEY", "")
    environment = os.getenv("APP_ENV", "development").strip().lower()
    min_length = 48 if environment == "production" else 32
    normalized_secret = secret.strip().lower()
    placeholders = ("replace-", "change-me", "changeme", "your-secret", "secret")
    if len(secret) < min_length or normalized_secret.startswith(placeholders):
        raise RuntimeError(
            f"JWT_SECRET_KEY 配置无效：{environment} 环境至少需要 {min_length} 个字符的非占位密钥"
        )
    registration_code = os.getenv("AUTH_REGISTRATION_CODE", "").strip()
    if environment == "production" and (
        len(registration_code) < 16
        or registration_code.lower().startswith(("replace-", "change-me", "your-"))
    ):
        raise RuntimeError("生产环境必须配置至少 16 字符的非占位 AUTH_REGISTRATION_CODE")


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY", "")
    environment = os.getenv("APP_ENV", "development").strip().lower()
    min_length = 48 if environment == "production" else 32
    if len(secret) < min_length or secret.strip().lower().startswith(
        ("replace-", "change-me", "changeme", "your-secret", "secret")
    ):
        raise HTTPException(status_code=503, detail="JWT 密钥未正确配置")
    return secret


def normalize_username(username: str) -> str:
    return username.strip().lower()


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _audit(event: str, username: str | None, source: str | None, succeeded: bool) -> None:
    try:
        auth_repository.audit(event, username, _source_hash(source) if source else None, succeeded)
    except Exception:
        # 审计故障不覆盖认证结果；应用日志记录事件类别和异常，不含凭据。
        import logging
        logging.getLogger(__name__).exception("认证审计写入失败 event=%s", event)


def register_user(username: str, password: str, verification_code: str, source: str | None = None) -> dict:
    normalized = normalize_username(username)
    expected_code = os.getenv("AUTH_REGISTRATION_CODE", "")
    if not expected_code or not hmac.compare_digest(verification_code, expected_code):
        _audit("register", normalized, source, False)
        raise HTTPException(status_code=403, detail="注册验证码无效或未配置")
    try:
        user_id = auth_repository.create_user(normalized, hash_password(password))
    except DuplicateUsernameError as error:
        _audit("register", normalized, source, False)
        raise HTTPException(status_code=409, detail="用户名已被注册") from error
    _audit("register", normalized, source, True)
    return {"user_id": user_id, "username": normalized}


def _make_token_pair(username: str, user_id: int) -> tuple[str, str, int, list[dict]]:
    secret = _jwt_secret()
    now = datetime.now(timezone.utc)
    access_expiry = now + timedelta(minutes=ACCESS_EXPIRE_MINUTES)
    refresh_expiry = now + timedelta(days=REFRESH_EXPIRE_DAYS)
    access_jti, refresh_jti = uuid4().hex, uuid4().hex
    common = {"sub": username, "iss": JWT_ISSUER, "iat": now}
    access = jwt.encode(
        {**common, "exp": access_expiry, "jti": access_jti,
         "token_use": "access", "refresh_jti": refresh_jti},
        secret, algorithm=JWT_ALGORITHM,
    )
    refresh = jwt.encode(
        {**common, "exp": refresh_expiry, "jti": refresh_jti,
         "token_use": "refresh", "access_jti": access_jti},
        secret, algorithm=JWT_ALGORITHM,
    )
    rows = [
        {"jti": access_jti, "user_id": user_id, "expires_at": access_expiry},
        {"jti": refresh_jti, "user_id": user_id, "expires_at": refresh_expiry},
    ]
    return access, refresh, access_jti, rows


def _issue_pair(username: str, user_id: int) -> dict:
    access, refresh, _, rows = _make_token_pair(username, user_id)
    auth_repository.create_tokens(rows)
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "expires_in": ACCESS_EXPIRE_MINUTES * 60,
        "refresh_expires_in": REFRESH_EXPIRE_DAYS * 24 * 60 * 60,
    }


def create_access_token(username: str, user_id: int) -> tuple[str, int]:
    """为现有 API 测试和内部调用保留单 access token helper。"""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=ACCESS_EXPIRE_MINUTES)
    jti = uuid4().hex
    token = jwt.encode(
        {"sub": username, "iss": JWT_ISSUER, "iat": now, "exp": expires_at,
         "jti": jti, "token_use": "access"},
        _jwt_secret(), algorithm=JWT_ALGORITHM,
    )
    auth_repository.create_token(jti, user_id, expires_at)
    return token, ACCESS_EXPIRE_MINUTES * 60


def login_user(username: str, password: str, source: str) -> dict | None:
    normalized = normalize_username(username)
    identity_hash = _source_hash(f"{normalized}\0{source}")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window_start = now - timedelta(minutes=LOGIN_WINDOW_MINUTES)
    if auth_repository.login_failure_count(identity_hash, window_start) >= LOGIN_FAILURE_LIMIT:
        _audit("login_rate_limited", normalized, source, False)
        raise HTTPException(status_code=429, detail="登录失败次数过多，请 15 分钟后重试")

    user = auth_repository.find_user(normalized)
    # 即使用户名不存在也执行一次 PBKDF2，减小用户名枚举造成的耗时差异。
    dummy_hash = "pbkdf2_sha256$310000$00112233445566778899aabbccddeeff$" + "0" * 64
    password_matches = verify_password(password, user["password_hash"] if user else dummy_hash)
    if not user or not password_matches:
        auth_repository.record_login_failure(identity_hash, now)
        _audit("login", normalized, source, False)
        return None

    auth_repository.clear_login_failures(identity_hash)
    token_pair = _issue_pair(user["username"], int(user["id"]))
    _audit("login", normalized, source, True)
    return token_pair


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            _jwt_secret(),
            algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER,
            options={"require": ["sub", "iss", "iat", "exp", "jti", "token_use"]},
        )
    except jwt.PyJWTError as error:
        raise HTTPException(status_code=401, detail="JWT 无效或已过期") from error


def refresh_token_pair(token: str, source: str | None = None) -> dict:
    try:
        payload = _decode_token(token)
    except HTTPException:
        _audit("refresh", None, source, False)
        raise
    username = payload.get("sub")
    refresh_jti = payload.get("jti")
    access_jti = payload.get("access_jti")
    if payload.get("token_use") != "refresh" or not all(
        isinstance(item, str) for item in (username, refresh_jti, access_jti)
    ):
        _audit("refresh", username if isinstance(username, str) else None, source, False)
        raise HTTPException(status_code=401, detail="刷新令牌无效")
    user = auth_repository.find_user(username)
    if not user:
        _audit("refresh", username, source, False)
        raise HTTPException(status_code=401, detail="刷新令牌无效")
    access, refresh, new_access_jti, rows = _make_token_pair(username, int(user["id"]))
    if not auth_repository.rotate_tokens(refresh_jti, access_jti, username, rows):
        _audit("refresh", username, source, False)
        raise HTTPException(status_code=401, detail="刷新令牌已过期或已使用")
    _audit("refresh", username, source, True)
    return {
        "access_token": access, "refresh_token": refresh, "token_type": "bearer",
        "expires_in": ACCESS_EXPIRE_MINUTES * 60,
        "refresh_expires_in": REFRESH_EXPIRE_DAYS * 24 * 60 * 60,
    }


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="需要 Bearer JWT 认证")
    payload = _decode_token(credentials.credentials)
    username, jti = payload.get("sub"), payload.get("jti")
    if (
        payload.get("token_use") != "access"
        or not isinstance(username, str)
        or not isinstance(jti, str)
        or not auth_repository.is_token_active(jti, username)
    ):
        raise HTTPException(status_code=401, detail="JWT 无效或已撤销")
    return username


def revoke_access_token(token: str, username: str, source: str | None = None) -> None:
    payload = _decode_token(token)
    jti, refresh_jti = payload.get("jti"), payload.get("refresh_jti")
    if payload.get("sub") != username or payload.get("token_use") != "access" or not isinstance(jti, str):
        raise HTTPException(status_code=401, detail="JWT 无效")
    auth_repository.revoke_token(jti, username, refresh_jti if isinstance(refresh_jti, str) else None)
    _audit("logout", username, source, True)
