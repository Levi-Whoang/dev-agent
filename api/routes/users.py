"""注册、登录和退出登录接口。"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from api.api_response import success_response
from api.api_schemas import APIResponse
from auth.jwt_auth import (
    bearer_scheme,
    get_current_user,
    login_user,
    refresh_token_pair,
    register_user,
    revoke_access_token,
)
from auth.schemas import (
    RefreshRequest,
    RegisteredUserData,
    RegisterRequest,
    TokenData,
    TokenRequest,
)

router = APIRouter(prefix="/auth", tags=["用户"])


# 校验登录凭据并向客户端签发 JWT。
@router.post("/token", response_model=APIResponse)
def issue_token(request: TokenRequest, http_request: Request):
    """校验数据库账号并签发 30 分钟 JWT。"""
    source = http_request.client.host if http_request.client else "unknown"
    token_data = login_user(request.username, request.password, source)
    if token_data is None:
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return success_response(
        data=TokenData(**token_data).model_dump()
    )


# 创建用户账号并返回注册结果。
@router.post("/register", response_model=APIResponse, status_code=201)
def register(request: RegisterRequest, http_request: Request):
    """在 MySQL 用户表中创建新账号并保存密码哈希。"""
    source = http_request.client.host if http_request.client else "unknown"
    user_data = register_user(
        request.username, request.password, request.verification_code, source
    )
    return success_response(
        data=RegisteredUserData(**user_data).model_dump(),
        message="注册成功，请登录",
    )


# 校验刷新令牌并轮换签发新的令牌对。
@router.post("/refresh", response_model=APIResponse)
def refresh(request: RefreshRequest, http_request: Request):
    source = http_request.client.host if http_request.client else "unknown"
    token_data = refresh_token_pair(request.refresh_token, source)
    return success_response(data=TokenData(**token_data).model_dump())


# 撤销当前认证用户的访问令牌。
@router.post("/logout", response_model=APIResponse)
def logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    current_user: str = Depends(get_current_user),
):
    """撤销当前 JWT，令牌登记保留在数据库中直到过期清理。"""
    if credentials is not None:
        source = request.client.host if request.client else "unknown"
        revoke_access_token(credentials.credentials, current_user, source)
    return success_response(message="已退出登录")
