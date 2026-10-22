"""注册、登录和退出登录接口。"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from api.api_response import success_response
from api.api_schemas import APIResponse
from auth.jwt_auth import (
    bearer_scheme,
    get_current_user,
    login_user,
    register_user,
    revoke_access_token,
)
from auth.schemas import RegisteredUserData, RegisterRequest, TokenData, TokenRequest

router = APIRouter(prefix="/auth", tags=["用户"])


@router.post("/token", response_model=APIResponse)
def issue_token(request: TokenRequest):
    """校验数据库账号并签发 30 分钟 JWT。"""
    token_data = login_user(request.username, request.password)
    if token_data is None:
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token, expires_in = token_data
    return success_response(
        data=TokenData(access_token=access_token, expires_in=expires_in).model_dump()
    )


@router.post("/register", response_model=APIResponse, status_code=201)
def register(request: RegisterRequest):
    """在 MySQL 用户表中创建新账号并保存密码哈希。"""
    user_data = register_user(request.username, request.password)
    return success_response(
        data=RegisteredUserData(**user_data).model_dump(),
        message="注册成功，请登录",
    )


@router.post("/logout", response_model=APIResponse)
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    current_user: str = Depends(get_current_user),
):
    """撤销当前 JWT，令牌登记保留在数据库中直到过期清理。"""
    if credentials is not None:
        revoke_access_token(credentials.credentials, current_user)
    return success_response(message="已退出登录")
