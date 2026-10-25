"""注册 / 登录 / 当前用户。

路由层只做三件事：拿参数、调 service、把业务异常翻译成状态码。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.api_response import success_response
from api.api_schemas import APIResponse
from api.deps import CurrentUser, DbSession
from auth.schemas import LoginRequest, RegisterRequest, UserOut
from auth.service import (
    InactiveUser,
    InvalidCredentials,
    UsernameTaken,
    authenticate_user,
    issue_token,
    register_user,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def _auth_payload(user, token: str, expires_in: int) -> dict:
    return {
        "user": UserOut.model_validate(user).model_dump(mode="json"),
        "token": {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": expires_in,
        },
    }


@router.post(
    "/register",
    response_model=APIResponse,
    status_code=status.HTTP_201_CREATED,
    summary="注册并直接登录",
)
async def register(payload: RegisterRequest, db: DbSession):
    try:
        user = await register_user(db, payload)
    except UsernameTaken:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名已被占用",
        )

    token, expires_in = issue_token(user)
    return success_response(
        data=_auth_payload(user, token, expires_in),
        message="注册成功",
    )


@router.post("/login", response_model=APIResponse, summary="登录换取 JWT")
async def login(payload: LoginRequest, db: DbSession):
    try:
        user = await authenticate_user(db, payload.username, payload.password)
    except InvalidCredentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )
    except InactiveUser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用",
        )

    token, expires_in = issue_token(user)
    return success_response(
        data=_auth_payload(user, token, expires_in),
        message="登录成功",
    )


@router.get("/me", response_model=APIResponse, summary="当前登录用户")
async def me(user: CurrentUser):
    return success_response(data=UserOut.model_validate(user).model_dump(mode="json"))
