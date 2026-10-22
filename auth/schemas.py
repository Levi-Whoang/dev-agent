"""认证 API 的请求和响应模型。"""

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    password: str = Field(min_length=8, max_length=128)


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class RegisteredUserData(BaseModel):
    user_id: int
    username: str


class TokenData(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
