"""认证 API 的请求和响应模型。"""

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class TokenData(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
