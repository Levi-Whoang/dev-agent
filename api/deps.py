"""类型别名 + 公共依赖。

写成 `DbSession = Annotated[AsyncSession, Depends(get_db)]` 之后，
路由签名就只剩业务参数，依赖注入的噪音少了一半。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from database.base import get_db
from database.models import User

DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
