"""已认证用户可查看的MCP运行状态接口。"""
from fastapi import APIRouter, Depends
from api.api_response import success_response
from api.session_helpers import database_user_id
from auth.jwt_auth import get_current_user
from integrations.mcp.runtime import filesystem

router = APIRouter(prefix="/mcp", tags=["MCP"])


# 返回Filesystem连接状态和可调用工具，接口要求登录。
@router.get("/filesystem")
async def filesystem_status(current_user: str = Depends(get_current_user)):
    owner_id = await database_user_id(current_user)
    return success_response({**filesystem.status(), "workspace": f"user-{owner_id}"})
