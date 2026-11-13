"""组装 FastAPI 应用及功能路由。"""

from contextlib import asynccontextmanager
import os
from starlette.concurrency import run_in_threadpool
from tools.knowledge import warmup_knowledge

from fastapi import FastAPI

from app.errors import install_error_handling
from app.process_guard import ensure_single_process
from api.routes.documents import router as documents_router
from api.routes.sessions import router as sessions_router
from api.routes.users import router as users_router
from api.routes.mcp import router as mcp_router
from auth.jwt_auth import validate_auth_settings


# 在 FastAPI 启动时检查认证配置并管理应用生命周期。
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 停止生成和同会话互斥依赖进程内状态，多 worker 下必须拒绝启动。
    ensure_single_process()
    validate_auth_settings()
    if os.getenv("RAG_WARMUP", "true").lower() in {"true", "1", "yes"}:
        await run_in_threadpool(warmup_knowledge)
    try:
        from agent.runtime import agent
        from integrations.mcp.runtime import filesystem
        await filesystem.start(agent)
        yield
    finally:
        from agent.runtime import client
        try:
            await client.close()
        finally:
            from modules.database import dispose_database
            await dispose_database()


app = FastAPI(title="Dev Agent API", lifespan=lifespan)


# 返回不依赖数据库和模型的服务存活状态。
@app.get("/health")
def health():
    """进程存活检查，不连接外部模型或数据库。"""
    return {"status": "ok"}


app.include_router(users_router)
app.include_router(sessions_router)
app.include_router(documents_router)
app.include_router(mcp_router)
install_error_handling(app)
