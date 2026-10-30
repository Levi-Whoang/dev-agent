"""组装 FastAPI 应用及功能路由。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.errors import install_error_handling
from api.routes.documents import router as documents_router
from api.routes.sessions import router as sessions_router
from api.routes.users import router as users_router
from auth.jwt_auth import validate_auth_settings


# 在 FastAPI 启动时检查认证配置并管理应用生命周期。
@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_auth_settings()
    yield


app = FastAPI(title="Dev Agent API", lifespan=lifespan)


# 返回不依赖数据库和模型的服务存活状态。
@app.get("/health")
def health():
    """进程存活检查，不连接外部模型或数据库。"""
    return {"status": "ok"}


app.include_router(users_router)
app.include_router(sessions_router)
app.include_router(documents_router)
install_error_handling(app)
