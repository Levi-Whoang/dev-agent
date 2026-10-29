"""组装 FastAPI 应用及功能路由。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.errors import install_error_handling
from app.sessions import router as sessions_router
from app.users import router as users_router
from auth.jwt_auth import validate_auth_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_auth_settings()
    yield


app = FastAPI(title="Dev Agent API", lifespan=lifespan)


@app.get("/health")
def health():
    """进程存活检查，不连接外部模型或数据库。"""
    return {"status": "ok"}


app.include_router(users_router)
app.include_router(sessions_router)
install_error_handling(app)
