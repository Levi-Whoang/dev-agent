"""FastAPI 应用装配。

这个文件现在只负责“组装”：路由、中间件、异常处理、静态页、启动/关闭。
具体业务都在 api/routers、agent/、rag/、auth/ 里，
所以改一个接口不用在几百行的大文件里翻。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from api.api_response import success_response
from api.exceptions import register_exception_handlers
from api.middleware import RequestLogMiddleware
from api.routers import auth_routes, chat_routes, session_routes
from config import BASE_DIR, get_settings
from database.base import dispose_db, get_session_factory, init_db
from logging_config import logger

WEB_DIR = BASE_DIR / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    if settings.is_default_jwt_secret:
        logger.warning(
            "JWT_SECRET 仍是默认值，请在 .env 中设置自己的密钥后再部署"
        )
    if not settings.llm_api_key:
        logger.warning("LLM_API_KEY 未配置，聊天接口调用模型时会失败")

    await init_db()
    logger.info("数据库已就绪：%s", settings.database_url.split("://")[0])
    logger.info("知识库目录：%s", settings.knowledge_dir)

    try:
        yield
    finally:
        await dispose_db()
        logger.info("服务已关闭，数据库连接池已释放")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Dev Agent API",
        version="2.0.0",
        description=(
            "企业知识库问答助手：工具调用 Agent + 混合检索 RAG + "
            "JWT 鉴权 + SSE 流式输出"
        ),
        lifespan=lifespan,
    )

    # 本地开发放开跨域：前端既可能从 8000 端口访问，也可能直接双击 index.html
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLogMiddleware)

    register_exception_handlers(app)

    app.include_router(auth_routes)
    app.include_router(session_routes)
    app.include_router(chat_routes)

    @app.get("/api/health", tags=["system"], summary="健康检查")
    async def health():
        database_ok = True
        try:
            async with get_session_factory()() as db:
                await db.execute(text("SELECT 1"))
        except Exception:
            database_ok = False

        # 只读统计，不触发向量模型加载
        chunks = 0
        knowledge_dir = get_settings().knowledge_dir
        if knowledge_dir.exists():
            chunks = sum(
                1 for _ in knowledge_dir.glob("*.txt")
            )

        return success_response(
            data={
                "status": "ok" if database_ok else "degraded",
                "database": "ok" if database_ok else "unavailable",
                "knowledge_files": chunks,
            }
        )

    # 静态前端挂在根路径，必须放在所有 API 路由之后，否则会先被它吃掉
    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
