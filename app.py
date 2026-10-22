"""兼容旧启动命令的 FastAPI 应用入口。"""

from app.main import app

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
