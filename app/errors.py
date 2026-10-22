"""请求日志中间件和统一异常处理。"""

import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.api_response import error_response
from logs.logging_config import logger


def install_error_handling(app: FastAPI) -> None:
    """安装统一异常响应和请求日志中间件。"""

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.perf_counter()
        try:
            response = await call_next(request)
            elapsed = time.perf_counter() - start_time
            logger.info(
                "%s %s status=%s duration=%.3fs",
                request.method,
                request.url.path,
                response.status_code,
                elapsed,
            )
            return response
        except Exception:
            elapsed = time.perf_counter() - start_time
            logger.exception(
                "%s %s failed duration=%.3fs",
                request.method,
                request.url.path,
                elapsed,
            )
            raise

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(
                code=exc.status_code,
                message=str(exc.detail),
            ),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        return JSONResponse(
            status_code=422,
            content=error_response(
                code=422,
                message="请求参数校验失败",
                data=exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(
        request: Request,
        exc: Exception,
    ):
        return JSONResponse(
            status_code=500,
            content=error_response(
                code=500,
                message="服务器内部错误",
            ),
        )
