"""统一异常处理：让所有错误都长成一个样子（code / message / data）。

注意 401 的 `WWW-Authenticate` 头必须原样带回去，否则浏览器和
Swagger UI 都认不出这是“需要登录”。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.api_response import error_response

logger = logging.getLogger("dev-agent.errors")


def _serializable_errors(exc: RequestValidationError) -> list[dict]:
    """把校验错误转成能进 JSON 的结构。

    Pydantic v2 的 errors() 里，自定义校验器抛的异常会原封不动放在
    ctx.error 里（比如 `ValueError("标题不能为空")`）。直接丢给 JSONResponse
    会 `TypeError: Object of type ValueError is not JSON serializable`，
    结果本该是 422 的请求变成 500 —— 报错信息还完全指不到真正原因。

    内置约束（min_length 之类）的 ctx 里是数字，不会触发这个坑，
    所以只有当项目里出现第一个自定义校验器时才会暴雷。
    """
    return jsonable_encoder(exc.errors(), custom_encoder={BaseException: str})


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.info(
            "%s %s -> %s %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(code=exc.status_code, message=str(exc.detail)),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content=error_response(
                code=422,
                message="请求参数校验失败",
                data=_serializable_errors(exc),
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exc: Exception):
        logger.exception("未预期异常：%s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=error_response(code=500, message="服务器内部错误"),
        )
