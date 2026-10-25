"""纯 ASGI 中间件：记录请求日志。

为什么不用 `@app.middleware("http")`（BaseHTTPMiddleware）：
它会把响应包成流自己再转发，对普通 JSON 没问题，但遇到
`StreamingResponse` 时容易把 SSE 的分块行为搞坏——流式输出变成“攒完再吐”。
纯 ASGI 只是包一层 send，body 一个字节都不碰，最安全。
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("dev-agent.access")


class RequestLogMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "%s %s 未捕获异常", scope.get("method"), scope.get("path")
            )
            raise
        finally:
            elapsed = time.perf_counter() - started
            logger.info(
                "%s %s status=%s duration=%.3fs",
                scope.get("method"),
                scope.get("path"),
                status_code,
                elapsed,
            )
