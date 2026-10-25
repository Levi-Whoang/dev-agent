"""统一响应格式。

原来这个文件里 `import typing as Any` 是错的——把模块当类型用，
`data: Any = None` 的注解其实是 typing 模块对象，只是没人在运行时检查它。
现在改成真正从 typing 导入。
"""

from __future__ import annotations

from typing import Any


def success_response(data: Any = None, message: str = "success") -> dict:
    return {"code": 0, "message": message, "data": data}


def error_response(code: int, message: str, data: Any = None) -> dict:
    return {"code": code, "message": message, "data": data}
