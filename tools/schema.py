"""从函数签名生成 Tool Calling 的 Schema。

关键点：用 `typing.get_type_hints()` 解析注解，而不是直接用 `param.annotation`。
因为工具文件里普遍写了 `from __future__ import annotations`，
此时注解是**字符串**（如 `"Literal['add', ...]"`），
动态建出来的 Pydantic 模型无法解析这些字符串，会在
`model_json_schema()` 时抛 `PydanticUserError: not fully defined`。
"""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

from pydantic import create_model


def create_args_model(func):
    signature = inspect.signature(func)

    try:
        hints = get_type_hints(func)
    except Exception:
        # 解析失败就退回原始注解，让后面的校验报出具体问题
        hints: dict[str, Any] = {}

    fields: dict[str, tuple[Any, Any]] = {}

    for name, parameter in signature.parameters.items():
        annotation = hints.get(name, parameter.annotation)

        if annotation is inspect.Parameter.empty:
            raise TypeError(f"工具 {func.__name__} 的参数 {name} 缺少类型注解")

        if parameter.default is inspect.Parameter.empty:
            fields[name] = (annotation, ...)
        else:
            fields[name] = (annotation, parameter.default)

    return create_model(f"{func.__name__}Args", **fields)


def function_to_schema(func) -> dict:
    args_model = create_args_model(func)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func) or "",
            "parameters": args_model.model_json_schema(),
        },
    }


def tools_to_schema(tools) -> list[dict]:
    return [function_to_schema(tool) for tool in tools.values()]
