"""工具执行器：解析参数 → Pydantic 校验 → 执行 → 统一结果格式。

单独抽出来的原因：非流式和流式两条链路都要执行工具，
这段逻辑如果各写一份，迟早会两边不一致（一边忘了校验参数之类）。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from tools.schema import create_args_model

logger = logging.getLogger("dev-agent.tools")

ToolRegistry = dict[str, Callable[..., Any]]


def _ensure_jsonable(value: Any) -> Any:
    """工具可能返回 set / 自定义对象，先归一化，避免 json.dumps 炸掉。"""
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return list(value)
    return str(value)


async def run_tool(
    name: str,
    raw_arguments: str | dict | None,
    registry: ToolRegistry,
) -> Any:
    """执行一次工具调用；失败也返回 dict（`{"error": ...}`），交给模型决定怎么办。

    成功时原样返回工具结果，不再包一层：字符串结果（比如
    `NO_RELEVANT_EVIDENCE`）必须保持原样，模型才认得出这是拒答协议。
    """
    if isinstance(raw_arguments, dict):
        arguments: dict[str, Any] = raw_arguments
    else:
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return {"error": f"工具参数不是合法 JSON：{exc}"}

    if not isinstance(arguments, dict):
        return {"error": "工具参数必须是 JSON 对象"}

    tool = registry.get(name)
    if tool is None:
        return {"error": f"工具不存在：{name}"}

    try:
        args_model = create_args_model(tool)
        validated = args_model(**arguments)
    except ValidationError as exc:
        return {
            "error": "工具参数校验失败",
            "details": exc.errors(include_context=False),
        }
    except Exception as exc:  # 比如工具参数缺类型注解
        return {"error": f"工具准备失败：{exc}"}

    kwargs = validated.model_dump()

    try:
        if inspect.iscoroutinefunction(tool):
            result = await tool(**kwargs)
        else:
            # 同步工具放进线程池，别让它堵住事件循环
            result = await asyncio.to_thread(tool, **kwargs)
    except Exception as exc:
        logger.exception("工具执行失败：%s", name)
        return {"error": f"工具执行失败：{exc}"}

    logger.info("工具 %s 执行成功", name)
    return _ensure_jsonable(result)


def tool_message_content(result: Any) -> str:
    """工具结果 → 给模型看的一段文本。

    字符串直接用（拒答标记需要精确匹配），其余序列化成 JSON。
    """
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


async def run_tool_async(
    name: str,
    raw_arguments: str | dict | None,
    registry: ToolRegistry,
) -> Any:
    """保留一个带 await 的别名，方便在测试里显式 await。"""
    return await run_tool(name, raw_arguments, registry)
