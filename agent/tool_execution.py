"""统一Agent工具调用的消息解析、参数验证与执行。"""
import json
import os

from jsonschema import ValidationError as SchemaValidationError
from pydantic import ValidationError

from integrations.mcp.tools import MCPTool
from tools.schema import create_args_model
from agent.control import execute_tool


# 将模型调用对象转换为统一字段结构。
def normalize_tool_call(tool_call):
    if isinstance(tool_call, dict):
        return tool_call
    return {
        "id": tool_call.id,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


# 为模型消息持久化时移除工具决策中的正文草稿。
def assistant_tool_message(message):
    saved = message.model_dump(exclude_none=True)
    if message.tool_calls:
        saved.pop("content", None)
    return saved


# 统一解析、校验本地/MCP参数，并按配置执行工具。
async def execute_tool_call(tool_registry, tool_call, timeout=None, stop_event=None):
    call = normalize_tool_call(tool_call)
    function = call.get("function") or {}
    name = function.get("name")
    raw_arguments = function.get("arguments", "{}")
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except (TypeError, json.JSONDecodeError) as error:
        return {"error": f"工具参数格式错误: {error}"}
    if not isinstance(arguments, dict):
        return {"error": "工具参数必须是JSON对象", "code": "invalid_tool_arguments"}
    tool = tool_registry.get(name)
    if tool is None:
        return {"error": f"工具不存在: {name}"}
    try:
        if isinstance(tool, MCPTool):
            validated = tool.validate_arguments(arguments)
        else:
            validated = create_args_model(tool)(**arguments).model_dump()
    except SchemaValidationError:
        return {"error": "工具参数校验失败", "code": "invalid_tool_arguments"}
    except ValidationError as error:
        return {"error": "工具参数校验失败", "details": error.errors(include_context=False)}
    except Exception as error:
        return {"error": f"工具执行前发生错误: {error}"}
    timeout = timeout if timeout is not None else float(os.getenv("TOOL_TIMEOUT_SECONDS", "15"))
    try:
        return await execute_tool(tool, validated, timeout, stop_event)
    except Exception as error:
        return {"error": f"工具执行失败: {error}"}
