"""MCP工具桥接、用户隔离、取消和失败降级的离线测试。"""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from jsonschema import ValidationError

from agent.agent import Agent
from agent.control import execute_tool
from integrations.mcp.config import FilesystemSettings
from integrations.mcp.filesystem import FilesystemMCP
from tools.knowledge import knowledge_owner_scope
from tools.schema import function_to_schema
from test.async_helpers import AsyncStream, collect
from test.test_agent import FakeSessionManager, llm_response, make_message, make_tool_call, stream_chunk

pytestmark = pytest.mark.anyio
SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}},
          "required": ["path"], "additionalProperties": False}


# 使用asyncio运行所有异步测试。
@pytest.fixture
def anyio_backend():
    return "asyncio"


# 构造无需Node或网络的MCP连接替身，并记录所有者目录及参数。
@pytest.fixture
def gateway(tmp_path, monkeypatch):
    import integrations.mcp.filesystem as module
    entry = tmp_path / "entry.js"
    entry.write_text("", encoding="utf-8")
    monkeypatch.setattr(module, "SERVER_ENTRY", entry)
    monkeypatch.setattr(module.shutil, "which", lambda command: command)
    instance = FilesystemMCP(FilesystemSettings(True, tmp_path / "workspace", "node", True, 1))
    instance.calls = []
    instance.closed = []
    definitions = [SimpleNamespace(name=name, description=name, input_schema=SCHEMA) for name in
                   ["read_text_file", "list_directory", "write_file", "unknown_tool"]]
    client = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=definitions, next_cursor=None)),
        call_tool=AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="文件内容")], is_error=False)),
    )

    # 保证进入和离开连接位于同一任务，同时跟踪断连清理。
    @asynccontextmanager
    async def connect(directory):
        instance.calls.append(Path(directory))
        try:
            yield client
        finally:
            instance.closed.append(Path(directory))

    monkeypatch.setattr(instance, "_connect", connect)
    instance.fake_client = client
    return instance


# 验证JSON Schema完整注册、命名隔离及重复启动不产生重复工具。
async def test_discovery_and_schema(gateway):
    agent = SimpleNamespace(tools=[], tool_registry={})
    await gateway.start(agent)
    assert gateway.available
    assert set(agent.tool_registry) == {"filesystem_read_text_file", "filesystem_list_directory", "filesystem_write_file"}
    tool = agent.tool_registry["filesystem_read_text_file"]
    assert function_to_schema(tool)["function"]["parameters"] == SCHEMA
    with pytest.raises(ValidationError):
        tool.validate_arguments({"path": 123})
    with pytest.raises(ValidationError):
        tool.validate_arguments({"path": ".", "owner_id": 2})
    await gateway.start(agent)
    assert len(agent.tools) == 3


# 模型不能指定用户ID或通过相对、绝对路径访问其他用户。
async def test_owner_scope_and_traversal(gateway):
    agent = SimpleNamespace(tools=[], tool_registry={})
    await gateway.start(agent)
    tool = agent.tool_registry["filesystem_read_text_file"]
    assert (await tool(path="demo.md"))["code"] == "mcp_auth_required"
    with knowledge_owner_scope(1):
        assert (await tool(path="demo.md"))["content"] == "文件内容"
        assert (await tool(path="../user-2/secret.md"))["code"] == "mcp_path_denied"
        assert (await tool(path=str(gateway.settings.root/"user-2/secret.md")))["code"] == "mcp_path_denied"
        assert (await tool(path=str(Path(__file__).resolve())))["code"] == "mcp_path_denied"
    assert gateway.calls[-1].name == "user-1"
    arguments = gateway.fake_client.call_tool.call_args.args[1]
    assert Path(arguments["path"]).parent.name == "user-1"
    with knowledge_owner_scope(2):
        await tool(path="demo.md")
    assert gateway.calls[-1].name == "user-2"


# 多文件读取、移动两端均需要目录校验，且用户目录不能重定向。
async def test_all_path_fields(gateway, tmp_path):
    root = tmp_path / "user-1"
    root.mkdir()
    with pytest.raises(PermissionError):
        gateway._arguments("read_multiple_files", {"paths": ["ok.md", "../secret.md"]}, root)
    with pytest.raises(PermissionError):
        gateway._arguments("move_file", {"source": "ok.md", "destination": "../secret.md"}, root)
    with pytest.raises(PermissionError):
        gateway._user_root(True)


# 连接异常时保留本地工具，禁用时不启动Server。
async def test_start_failure_and_disabled(gateway, monkeypatch):
    agent = SimpleNamespace(tools=[], tool_registry={"local": lambda: None})
    gateway.fake_client.list_tools.side_effect = RuntimeError("断开")
    await gateway.start(agent)
    assert gateway.status()["error_code"] == "mcp_unavailable"
    assert list(agent.tool_registry) == ["local"]
    gateway.settings = FilesystemSettings(False, gateway.settings.root, "node", True, 1)
    await gateway.start(agent)
    assert not gateway.status()["enabled"]


# 只读配置不会注册写工具，也拒绝绕过模型直接调用写工具。
async def test_read_only(gateway):
    gateway.settings = FilesystemSettings(True, gateway.settings.root, "node", False, 1)
    agent = SimpleNamespace(tools=[], tool_registry={})
    await gateway.start(agent)
    assert "filesystem_write_file" not in agent.tool_registry
    assert (await gateway.call_tool("write_file", {"path": "a.md"}, 1))["code"] == "mcp_unavailable"


# 异步工具停止或超时后，连接清理完成且没有占用本地工具线程。
@pytest.mark.parametrize("stop", [True, False])
async def test_async_tool_cancel_and_timeout(gateway, stop):
    agent = SimpleNamespace(tools=[], tool_registry={})
    await gateway.start(agent)
    entered = asyncio.Event()

    # 模拟持续等待的远端工具操作。
    async def waiting(*args):
        entered.set()
        await asyncio.Event().wait()

    gateway.fake_client.call_tool.side_effect = waiting
    event = Event()
    with knowledge_owner_scope(1):
        task = asyncio.create_task(execute_tool(agent.tool_registry["filesystem_read_text_file"],
                                               {"path": "a.md"}, 0.15, event))
        await asyncio.wait_for(entered.wait(), timeout=1)
        if stop:
            event.set()
        result = await asyncio.wait_for(task, timeout=2)
    assert result["code"] == ("cancelled" if stop else "tool_timeout")
    assert gateway.closed == gateway.calls


# 验证Agent的完整回答与流式回答均能调用MCP并持久化结果。
@pytest.mark.parametrize("streaming", [True, False])
async def test_agent_mcp_loop(gateway, monkeypatch, streaming):
    import agent.agent as module
    monkeypatch.setattr(module, "SessionManager", FakeSessionManager)
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    agent = Agent(client, [], {})
    await gateway.start(agent)
    decision = llm_response(make_message(tool_calls=[make_tool_call("filesystem_read_text_file", '{"path":"demo.md"}')]))
    final = llm_response(make_message(content="文件内容"))
    client.chat.completions.create.side_effect = ([decision, final, AsyncStream([stream_chunk(content="文件内容")])]
                                                 if streaming else [decision, final])
    with knowledge_owner_scope(9):
        answer = "".join(await collect(agent.run_stream("读取demo.md", "mcp"))) if streaming else await agent.run("读取demo.md", "mcp")
    assert answer == "文件内容"
    messages = await agent.session_manager.get_messages("mcp")
    result = json.loads(next(m["content"] for m in messages if m["role"] == "tool"))
    assert result["server"] == "filesystem"
    assert gateway.calls[-1].name == "user-9"


# 检索完成后仍能继续调用文件工具，避免RAG优化提前结束多步任务。
async def test_rag_then_filesystem(gateway, monkeypatch):
    import agent.agent as module
    monkeypatch.setattr(module, "SessionManager", FakeSessionManager)

    # 提供固定证据，测试不加载向量模型。
    def knowledge_search_context(query: str):
        return {"evidence": "公司示例"}

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[
        llm_response(make_message(tool_calls=[make_tool_call("knowledge_search_context", '{"query":"公司"}', "rag")])),
        llm_response(make_message(tool_calls=[make_tool_call("filesystem_read_text_file", '{"path":"demo.md"}', "file")])),
        llm_response(make_message(content="任务完成")),
        AsyncStream([stream_chunk(content="完成")]),
    ])
    agent = Agent(client, [], {"knowledge_search_context": knowledge_search_context})
    await gateway.start(agent)
    with knowledge_owner_scope(3):
        assert await collect(agent.run_stream("检索后读取文件", "chain")) == ["完成"]
    assert gateway.fake_client.call_tool.await_count == 1
