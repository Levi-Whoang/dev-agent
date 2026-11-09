"""不依赖LLM或数据库密钥的真实Filesystem MCP冒烟检查。"""
import asyncio
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.control import execute_tool
from integrations.mcp.config import FilesystemSettings
from integrations.mcp.filesystem import FilesystemMCP
from tools.knowledge import knowledge_owner_scope


# 在临时目录连接真实Server，验证工具注册、读写、用户隔离和目录限制。
async def main():
    with tempfile.TemporaryDirectory(prefix="dev-agent-mcp-") as temporary:
        gateway = FilesystemMCP(FilesystemSettings(True, Path(temporary).resolve(), "node", True, 15))
        agent = SimpleNamespace(tools=[], tool_registry={})
        await gateway.start(agent)
        if not gateway.available:
            raise RuntimeError("MCP连接失败，请先执行 npm ci --prefix integrations/mcp/node")
        write = agent.tool_registry["filesystem_write_file"]
        read = agent.tool_registry["filesystem_read_text_file"]
        with knowledge_owner_scope(101):
            written = await execute_tool(write, {"path": "demo.md", "content": "Filesystem MCP连接成功"}, 15)
            assert not written.get("is_error") and "error" not in written, written
            result = await execute_tool(read, {"path": "demo.md"}, 15)
            assert "Filesystem MCP连接成功" in result.get("content", ""), result
            denied = await execute_tool(read, {"path": "../user-202/demo.md"}, 15)
            assert denied.get("code") == "mcp_path_denied", denied
        with knowledge_owner_scope(202):
            other = await execute_tool(read, {"path": "demo.md"}, 15)
            assert other.get("is_error") is True, other
        # 绕过适配层直接探测Server，确认服务端也拒绝另一个用户的目录。
        async with gateway._connect(gateway._user_root(202)) as client:
            forbidden = await client.call_tool("read_text_file", {"path": str(Path(temporary)/"user-101/demo.md")})
            assert forbidden.is_error
        print(f"Filesystem MCP检查通过：发现{len(agent.tools)}个工具，读写、用户隔离、客户端及Server路径限制正常。")


if __name__ == "__main__":
    asyncio.run(main())
