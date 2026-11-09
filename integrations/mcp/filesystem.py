"""通过stdio连接官方Filesystem MCP Server，并按用户限制目录。"""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import logging
from pathlib import Path
import shutil

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from starlette.concurrency import run_in_threadpool

from integrations.mcp.config import FilesystemSettings, SERVER_ENTRY
from integrations.mcp.tools import MCPTool

logger = logging.getLogger(__name__)
# 仅适配已明确路径字段的工具，Server升级新增工具不会自动获得权限。
PATH_FIELDS = {
    "read_text_file": ("path",), "read_multiple_files": ("paths",),
    "list_directory": ("path",), "list_directory_with_sizes": ("path",),
    "directory_tree": ("path",), "search_files": ("path",),
    "get_file_info": ("path",), "write_file": ("path",),
    "edit_file": ("path",), "create_directory": ("path",),
    "move_file": ("source", "destination"),
}
WRITE_TOOLS = {"write_file", "edit_file", "create_directory", "move_file"}


class FilesystemMCP:
    # 初始化轻量网关，导入应用时不启动子进程或连接外部服务。
    def __init__(self, settings=None):
        self.settings = settings
        self.tools = {}
        self.available = False
        self.error_code = None
        self._slots = asyncio.Semaphore(4)

    # 每条连接都由同一个异步任务创建和关闭，避免SDK取消域跨任务退出。
    @asynccontextmanager
    async def _connect(self, directory):
        parameters = StdioServerParameters(
            command=self.settings.command,
            args=[str(SERVER_ENTRY), str(directory)],
            cwd=str(directory),
        )
        async with Client(parameters, read_timeout_seconds=self.settings.timeout) as client:
            yield client

    # 在启动时发现工具并注册；连接失败时保留原有聊天和RAG功能。
    async def start(self, agent):
        self.available = False
        self.error_code = None
        previous = set(self.tools)
        if previous:
            agent.tool_registry = {k: v for k, v in agent.tool_registry.items() if k not in previous}
            agent.tools = [s for s in agent.tools if s["function"]["name"] not in previous]
            self.tools = {}
        try:
            self.settings = self.settings or FilesystemSettings.from_env()
            if not self.settings.enabled:
                return
            if not SERVER_ENTRY.is_file() or not shutil.which(self.settings.command):
                raise RuntimeError("MCP运行依赖尚未安装")
            discovery = self.settings.root / "_discovery"
            await run_in_threadpool(discovery.mkdir, parents=True, exist_ok=True)
            async with asyncio.timeout(self.settings.timeout):
                async with self._connect(discovery) as client:
                    definitions = []
                    cursor = None
                    while True:
                        result = await client.list_tools(cursor=cursor)
                        definitions.extend(result.tools)
                        cursor = result.next_cursor
                        if not cursor:
                            break
            tools = {}
            for definition in definitions:
                if definition.name not in PATH_FIELDS:
                    continue
                if not self.settings.allow_write and definition.name in WRITE_TOOLS:
                    continue
                tool = MCPTool(self, definition.name, definition.description, definition.input_schema)
                tools[tool.__name__] = tool
            if not tools:
                raise RuntimeError("Server没有可用的Filesystem工具")
            # 使用实例副本，不污染本地TOOLS常量或其他Agent实例。
            agent.tool_registry = dict(agent.tool_registry)
            agent.tool_registry.update(tools)
            agent.tools = list(agent.tools)
            agent.tools.extend(tool.openai_schema() for tool in tools.values())
            self.tools = tools
            self.available = True
            logger.info("mcp_connected server=filesystem tools=%s", len(tools))
        except Exception:
            self.error_code = "mcp_unavailable"
            logger.warning("mcp_unavailable server=filesystem 请检查Node、npm依赖及MCP配置")

    # 验证用户目录本身未被替换为指向其他位置的链接。
    def _user_root(self, owner_id):
        if type(owner_id) is not int or owner_id <= 0:
            raise PermissionError("无效的文件所有者")
        root = self.settings.root.resolve()
        directory = root / f"user-{owner_id}"
        if directory.resolve() != directory:
            raise PermissionError("用户工作目录不能是符号链接")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    # 将模型的相对路径限定到用户根目录，拒绝穿越、跨用户及符号链接越界。
    def _arguments(self, name, arguments, directory):
        result = deepcopy(arguments)
        for field in PATH_FIELDS[name]:
            values = result[field] if field == "paths" else [result[field]]
            resolved = []
            for value in values:
                candidate = Path(value)
                target = (candidate if candidate.is_absolute() else directory / candidate).resolve()
                if not target.is_relative_to(directory):
                    raise PermissionError("路径超出当前用户工作目录")
                resolved.append(str(target))
            result[field] = resolved if field == "paths" else resolved[0]
        return result

    # 为每次调用启动仅允许当前用户目录的Server，结束、超时或取消时关闭连接。
    async def call_tool(self, name, arguments, owner_id):
        if not self.available or f"filesystem_{name}" not in self.tools:
            return {"error": "Filesystem MCP暂不可用", "code": "mcp_unavailable"}
        try:
            directory = await run_in_threadpool(self._user_root, owner_id)
            resolved = await run_in_threadpool(self._arguments, name, arguments, directory)
            async with asyncio.timeout(self.settings.timeout):
                async with self._slots:
                    async with self._connect(directory) as client:
                        result = await client.call_tool(name, resolved)
            # 仅向模型传递文本，限制大文件内容占用上下文。
            content = "\n".join(block.text for block in result.content if block.type == "text")
            limit = self.settings.max_result_chars
            return {"server": "filesystem", "tool": name, "is_error": result.is_error,
                    "content": content[:limit], "truncated": len(content) > limit}
        except PermissionError:
            return {"error": "只能访问当前用户的文件工作目录", "code": "mcp_path_denied"}
        except TimeoutError:
            return {"error": "Filesystem MCP调用超时", "code": "tool_timeout"}
        except Exception:
            logger.warning("mcp_call_failed server=filesystem tool=%s", name)
            return {"error": "Filesystem MCP调用失败", "code": "mcp_call_failed"}

    # 返回运行状态与工具名称，不公开服务器路径或其他用户信息。
    def status(self):
        return {"enabled": bool(self.settings and self.settings.enabled),
                "available": self.available, "error_code": self.error_code,
                "allow_write": bool(self.settings and self.settings.allow_write),
                "tools": sorted(self.tools) if self.available else []}
