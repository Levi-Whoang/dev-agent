"""Filesystem MCP的运行配置。"""
from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_ENTRY = Path(__file__).parent / "node/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js"


@dataclass(frozen=True)
class FilesystemSettings:
    enabled: bool
    root: Path
    command: str
    allow_write: bool
    timeout: float
    max_result_chars: int = 24000

    # 从环境变量读取配置，固定Server入口，避免通过shell拼接命令。
    @classmethod
    def from_env(cls):
        root = Path(os.getenv("MCP_FILESYSTEM_ROOT", "mcp_workspace"))
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        timeout = float(os.getenv("MCP_TIMEOUT_SECONDS", "15"))
        if not 0 < timeout <= 120:
            raise ValueError("MCP_TIMEOUT_SECONDS 必须大于0且不超过120秒")
        return cls(
            enabled=os.getenv("MCP_FILESYSTEM_ENABLED", "true").lower() in {"true", "1", "yes"},
            root=root.resolve(), command=os.getenv("MCP_NODE_COMMAND", "node"),
            allow_write=os.getenv("MCP_FILESYSTEM_ALLOW_WRITE", "true").lower() in {"true", "1", "yes"},
            timeout=timeout,
        )
