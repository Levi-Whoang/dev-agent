"""共享Filesystem网关，不在导入阶段启动Server。"""
from integrations.mcp.filesystem import FilesystemMCP

filesystem = FilesystemMCP()
