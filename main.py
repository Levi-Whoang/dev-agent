"""兼容旧导入路径并导出共享 Agent 实例。"""

from agent.runtime import agent, client

__all__ = ["agent", "client"]
