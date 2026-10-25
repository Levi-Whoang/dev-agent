"""流式事件定义。

前后端约定一套事件名，比“后端直接吐纯文本”多一步，但换来两件事：
- 前端能区分“模型正在说话”和“正在查知识库”，界面可以给不同反馈；
- 工具调用过程可观测，出错能定位是模型的问题还是工具的问题。

传输格式是 SSE（`event: xxx\ndata: {...}\n\n`），浏览器 fetch 可逐块解析。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# 事件类型
META = "meta"            # 本次请求的元信息（session_id、模型名）
DELTA = "delta"          # 一段增量文本
TOOL_START = "tool_start"    # 模型决定调用某工具
TOOL_RESULT = "tool_result"  # 工具执行完
DONE = "done"            # 全部结束，附完整回答
ERROR = "error"          # 出错，附原因


@dataclass(slots=True)
class AgentEvent:
    event: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        payload = json.dumps(self.data, ensure_ascii=False)
        return f"event: {self.event}\ndata: {payload}\n\n"

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.event, "data": self.data}
