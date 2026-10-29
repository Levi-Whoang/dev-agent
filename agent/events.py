"""流式事件定义。

前后端约定一套事件名，比“后端直接吐纯文本”多一步，但换来两件事：
- 前端能区分“模型正在说话”和“正在查知识库”，界面可以给不同反馈；
- 工具调用过程可观测，出错能定位是模型的问题还是工具的问题。

一次 ReAct 循环的事件顺序：

    meta
    delta × N          正文增量（可能是本步的思考，也可能是最终答案）
    thought            本步思考（内容已包含在上面那些 delta 里）
    tool_start         开始调工具
    tool_result        工具返回
    delta × N          ……下一轮的思考，如此往复……
    done               附带 termination 和 trace（整条推理轨迹）

关于 `thought` 和 `delta` 的关系：thought 的文本一定已经在 delta 里出现过了。
之所以还单独发一条，是为了让不逐字渲染的消费方（脚本、日志、评测）
也能直接拿到推理内容，而不用自己把 delta 拼起来猜哪一段是思考。

传输格式是 SSE（`event: xxx\ndata: {...}\n\n`），浏览器 fetch 可逐块解析。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# 事件类型
META = "meta"            # 本次请求的元信息（session_id、模型名）
THOUGHT = "thought"      # 本步的推理（正文原文，已在 delta 里出现过）
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
