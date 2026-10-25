"""端到端回答评估：真实模型 + 真实知识库。

用法：
    python test/evaluate_rag_answers.py

会真的调用模型 API，按调用量产生费用。指标只是初步的：
- 可回答问题：检查预期关键词是否出现在回答里；
- 无答案问题：检查回答里是否有拒答措辞。
最终仍需要人工复核。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.agent import Agent  # noqa: E402
from agent.factory import get_llm_client, get_tool_schemas  # noqa: E402
from config import get_settings  # noqa: E402
from retrieval_dataset import EVALUATION_CASES  # noqa: E402
from tools.registry import TOOLS  # noqa: E402

REFUSAL_MARKERS = (
    "无法确认", "无法确定", "没有相关信息", "没有足够的信息",
    "知识库中没有", "未提供", "不能确定", "没有提到",
)


class MemoryStore:
    """评估时用内存存对话，不写数据库。"""

    def __init__(self) -> None:
        self.sessions: dict[str, list[dict]] = {}

    async def history(self, session_id: str, limit: int = 40) -> list[dict]:
        return list(self.sessions.get(session_id, []))

    async def append(self, session_id, role, content, payload=None) -> None:
        message = {"role": role, "content": content}
        if payload:
            message.update(payload)
        self.sessions.setdefault(session_id, []).append(message)


async def main() -> None:
    if not os.getenv("LLM_API_KEY"):
        raise SystemExit("请先在 .env 中配置 LLM_API_KEY")

    store = MemoryStore()
    answerable = correct = 0
    unanswerable = refused = 0
    errors = 0
    used_retrieval = 0

    print("\nRAG 真实 LLM 回答评估")
    print("=" * 72)

    for index, case in enumerate(EVALUATION_CASES, start=1):
        query = case["query"]
        expected = case["expected"]
        session_id = f"rag_eval_{index}"

        agent = Agent(
            client=get_llm_client(),
            store=store,
            session_id=session_id,
            tools=get_tool_schemas(),
            tool_registry=TOOLS,
            model=get_settings().llm_model,
        )

        try:
            answer = await agent.run(query)
        except Exception as exc:
            errors += 1
            print(f"\n[{index}] {query}\n执行失败：{type(exc).__name__}: {exc}")
            continue

        messages = store.sessions.get(session_id, [])
        called = any(
            call.get("function", {}).get("name") == "knowledge_search_context"
            for message in messages
            if message.get("role") == "assistant"
            for call in message.get("tool_calls", [])
        )
        used_retrieval += int(called)

        if expected is None:
            unanswerable += 1
            ok = any(marker in answer for marker in REFUSAL_MARKERS)
            refused += int(ok)
            status = "拒答符合预期" if ok else "需要人工检查（可能编造）"
        else:
            answerable += 1
            ok = all(keyword.lower() in answer.lower() for keyword in expected)
            correct += int(ok)
            status = "关键词通过" if ok else "关键词未通过"

        print(f"\n[{index}] {query}")
        print(f"评估结果：{status}｜调用检索工具：{'是' if called else '否'}")
        print(f"模型回答：{answer}")

    total = len(EVALUATION_CASES)
    print("\n" + "=" * 72)
    print(f"执行成功：{total - errors}/{total}，失败：{errors}")
    print(f"可回答问题关键词通过率：{correct}/{answerable}")
    print(f"无答案问题拒答率：{refused}/{unanswerable}")
    print(f"调用检索工具的问题数：{used_retrieval}/{total - errors}")


if __name__ == "__main__":
    asyncio.run(main())
