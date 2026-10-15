
"""评估 Agent 最终回答是否符合预期。"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

import agent.agent as agent_module
from agent.agent import Agent
from retrieval_dataset import EVALUATION_CASES
from tools.registry import TOOLS
from tools.schema import tools_to_schema


class MemorySessionManager:
    """评估时使用内存保存对话，不连接 MySQL。"""

    def __init__(self):
        self.sessions = {}

    def get_messages(self, session_id):
        return list(self.sessions.get(session_id, []))

    def save(self, session_id, messages):
        self.sessions[session_id] = list(messages)


REFUSAL_MARKERS = (
    "无法确认",
    "无法确定",
    "没有相关信息",
    "没有足够的信息",
    "知识库中没有",
    "未提供",
    "不能确定",
)


def main():
    if not os.getenv("LLM_API_KEY"):
        raise RuntimeError("请先在 .env 中配置 LLM_API_KEY")

    # 替换 Agent 使用的会话管理器，避免连接 MySQL
    agent_module.SessionManager = MemorySessionManager

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    agent = Agent(
        client=client,
        tools=tools_to_schema(TOOLS),
        tool_registry=TOOLS,
    )

    answerable_total = 0
    answerable_correct = 0

    unanswerable_total = 0
    unanswerable_refused = 0

    tool_call_total = 0
    execution_errors = 0
    execution_successes = 0

    print("\nRAG 真实 LLM 回答评估")
    print("=" * 70)

    for index, case in enumerate(EVALUATION_CASES, start=1):
        query = case["query"]
        expected = case["expected"]
        session_id = f"rag_eval_{index}"

        try:
            answer = agent.run(query, session_id=session_id) or ""
            messages = agent.session_manager.get_messages(session_id)

            execution_successes += 1

            used_retrieval = any(
                call.get("function", {}).get("name")
                == "knowledge_search_context"
                for message in messages
                if message.get("role") == "assistant"
                for call in message.get("tool_calls", [])
            )

            if used_retrieval:
                tool_call_total += 1

            if expected is not None:
                answerable_total += 1

                # 初步指标：检查预期关键词是否全部出现在回答中
                correct = all(
                    keyword.lower() in answer.lower()
                    for keyword in expected
                )

                if correct:
                    answerable_correct += 1

                status = "关键词通过" if correct else "关键词未通过"

            else:
                unanswerable_total += 1

                refused = any(
                    marker in answer
                    for marker in REFUSAL_MARKERS
                )

                if refused:
                    unanswerable_refused += 1

                status = "拒答符合预期" if refused else "需要人工检查"

            print(f"\n[{index}] {query}")
            print(f"评估结果：{status}")
            print(
                f"调用知识检索工具：{'是' if used_retrieval else '否'}"
            )
            print(f"模型回答：{answer}")

        except Exception as exc:
            execution_errors += 1

            print(f"\n[{index}] {query}")
            print(f"执行失败：{type(exc).__name__}: {exc}")

    print("\n" + "=" * 70)
    print("评估汇总")

    print(
        f"执行成功数："
        f"{execution_successes}/{len(EVALUATION_CASES)}"
    )
    print(
        f"执行失败数："
        f"{execution_errors}/{len(EVALUATION_CASES)}"
    )
    print(
        f"可回答问题关键词通过率："
        f"{answerable_correct}/{answerable_total}"
    )
    print(
        f"无答案问题拒答率："
        f"{unanswerable_refused}/{unanswerable_total}"
    )
    print(
        f"调用知识检索工具的问题数："
        f"{tool_call_total}/{execution_successes}"
    )

    print(
        "\n注意：关键词匹配和拒答词匹配只是初步指标，"
        "不能完全代表回答正确性，需要人工复核。"
    )


if __name__ == "__main__":
    main()
