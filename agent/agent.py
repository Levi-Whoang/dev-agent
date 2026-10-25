"""Agent 决策循环（全异步 + 支持流式）。

一次提问的流程没变：LLM 决策 → 调工具 → 结果回灌 → 再决策 → 出答案。
变的是三件事：
1. `await` 替代阻塞调用：等模型返回时事件循环能服务别的请求；
2. 每条消息即时落库，而不是最后整体覆盖；
3. 多了一个 `stream()`：边收边吐，用户不用盯着空白等十几秒。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from agent.events import DELTA, DONE, ERROR, META, TOOL_RESULT, TOOL_START, AgentEvent
from agent.prompts import SYSTEM_PROMPT
from agent.store import SessionStore
from agent.tools_runner import run_tool, tool_message_content

logger = logging.getLogger("dev-agent.agent")


class AgentError(RuntimeError):
    pass


class Agent:
    def __init__(
        self,
        client: Any,
        store: SessionStore,
        session_id: str,
        tools: list[dict] | None = None,
        tool_registry: dict[str, Callable] | None = None,
        model: str = "gpt-4o-mini",
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = 10,
    ) -> None:
        self.client = client
        self.store = store
        self.session_id = session_id
        self.tools = tools or []
        self.tool_registry = tool_registry or {}
        self.model = model
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations

    # ---------- 公共入口 ----------
    async def run(self, user_input: str) -> str:
        """非流式：等完整答案。适合脚本和测试。"""
        messages = await self._prepare(user_input)

        for step in range(1, self.max_iterations + 1):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools or None,
                tool_choice="auto" if self.tools else None,
            )
            message = response.choices[0].message
            tool_calls = self._dump_tool_calls(message)

            await self._save_assistant(message.content or "", tool_calls)
            messages.append(self._assistant_message(message.content or "", tool_calls))

            if not tool_calls:
                return message.content or ""

            logger.info("第 %d 轮：模型请求 %d 个工具", step, len(tool_calls))
            await self._execute_tools(messages, tool_calls)

        raise AgentError("Agent 超过最大决策轮数，已停止执行")

    async def stream(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """流式：逐段吐字，工具调用过程也作为事件发出去。"""
        messages = await self._prepare(user_input)

        yield AgentEvent(
            META,
            {"session_id": self.session_id, "model": self.model},
        )

        try:
            for step in range(1, self.max_iterations + 1):
                text_parts: list[str] = []
                collected: dict[int, dict[str, str]] = {}

                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools or None,
                    tool_choice="auto" if self.tools else None,
                    stream=True,
                )

                async for chunk in stream:
                    # 有些服务商会在末尾发一个只带 usage 的空 choices 块
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue

                    if delta.content:
                        text_parts.append(delta.content)
                        yield AgentEvent(DELTA, {"text": delta.content})

                    for call in delta.tool_calls or []:
                        self._collect_tool_delta(collected, call)

                text = "".join(text_parts)
                tool_calls = self._finalize_tool_calls(collected)

                await self._save_assistant(text, tool_calls)
                messages.append(self._assistant_message(text, tool_calls))

                if not tool_calls:
                    yield AgentEvent(
                        DONE,
                        {
                            "answer": text,
                            "session_id": self.session_id,
                            "steps": step,
                        },
                    )
                    return

                for call in tool_calls:
                    name = call["function"]["name"]
                    yield AgentEvent(
                        TOOL_START,
                        {"name": name, "arguments": call["function"]["arguments"]},
                    )
                    result = await run_tool(
                        name, call["function"]["arguments"], self.tool_registry
                    )
                    yield AgentEvent(TOOL_RESULT, {"name": name, "result": result})

                    content = tool_message_content(result)
                    await self.store.append(
                        self.session_id,
                        "tool",
                        content,
                        {"tool_call_id": call["id"], "name": name},
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": content,
                        }
                    )

            yield AgentEvent(ERROR, {"message": "Agent 超过最大决策轮数，已停止执行"})

        except Exception as exc:  # 流式响应已经开始，不能改状态码，只能用事件报错
            logger.exception("流式回答失败")
            yield AgentEvent(ERROR, {"message": str(exc)})

    # ---------- 内部 ----------
    async def _prepare(self, user_input: str) -> list[dict[str, Any]]:
        history = await self.store.history(self.session_id)
        await self.store.append(self.session_id, "user", user_input)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": user_input},
        ]
        return messages

    async def _save_assistant(
        self, content: str, tool_calls: list[dict[str, Any]]
    ) -> None:
        payload = {"tool_calls": tool_calls} if tool_calls else None
        await self.store.append(self.session_id, "assistant", content, payload)

    async def _execute_tools(
        self, messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]]
    ) -> None:
        for call in tool_calls:
            name = call["function"]["name"]
            result = await run_tool(
                name, call["function"]["arguments"], self.tool_registry
            )
            content = tool_message_content(result)

            await self.store.append(
                self.session_id,
                "tool",
                content,
                {"tool_call_id": call["id"], "name": name},
            )
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": content}
            )

    @staticmethod
    def _assistant_message(
        content: str, tool_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    @staticmethod
    def _dump_tool_calls(message: Any) -> list[dict[str, Any]]:
        calls = []
        for call in getattr(message, "tool_calls", None) or []:
            calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    },
                }
            )
        return calls

    @staticmethod
    def _collect_tool_delta(collected: dict[int, dict[str, str]], call: Any) -> None:
        index = call.index if getattr(call, "index", None) is not None else 0
        slot = collected.setdefault(index, {"id": "", "name": "", "arguments": ""})

        if getattr(call, "id", None):
            slot["id"] = call.id

        function = getattr(call, "function", None)
        if function is None:
            return

        # 名字和参数都可能被拆成多个 chunk 分片到达，所以用累加而不是赋值
        if getattr(function, "name", None):
            slot["name"] += function.name
        if getattr(function, "arguments", None):
            slot["arguments"] += function.arguments

    @staticmethod
    def _finalize_tool_calls(collected: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
        calls = []
        for index in sorted(collected):
            slot = collected[index]
            calls.append(
                {
                    "id": slot["id"] or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": slot["name"],
                        "arguments": slot["arguments"] or "{}",
                    },
                }
            )
        return calls
