"""Agent 的 ReAct 循环（全异步）。

一个完整的 ReAct 循环由四件事构成：

    Thought       模型在正文里说明"我需要什么、打算查什么"
    Action        调用工具（结构化参数，走 function calling）
    Observation   工具结果（压缩后回灌）
    Termination   有答案 / 步数用完 / 检测到打转 / 出错

三条设计主线：

1. **只有一份循环。** 原来 run() 和 stream() 各写了一遍决策循环，
   任何一处改动都得记得改两遍，迟早分叉。现在两者都消费同一个 `_cycle()`：
   流式模式把增量吐成事件，非流式模式吞掉增量、只取最终答案。
   "要不要流式"只在调模型那一步分叉（`_one_turn`）。

2. **动作走 function calling，思考才用自然语言。**
   不用论文原版那种 `Action: search({"q":...})` 的文本解析——
   参数里多一个引号就崩。工具名和参数由 schema 校验，靠调用 id 配对。

3. **循环必须能停下来。** 光有 max_iterations 不够：
   模型会拿一模一样的参数反复重试同一个失败的工具，把预算烧光。
   所以还有一个重复动作守卫（见 agent/react.py 的 LoopGuard）。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from agent.events import (
    DELTA,
    DONE,
    ERROR,
    META,
    THOUGHT,
    TOOL_RESULT,
    TOOL_START,
    AgentEvent,
)
from agent.memory import MemoryManager
from agent.prompts import SYSTEM_PROMPT
from agent.react import (
    ANSWER,
    ERROR as REASON_ERROR,
    LOOP_DETECTED,
    MAX_STEPS,
    OBSERVATION_LIMIT,
    LoopGuard,
    ReactStep,
    ReactTrace,
    compress_observation,
    extract_thought,
)
from agent.store import SessionStore
from agent.tools_runner import run_tool, tool_message_content

logger = logging.getLogger("dev-agent.agent")

# 没显式配置记忆时的兜底窗口，和 config 里的默认值保持一致
DEFAULT_WINDOW_TOKENS = 3000
DEFAULT_MAX_TURNS = 12


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
        memory: MemoryManager | None = None,
        loop_threshold: int = 2,
        observation_limit: int = OBSERVATION_LIMIT,
    ) -> None:
        self.client = client
        self.store = store
        self.session_id = session_id
        self.tools = tools or []
        self.tool_registry = tool_registry or {}
        self.model = model
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.memory = memory
        self.loop_threshold = loop_threshold
        self.observation_limit = observation_limit

    # ---------- 公共入口 ----------
    async def run(self, user_input: str) -> str:
        """非流式：等完整答案。适合脚本和测试。

        内部走的是同一条循环，只是把增量丢掉了。
        为什么不让它单独写一份：两份循环的行为差异（比如"重复动作要不要中断"）
        会随着迭代慢慢分叉，而测试通常只覆盖其中一条。
        """
        answer = ""
        async for event in self._cycle(user_input, stream=False):
            if event.event == DONE:
                answer = event.data.get("answer") or ""
            elif event.event == ERROR:
                raise AgentError(event.data.get("message", "Agent 执行失败"))

        return answer

    async def stream(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """流式：边跑边把思考、动作、观测吐成事件。"""
        async for event in self._cycle(user_input, stream=True):
            yield event

    # ---------- 循环本体 ----------
    async def _cycle(
        self, user_input: str, *, stream: bool
    ) -> AsyncIterator[AgentEvent]:
        messages = await self._prepare(user_input)
        trace = ReactTrace()
        guard = LoopGuard(self.loop_threshold)

        yield AgentEvent(
            META,
            {"session_id": self.session_id, "model": self.model},
        )

        try:
            for index in range(1, self.max_iterations + 1):
                text = ""
                tool_calls: list[dict[str, Any]] = []

                # ---------- Thought + Action ----------
                async for kind, payload in self._one_turn(messages, stream=stream):
                    if kind == "delta":
                        yield AgentEvent(DELTA, {"text": payload})
                    else:
                        text, tool_calls = payload

                step = trace.new_step(extract_thought(text))
                await self._save_assistant(text, tool_calls)
                messages.append(self._assistant_message(text, tool_calls))

                # 没有动作 = 模型认为可以直接回答
                if not tool_calls:
                    step.answer = text
                    yield self._done(
                        answer=text,
                        termination=ANSWER,
                        trace=trace,
                        stream=stream,
                    )
                    return

                if step.thought:
                    # 已经在 delta 里吐过了，这里单独发一条是为了让不渲染增量的
                    # 客户端（脚本、日志）也能拿到推理过程
                    yield AgentEvent(
                        THOUGHT, {"text": step.thought, "step": index}
                    )

                logger.info(
                    "第 %d 步：思考 %d 字，请求 %d 个工具",
                    index,
                    len(step.thought),
                    len(tool_calls),
                )

                # ---------- Action → Observation ----------
                aborted = False
                for call in tool_calls:
                    name = call["function"]["name"]
                    arguments = call["function"]["arguments"]
                    step.actions.append({"name": name, "arguments": arguments})

                    verdict = guard.check(name, arguments)
                    if verdict == "abort":
                        logger.warning("检测到重复动作，中断循环：%s", name)
                        yield AgentEvent(
                            ERROR,
                            {
                                "message": "检测到重复调用同一个工具，已中断以免空转",
                                "step": index,
                                "termination": LOOP_DETECTED,
                            },
                        )
                        yield self._done(
                            answer="",
                            termination=LOOP_DETECTED,
                            trace=trace,
                            stream=stream,
                        )
                        aborted = True
                        break

                    yield AgentEvent(
                        TOOL_START,
                        {"name": name, "arguments": arguments, "step": index},
                    )

                    result = await run_tool(name, arguments, self.tool_registry)
                    ok = not (isinstance(result, dict) and result.get("error"))

                    content = compress_observation(
                        tool_message_content(result), self.observation_limit
                    )
                    if verdict == "warn":
                        # 把提醒贴在观测结果前面，而不是插一条额外的 system 消息：
                        # 不同服务商对"对话中途出现 system"的支持不一致
                        content = f"[系统提示] {guard.stuck_hint}\n\n{content}"

                    step.observations.append(
                        {"name": name, "ok": ok, "summary": content[:200]}
                    )

                    yield AgentEvent(
                        TOOL_RESULT,
                        {"name": name, "result": result, "step": index, "ok": ok},
                    )

                    # 落库 + 回灌。存的是压缩后的版本：
                    # 一条原始检索结果两三千字，全量进历史会迅速吃满上下文窗口
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

                if aborted:
                    return

            # ---------- 预算用完 ----------
            logger.warning("达到最大步数 %d，停止", self.max_iterations)
            yield AgentEvent(
                ERROR,
                {
                    "message": f"超过最大推理步数（{self.max_iterations}），已停止执行",
                    "termination": MAX_STEPS,
                },
            )
            yield self._done(
                answer="", termination=MAX_STEPS, trace=trace, stream=stream
            )

        except Exception as exc:  # noqa: BLE001
            # 流式响应头已经发出去了，改不了状态码，只能用事件报错。
            # 非流式没有这个限制，直接往上抛，保留完整 traceback。
            if not stream:
                raise
            logger.exception("ReAct 循环中断")
            yield AgentEvent(ERROR, {"message": str(exc), "termination": REASON_ERROR})
            yield self._done(
                answer="", termination=REASON_ERROR, trace=trace, stream=stream
            )

    # ---------- 单步：一次模型调用 ----------
    async def _one_turn(
        self, messages: list[dict[str, Any]], *, stream: bool
    ) -> AsyncIterator[tuple[str, Any]]:
        """产出 ("delta", 文本) 若干次，最后产出 ("result", (完整文本, 工具调用))。

        之所以做成异步生成器而不是普通函数：流式模式需要边收边吐，
        但"这一轮最终拿到了什么"要等流结束才知道。让同一个函数既吐增量
        又给结果，上层循环就不用为两种模式写两遍。
        """
        if not stream:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools or None,
                tool_choice="auto" if self.tools else None,
            )
            message = response.choices[0].message
            yield ("result", (message.content or "", self._dump_tool_calls(message)))
            return

        text_parts: list[str] = []
        collected: dict[int, dict[str, str]] = {}

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools or None,
            tool_choice="auto" if self.tools else None,
            stream=True,
        )

        async for chunk in completion:
            # 有些服务商会在末尾发一个只带 usage 的空 choices 块
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta is None:
                continue

            if delta.content:
                text_parts.append(delta.content)
                yield ("delta", delta.content)

            for call in delta.tool_calls or []:
                self._collect_tool_delta(collected, call)

        yield (
            "result",
            ("".join(text_parts), self._finalize_tool_calls(collected)),
        )

    # ---------- 收尾 ----------
    def _done(
        self, *, answer: str, termination: str, trace: ReactTrace, stream: bool
    ) -> AgentEvent:
        payload: dict[str, Any] = {
            "answer": answer,
            "session_id": self.session_id,
            "termination": termination,
            **trace.summary(),
        }
        if stream:
            # 把推理轨迹一起发出去，前端可以回放每一步
            payload["trace"] = [s.to_dict() for s in trace.steps]

        return AgentEvent(DONE, payload)

    # ---------- 内部 ----------
    @property
    def memory_manager(self) -> MemoryManager:
        """没注入就用保守默认值。

        为什么不直接读 config：Agent 在测试里是被单独构造的，
        偶尔还要用假 client；如果默认就去调摘要模型，测试会莫名其妙多出一次请求。
        需要完整记忆能力时，由 factory 显式注入一个配置好的 manager。
        """
        if self.memory is None:
            self.memory = MemoryManager(
                store=self.store,
                session_id=self.session_id,
                window_tokens=DEFAULT_WINDOW_TOKENS,
                max_turns=DEFAULT_MAX_TURNS,
                summary_enabled=False,
                longterm_enabled=False,
            )
        return self.memory

    async def _prepare(self, user_input: str) -> list[dict[str, Any]]:
        context = await self.memory_manager.prepare(self.system_prompt, user_input)
        await self.store.append(self.session_id, "user", user_input)

        logger.debug("记忆上下文：%s", context.stats)

        return [
            {"role": "system", "content": context.system_prompt},
            *context.history,
            {"role": "user", "content": user_input},
        ]

    async def _save_assistant(
        self, content: str, tool_calls: list[dict[str, Any]]
    ) -> None:
        payload = {"tool_calls": tool_calls} if tool_calls else None
        await self.store.append(self.session_id, "assistant", content, payload)

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
