import json
import os
import re
import time
import logging

logger = logging.getLogger(__name__)

from tools.schema import create_args_model
from pydantic import ValidationError
from agent.session import SessionManager
from agent.control import execute_tool, await_or_stop, GenerationStopped

class Agent:

    _system_prompt = {
        "role": "system",
        "content": (
            "你是一个基于企业知识库回答问题的助手。"
            "涉及公司信息、招聘岗位和岗位要求时，必须调用 "
            "knowledge_search_context 工具检索。"
            "只能依据检索到的证据回答。"
            "如果工具返回 NO_RELEVANT_EVIDENCE，"
            "必须明确告知用户当前知识库无法确认答案，禁止猜测。"
            "请在内部完成分析，不要输出或复述思维链、推理草稿、工具规划或隐藏分析；"
            "只输出面向用户的最终答复。"
        ),
    }

    @staticmethod
    def _public_answer(content: str) -> str:
        """移除常见模型思维标签及其包裹的内部分析。"""
        for tag in ("think", "analysis", "reasoning"):
            content = re.sub(
                rf"<{tag}\b[^>]*>.*?(?:</{tag}\s*>|$)",
                "",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
        return content.strip()

    def __init__(self, client, tools, tool_registry):
        self.client = client
        self.tools = tools
        self.tool_registry = tool_registry
        self.session_manager = SessionManager()

    # 异步执行模型决策和工具调用，返回完整回答供内部评估使用。
    async def run(self, user_input, session_id):

        messages = await self.session_manager.get_messages(session_id)

        messages.append({
            "role": "user",
            "content": user_input
        })

        await self.session_manager.save(session_id, messages)
        #最多10轮决策
        max_iterations = 10

        if not messages or messages[0].get("role") != "system":
            messages.insert(0, self._system_prompt)
        else:
            messages[0] = self._system_prompt

        for _ in range(max_iterations):
            response = await self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            message = response.choices[0].message
            saved_message = message.model_dump(exclude_none=True)
            if message.tool_calls:
                saved_message.pop("content", None)
            messages.append(saved_message)

            # LLM 不需要工具
            if not message.tool_calls:
                await self.session_manager.save(session_id, messages)
                return message.content

            # LLM 要调用工具
            for tool_call in message.tool_calls:

                tool_name = tool_call.function.name

                try:
                    tool_arguments = json.loads(
                        tool_call.function.arguments
                    )

                except json.JSONDecodeError as e:
                    result = {
                        "error": f"工具参数格式错误: {str(e)}"
                    }

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            result,
                            ensure_ascii=False
                        )
                    })

                    await self.session_manager.save(session_id, messages)

                    continue

                tool = self.tool_registry.get(tool_name)

                if tool is None:

                    result = {
                        "error": f"工具不存在: {tool_name}"
                    }

                else:

                    try:
                        args_model = create_args_model(tool)
                        validated_args = args_model(**tool_arguments)

                    except ValidationError as e:
                        result = {
                            "error": "工具参数校验失败",
                            "details":  e.errors(include_context=False)
                        }

                    except Exception as e:
                        result = {
                            "error": f"工具执行前发生错误: {str(e)}"
                        }

                    else:
                        try:
                            result = await execute_tool(tool, validated_args.model_dump(), float(os.getenv("TOOL_TIMEOUT_SECONDS", "15")))
                        except Exception as e:
                            result = {
                                "error": f"工具执行失败: {str(e)}"
                            }

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False
                    )
                })

            await self.session_manager.save(session_id, messages)

        else:
            await self.session_manager.save(session_id, messages)
            raise RuntimeError("Agent 超过最大决策轮数，已停止执行")

    # 工具决策与最终输出分离，最终阶段不注册工具以防输出工具草稿。
    async def run_stream(self, user_input, session_id, stop_event=None):
        from threading import Event
        from agent.stream_filter import PublicTextFilter
        stop_event = stop_event or Event()
        started = time.perf_counter()
        messages = await self.session_manager.get_messages(session_id)
        messages.append({"role": "user", "content": user_input})
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, self._system_prompt)
        else:
            messages[0] = self._system_prompt
        await self.session_manager.save(session_id, messages)
        for _ in range(10):
            if stop_event.is_set():
                return
            decision_started = time.perf_counter()
            response = await await_or_stop(self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL"), messages=messages, tools=self.tools,
                tool_choice="auto", timeout=20,
            ), stop_event)
            logger.info("llm_decision session_id=%s duration=%.3f", session_id, time.perf_counter()-decision_started)
            decision = response.choices[0].message
            if not decision.tool_calls:
                break
            saved = decision.model_dump(exclude_none=True)
            saved.pop("content", None)
            messages.append(saved)
            for call in saved["tool_calls"]:
                result = await self._execute_streamed_tool(call, stop_event)
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, ensure_ascii=False)})
            await self.session_manager.save(session_id, messages)
            # 知识证据已经就绪，直接进入最终流式回答，避免再生成并丢弃一遍完整答案。
            if all(call["function"]["name"] in {"knowledge_search", "knowledge_search_context"} for call in saved["tool_calls"]):
                break
        else:
            raise RuntimeError("Agent 超过最大决策轮数")
        if stop_event.is_set():
            return
        response_stream = await await_or_stop(self.client.chat.completions.create(
            model=os.getenv("LLM_MODEL"), messages=messages, stream=True, timeout=20,
        ), stop_event)
        public_filter = PublicTextFilter()
        parts = []
        try:
            while not stop_event.is_set():
                try:
                    chunk = await await_or_stop(anext(response_stream), stop_event)
                except (StopAsyncIteration, GenerationStopped):
                    break
                if stop_event.is_set():
                    break
                if not chunk.choices:
                    continue
                content = public_filter.feed(getattr(chunk.choices[0].delta, "content", None) or "")
                if content:
                    if not parts:
                        logger.info("agent_first_text session_id=%s duration=%.3f", session_id, time.perf_counter()-started)
                    parts.append(content)
                    yield content
        finally:
            import anyio
            # 断连取消期间仍关闭连接，并保存已经输出的正文。
            with anyio.CancelScope(shield=True):
                try:
                    await response_stream.close()
                finally:
                    if parts:
                        messages.append({"role": "assistant", "content": "".join(parts)})
                        await self.session_manager.save(session_id, messages)

    # 校验工具参数并等待有界线程池执行结果。

    async def _execute_streamed_tool(self, tool_call, stop_event=None):
        function = tool_call["function"]
        tool_name = function["name"]

        try:
            tool_arguments = json.loads(function["arguments"])
        except json.JSONDecodeError as error:
            return {"error": f"工具参数格式错误: {error}"}

        tool = self.tool_registry.get(tool_name)
        if tool is None:
            return {"error": f"工具不存在: {tool_name}"}

        try:
            args_model = create_args_model(tool)
            validated_args = args_model(**tool_arguments)
        except ValidationError as error:
            return {
                "error": "工具参数校验失败",
                "details": error.errors(include_context=False),
            }
        except Exception as error:
            return {"error": f"工具执行前发生错误: {error}"}

        try:
            from agent.control import execute_tool
            return await execute_tool(tool, validated_args.model_dump(),
                                float(os.getenv("TOOL_TIMEOUT_SECONDS", "15")), stop_event)
        except Exception as error:
            return {"error": f"工具执行失败: {error}"}
