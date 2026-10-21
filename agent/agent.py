import json
import os

from tools.schema import create_args_model
from pydantic import ValidationError
from agent.session import SessionManager

class Agent:

    def __init__(self, client, tools, tool_registry):
        self.client = client
        self.tools = tools
        self.tool_registry = tool_registry
        self.session_manager = SessionManager()

    def run(self, user_input, session_id="default"):

        messages = self.session_manager.get_messages(session_id)

        messages.append({
            "role": "user",
            "content": user_input
        })

        self.session_manager.save(session_id, messages)
        #最多10轮决策
        max_iterations = 10

        system_prompt = {
            "role": "system",
            "content": (
                "你是一个基于企业知识库回答问题的助手。"
                "涉及公司信息、招聘岗位和岗位要求时，必须调用 "
                "knowledge_search_context 工具检索。"
                "只能依据检索到的证据回答。"
                "如果工具返回 NO_RELEVANT_EVIDENCE，"
                "必须明确告知用户当前知识库无法确认答案，禁止猜测。"
            ),
        }

        if not messages or messages[0].get("role") != "system":
            messages.insert(0, system_prompt)

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            message = response.choices[0].message
            print("本轮 tool_calls：", message.tool_calls)
            if message.tool_calls:
                print("LLM 请求调用的工具：")
                for tool_call in message.tool_calls:
                    print(tool_call.function.name)
            else:
                print("LLM 本轮没有请求调用工具")

            messages.append(message.model_dump(exclude_none=True))

            # LLM 不需要工具
            if not message.tool_calls:
                self.session_manager.save(session_id, messages)
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

                    self.session_manager.save(session_id, messages)

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
                            result = tool(**validated_args.model_dump())
                            print("工具执行结果：", result)

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

            self.session_manager.save(session_id, messages)

        else:
            self.session_manager.save(session_id, messages)
            raise RuntimeError("Agent 超过最大决策轮数，已停止执行")

    def run_stream(self, user_input, session_id="default"):
        """逐段输出最终回答；工具调用轮次先执行工具，再继续生成。"""
        messages = self.session_manager.get_messages(session_id)
        messages.append({"role": "user", "content": user_input})
        self.session_manager.save(session_id, messages)

        system_prompt = {
            "role": "system",
            "content": (
                "你是一个基于企业知识库回答问题的助手。"
                "涉及公司信息、招聘岗位和岗位要求时，必须调用 "
                "knowledge_search_context 工具检索。"
                "只能依据检索到的证据回答。"
                "如果工具返回 NO_RELEVANT_EVIDENCE，"
                "必须明确告知用户当前知识库无法确认答案，禁止猜测。"
            ),
        }
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, system_prompt)

        for _ in range(10):
            response_stream = self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                stream=True,
            )
            content_parts = []
            streamed_tool_calls = {}

            try:
                for chunk in response_stream:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        content_parts.append(content)
                        yield content

                    for tool_delta in getattr(delta, "tool_calls", None) or []:
                        index = tool_delta.index
                        tool_call = streamed_tool_calls.setdefault(
                            index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tool_delta.id:
                            tool_call["id"] = tool_delta.id

                        function_delta = tool_delta.function
                        if function_delta:
                            if function_delta.name:
                                tool_call["function"]["name"] += function_delta.name
                            if function_delta.arguments:
                                tool_call["function"]["arguments"] += function_delta.arguments
            finally:
                close = getattr(response_stream, "close", None)
                if close:
                    close()

            message = {
                "role": "assistant",
                "content": "".join(content_parts) or None,
            }
            tool_calls = [
                streamed_tool_calls[index]
                for index in sorted(streamed_tool_calls)
            ]
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append({key: value for key, value in message.items() if value is not None})

            if not tool_calls:
                self.session_manager.save(session_id, messages)
                return

            for tool_call in tool_calls:
                result = self._execute_streamed_tool(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

            self.session_manager.save(session_id, messages)

        self.session_manager.save(session_id, messages)
        raise RuntimeError("Agent 超过最大决策轮数，已停止执行")

    def _execute_streamed_tool(self, tool_call):
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
            return tool(**validated_args.model_dump())
        except Exception as error:
            return {"error": f"工具执行失败: {error}"}
