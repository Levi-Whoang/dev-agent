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