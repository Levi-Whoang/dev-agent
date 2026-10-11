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

        self.session_manager.save()

        while True:

            response = self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            messages.append(message)

            # LLM 不需要工具
            if not message.tool_calls:
                self.session_manager.save()
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

                    self.session_manager.save()

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
                            "details": e.errors()
                        }

                    except Exception as e:
                        result = {
                            "error": f"工具执行前发生错误: {str(e)}"
                        }

                    else:
                        try:
                            result = tool(**validated_args.model_dump())
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