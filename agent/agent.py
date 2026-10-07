import json
import os

from tools.schema import create_args_model


class Agent:

    def __init__(self, client, tools, tool_registry):
        self.client = client
        self.tools = tools
        self.tool_registry = tool_registry

    def run(self, user_input):

        messages = [
            {
                "role": "user",
                "content": user_input
            }
        ]

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

                    continue

                tool = self.tool_registry.get(tool_name)

                if tool is None:

                    result = {
                        "error": f"工具不存在: {tool_name}"
                    }

                else:

                    try:
                        args_model = create_args_model(tool)

                        validated_args = args_model(
                            **tool_arguments
                        )

                        result = tool(
                            **validated_args.model_dump()
                        )

                    except Exception as e:

                        result = {
                            "error": str(e)
                        }

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False
                    )
                })