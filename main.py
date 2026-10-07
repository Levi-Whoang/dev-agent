import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from agent.agent import Agent
from tools.registry import TOOLS
from tools.schema import tools_to_schema

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# ② 把工具交给模型

messages = [
    {
        "role": "user",
        "content": "查询订单10086，如果已经支付，就计算支付金额乘以2"
    }
]

agent = Agent(
    client=client,
    tools=tools_to_schema(TOOLS),
    tool_registry=TOOLS
)

result = agent.run(
    "查询订单10086，如果已经支付，就计算支付金额乘以2"
)

print(result)