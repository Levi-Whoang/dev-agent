import os

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

agent = Agent(
    client=client,
    tools=tools_to_schema(TOOLS),
    tool_registry=TOOLS
)

