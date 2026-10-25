"""命令行入口：不启动 HTTP 服务，直接在终端问一句。

用法：
    python main.py                 # 用默认问题跑一次完整 Agent 流程
    python main.py "公司在哪里？"   # 指定问题

它和 API 走的是同一套 Agent + 同一个数据库，只是省掉 HTTP 这一层，
适合调试提示词和工具调用。
"""

from __future__ import annotations

import asyncio
import sys

from agent.factory import build_agent
from auth.schemas import RegisterRequest
from auth.service import register_user, get_user_by_username
from database.base import dispose_db, get_session_factory, init_db
from database.models import User
from auth.passwords import hash_password

DEMO_USERNAME = "demo"
DEMO_SESSION = "cli-demo"


async def _ensure_demo_user(db) -> User:
    user = await get_user_by_username(db, DEMO_USERNAME)
    if user is not None:
        return user

    return await register_user(
        db,
        RegisterRequest(username=DEMO_USERNAME, password="demo123456"),
    )


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "公司在哪里？"

    await init_db()

    async with get_session_factory()() as db:
        user = await _ensure_demo_user(db)
        agent = build_agent(db, DEMO_SESSION)
        await SessionStore(db).get_or_create(DEMO_SESSION, user.id)

        print(f"\n用户提问：{question}\n" + "-" * 40)
        async for event in agent.stream(question):
            if event.event == "delta":
                print(event.data["text"], end="", flush=True)
            elif event.event == "tool_start":
                print(f"\n[调用工具] {event.data['name']}")
            elif event.event == "tool_result":
                print(f"[工具结果] {event.data['result']}")
            elif event.event == "error":
                print(f"\n[错误] {event.data['message']}")
        print("\n" + "-" * 40)

    await dispose_db()


if __name__ == "__main__":
    asyncio.run(main())
