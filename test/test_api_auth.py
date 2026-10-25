"""HTTP 层测试：注册登录、JWT 鉴权、会话隔离、SSE 流式。

不打真实模型：把 chat_router 里的 build_agent 换成假 Agent，
这样测的是“接口是否把请求正确地交给了 Agent、响应格式对不对”。
"""

from __future__ import annotations

import json

import pytest

import api.routers.chat_router as chat_module
from agent.events import DELTA, DONE, META, AgentEvent
from agent.store import SessionStore


class FakeAgent:
    """假 Agent，但会真的写库——这样才能验证消息落库链路。"""

    def __init__(self, db, session_id: str) -> None:
        self.store = SessionStore(db)
        self.session_id = session_id

    async def run(self, message: str) -> str:
        await self.store.append(self.session_id, "user", message)
        await self.store.append(self.session_id, "assistant", f"echo: {message}")
        return f"echo: {message}"

    async def stream(self, message: str):
        await self.store.append(self.session_id, "user", message)
        yield AgentEvent(META, {"session_id": self.session_id, "model": "fake"})
        yield AgentEvent(DELTA, {"text": "你"})
        yield AgentEvent(DELTA, {"text": "好"})
        await self.store.append(self.session_id, "assistant", "你好")
        yield AgentEvent(DONE, {"answer": "你好", "session_id": self.session_id})


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name, data = "message", ""
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((name, json.loads(data)))
    return events


# ---------- 鉴权 ----------
async def test_register_returns_token(client):
    res = await client.post(
        "/auth/register", json={"username": "bob", "password": "bob123456"}
    )

    assert res.status_code == 201
    data = res.json()["data"]
    assert data["user"]["username"] == "bob"
    assert data["token"]["token_type"] == "bearer"
    assert data["token"]["expires_in"] > 0
    # 密码哈希绝不能出现在响应里
    assert "password" not in json.dumps(data["user"])


async def test_duplicate_username_rejected(client):
    payload = {"username": "carol", "password": "carol123456"}
    assert (await client.post("/auth/register", json=payload)).status_code == 201

    res = await client.post("/auth/register", json=payload)
    assert res.status_code == 409
    assert res.json()["code"] == 409


async def test_register_validates_input(client):
    res = await client.post(
        "/auth/register", json={"username": "ab", "password": "123"}
    )
    assert res.status_code == 422


async def test_login_success_and_failure(client):
    await client.post(
        "/auth/register", json={"username": "dave", "password": "dave123456"}
    )

    ok = await client.post(
        "/auth/login", json={"username": "dave", "password": "dave123456"}
    )
    assert ok.status_code == 200
    assert ok.json()["data"]["token"]["access_token"]

    bad = await client.post(
        "/auth/login", json={"username": "dave", "password": "wrong-pass"}
    )
    assert bad.status_code == 401

    ghost = await client.post(
        "/auth/login", json={"username": "nobody", "password": "whatever"}
    )
    assert ghost.status_code == 401


async def test_me_requires_token(client):
    assert (await client.get("/auth/me")).status_code == 401

    bad = await client.get(
        "/auth/me", headers={"Authorization": "Bearer not-a-jwt"}
    )
    assert bad.status_code == 401
    assert bad.headers.get("www-authenticate") == "Bearer"


async def test_me_returns_current_user(auth_client):
    res = await auth_client.get("/auth/me")

    assert res.status_code == 200
    assert res.json()["data"]["username"] == auth_client.username


# ---------- 会话 ----------
async def test_session_lifecycle(auth_client):
    created = await auth_client.post("/sessions", json={})
    session_id = created.json()["data"]["session_id"]

    listed = await auth_client.get("/sessions")
    assert session_id in [s["session_id"] for s in listed.json()["data"]]

    messages = await auth_client.get(f"/sessions/{session_id}/messages")
    assert messages.status_code == 200
    assert messages.json()["data"]["messages"] == []

    cleared = await auth_client.post(f"/sessions/{session_id}/clear")
    assert cleared.status_code == 200

    deleted = await auth_client.delete(f"/sessions/{session_id}")
    assert deleted.status_code == 200

    gone = await auth_client.get(f"/sessions/{session_id}/messages")
    assert gone.status_code == 404


async def test_rename_session(auth_client):
    created = await auth_client.post("/sessions", json={})
    session_id = created.json()["data"]["session_id"]

    res = await auth_client.patch(
        f"/sessions/{session_id}", json={"title": "  报销流程  "}
    )
    assert res.status_code == 200
    # 首尾空白和连续空白在校验器里洗掉
    assert res.json()["data"]["title"] == "报销流程"

    listed = await auth_client.get("/sessions")
    titles = {s["session_id"]: s["title"] for s in listed.json()["data"]}
    assert titles[session_id] == "报销流程"

    # 只有空白 → 拒绝（min_length 拦不住这种情况）
    blank = await auth_client.patch(f"/sessions/{session_id}", json={"title": "   "})
    assert blank.status_code == 422

    # 超长 → 拒绝
    long_title = await auth_client.patch(
        f"/sessions/{session_id}", json={"title": "长" * 61}
    )
    assert long_title.status_code == 422

    # 改不存在的会话 → 404
    missing = await auth_client.patch("/sessions/nope", json={"title": "x"})
    assert missing.status_code == 404


async def test_rename_does_not_reorder_list(auth_client):
    """改标题是纯展示操作，不该让会话跳到列表顶部。"""
    first = (await auth_client.post("/sessions", json={})).json()["data"]["session_id"]
    second = (await auth_client.post("/sessions", json={})).json()["data"]["session_id"]

    before = [s["session_id"] for s in (await auth_client.get("/sessions")).json()["data"]]
    assert before[:2] == [second, first]  # 按 updated_at 倒序

    await auth_client.patch(f"/sessions/{first}", json={"title": "最老的会话"})

    after = [s["session_id"] for s in (await auth_client.get("/sessions")).json()["data"]]
    assert after == before


async def test_rename_is_scoped_to_owner(auth_client):
    session_id = (await auth_client.post("/sessions", json={})).json()["data"]["session_id"]

    other = await auth_client.post(
        "/auth/register", json={"username": "mallory", "password": "mallory123"}
    )
    other_token = other.json()["data"]["token"]["access_token"]

    res = await auth_client.patch(
        f"/sessions/{session_id}",
        json={"title": "被别人改掉"},
        headers={"Authorization": f"Bearer {other_token}"},
    )
    # 和“不存在”同一个状态码，不暴露这个 id 真实存在
    assert res.status_code == 404

    listed = await auth_client.get("/sessions")
    titles = {s["session_id"]: s["title"] for s in listed.json()["data"]}
    assert titles[session_id] == "新对话"


async def test_sessions_are_isolated_between_users(auth_client):
    created = await auth_client.post("/sessions", json={})
    session_id = created.json()["data"]["session_id"]

    other = await auth_client.post(
        "/auth/register", json={"username": "eve", "password": "eve123456"}
    )
    other_token = other.json()["data"]["token"]["access_token"]

    res = await auth_client.get(
        f"/sessions/{session_id}/messages",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    # 别人的会话一律 404，不暴露“它存在”
    assert res.status_code == 404

    listed = await auth_client.get(
        "/sessions", headers={"Authorization": f"Bearer {other_token}"}
    )
    assert listed.json()["data"] == []


# ---------- 聊天 ----------
async def test_chat_returns_agent_answer(auth_client, monkeypatch):
    monkeypatch.setattr(
        chat_module, "build_agent", lambda db, session_id: FakeAgent(db, session_id)
    )

    res = await auth_client.post("/chat", json={"message": "你好"})

    assert res.status_code == 200
    data = res.json()["data"]
    assert data["answer"] == "echo: 你好"
    assert data["session_id"]


async def test_chat_requires_login(client):
    res = await client.post("/chat", json={"message": "你好"})
    assert res.status_code == 401


async def test_chat_stream_returns_sse(auth_client, monkeypatch):
    monkeypatch.setattr(
        chat_module, "build_agent", lambda db, session_id: FakeAgent(db, session_id)
    )

    res = await auth_client.post("/chat/stream", json={"message": "你好"})

    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(res.text)
    names = [name for name, _ in events]

    assert names[0] == META
    assert DELTA in names
    assert names[-1] == DONE
    assert events[-1][1]["answer"] == "你好"
    # 分块没有被聚合：两个 delta 事件各自独立
    assert [data["text"] for name, data in events if name == DELTA] == ["你", "好"]


async def test_stream_stores_messages(auth_client, monkeypatch):
    """流式完成后消息要落库，刷新页面还能看到历史。"""
    monkeypatch.setattr(
        chat_module, "build_agent", lambda db, session_id: FakeAgent(db, session_id)
    )

    res = await auth_client.post("/chat/stream", json={"message": "第一句"})
    session_id = res.headers["x-session-id"]

    history = await auth_client.get(f"/sessions/{session_id}/messages")
    messages = history.json()["data"]["messages"]

    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "第一句"
    assert messages[1]["content"] == "你好"
