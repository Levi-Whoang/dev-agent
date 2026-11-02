import importlib
import json
import sys
from datetime import datetime, timezone
from types import ModuleType
from unittest.mock import MagicMock

import anyio
import httpx
import pytest

from api.session_helpers import stored_session_id as make_stored_session_id

ALICE_HEADERS = {}
BOB_HEADERS = {}


# 从 HTTP SSE 正文提取事件，验证真实流响应中的完成或失败状态。
def parse_sse_events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


class MemorySessionManager:
    def __init__(self):
        self.sessions = {}

    def create(self, session_id, owner_user_id=None, title="新对话"):
        if session_id in self.sessions:
            raise ValueError("duplicate")
        self.sessions[session_id] = {
            "messages": [],
            "owner_user_id": owner_user_id,
            "title": title,
        }

    def exists(self, session_id):
        return session_id in self.sessions

    def get_messages(self, session_id):
        session = self.sessions.get(session_id)
        return list(session["messages"] if session else [])

    # 保存真实 Agent 产生的消息，保持内存会话所有权和标题不变。
    def save(self, session_id, messages):
        self.sessions[session_id]["messages"] = list(messages)

    def clear(self, session_id):
        self.sessions[session_id]["messages"] = []

    def delete(self, session_id):
        self.sessions.pop(session_id, None)

    def rename(self, session_id, owner_user_id, title):
        session = self.sessions.get(session_id)
        if not session or session["owner_user_id"] != owner_user_id:
            return False
        session["title"] = title
        return True

    def list_sessions(self, owner_user_id):
        return [
            {"session_id": key, "title": session["title"]}
            for key, session in self.sessions.items()
            if session["owner_user_id"] == owner_user_id
        ]


class MemoryAuthRepository:
    def __init__(self):
        self.users = {}
        self.tokens = {}
        self.next_user_id = 1
        self.login_failures = {}

    def create_user(self, username, password_hash):
        from auth.repository import DuplicateUsernameError

        if username in self.users:
            raise DuplicateUsernameError
        user_id = self.next_user_id
        self.next_user_id += 1
        self.users[username] = {
            "id": user_id,
            "username": username,
            "password_hash": password_hash,
        }
        return user_id

    def find_user(self, username):
        return self.users.get(username)

    def create_token(self, jti, user_id, expires_at):
        self.tokens[jti] = {
            "user_id": user_id,
            "expires_at": expires_at,
            "revoked": False,
        }

    def create_tokens(self, rows):
        for row in rows:
            self.create_token(row["jti"], row["user_id"], row["expires_at"])

    def rotate_tokens(self, old_refresh_jti, old_access_jti, username, rows):
        if not self.is_token_active(old_refresh_jti, username):
            return False
        self.tokens[old_refresh_jti]["revoked"] = True
        if old_access_jti in self.tokens:
            self.tokens[old_access_jti]["revoked"] = True
        self.create_tokens(rows)
        return True

    def login_failure_count(self, identity_hash, since):
        return len(self.login_failures.get(identity_hash, []))

    def record_login_failure(self, identity_hash, attempted_at):
        self.login_failures.setdefault(identity_hash, []).append(attempted_at)

    def clear_login_failures(self, identity_hash):
        self.login_failures.pop(identity_hash, None)

    def audit(self, event_type, username, source_hash, succeeded):
        return None

    def is_token_active(self, jti, username):
        token = self.tokens.get(jti)
        user = self.users.get(username)
        return bool(
            token
            and user
            and token["user_id"] == user["id"]
            and not token["revoked"]
            and token["expires_at"] > datetime.now(timezone.utc)
        )

    def revoke_token(self, jti, username, related_jti=None):
        token = self.tokens.get(jti)
        user = self.users.get(username)
        if token and user and token["user_id"] == user["id"]:
            token["revoked"] = True
            if related_jti in self.tokens:
                self.tokens[related_jti]["revoked"] = True


class ASGIFakeAgent:
    def __init__(self):
        self.session_manager = MemorySessionManager()
        self.calls = []

    def run(self, message, session_id):
        self.calls.append(("json", message, session_id))
        return f"完整回答：{message}"

    def run_stream(self, message, session_id):
        self.calls.append(("stream", message, session_id))
        yield "第一段"
        yield "第二段"


@pytest.fixture
def asgi_app(monkeypatch):
    from auth.passwords import hash_password
    import auth.jwt_auth as jwt_auth

    auth_repository = MemoryAuthRepository()
    auth_repository.create_user(
        "alice", hash_password("alice-password", iterations=100_000)
    )
    auth_repository.create_user(
        "bob", hash_password("bob-password", iterations=100_000)
    )
    monkeypatch.setattr(jwt_auth, "auth_repository", auth_repository)
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-with-more-than-32-characters")
    monkeypatch.setenv("AUTH_REGISTRATION_CODE", "test-registration-code")
    fake_agent = ASGIFakeAgent()
    fake_runtime = ModuleType("agent.runtime")
    fake_runtime.agent = fake_agent
    fake_logs = ModuleType("logs")
    fake_logs.__path__ = []
    fake_logging = ModuleType("logs.logging_config")
    fake_logging.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "logs", fake_logs)
    monkeypatch.setitem(sys.modules, "logs.logging_config", fake_logging)
    monkeypatch.setitem(sys.modules, "agent.runtime", fake_runtime)
    for module_name in (
        "app", "app.main", "api.routes", "api.routes.sessions",
        "api.routes.users", "api.routes.documents", "app.errors",
    ):
        sys.modules.pop(module_name, None)
    app_module = importlib.import_module("app")
    global ALICE_HEADERS, BOB_HEADERS
    alice_token, _ = jwt_auth.create_access_token(
        "alice", auth_repository.users["alice"]["id"]
    )
    bob_token, _ = jwt_auth.create_access_token(
        "bob", auth_repository.users["bob"]["id"]
    )
    ALICE_HEADERS = {"Authorization": f"Bearer {alice_token}"}
    BOB_HEADERS = {"Authorization": f"Bearer {bob_token}"}

    try:
        yield app_module.app, fake_agent
    finally:
        for module_name in (
            "app", "app.main", "api.routes", "api.routes.sessions",
            "api.routes.users", "api.routes.documents", "app.errors",
        ):
            sys.modules.pop(module_name, None)


def test_asgi_health_and_sse_chat_with_credentials(asgi_app):
    app, fake_agent = asgi_app
    fake_agent.session_manager.sessions[make_stored_session_id("alice", "asgi-1")] = {
        "owner_user_id": 1,
        "title": "查询办公地点",
        "messages": [
        {"role": "user", "content": "查询办公地点"},
        {
            "role": "tool",
            "content": json.dumps({
                "citations": [
                    {
                        "source": "company.txt",
                        "content": "办公室位于深圳市南山区。",
                        "score": 0.91,
                        "dense_score": 0.88,
                        "keyword_score": 1.2,
                    }
                ]
            }, ensure_ascii=False),
        },
        ],
    }

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=ALICE_HEADERS,
        ) as client:
            health = await client.get("/health")
            chat = await client.post(
                "/chat",
                json={"message": "测试流式", "session_id": "asgi-1"},
            )
            return health, chat

    health, chat = anyio.run(make_requests)

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert chat.status_code == 200
    assert chat.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line.removeprefix("data: "))
        for line in chat.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events == [
        {"type": "delta", "content": "第一段"},
        {"type": "delta", "content": "第二段"},
        {
            "type": "done",
            "session_id": "asgi-1",
            "citations": [
                {
                    "source": "company.txt",
                    "content": "办公室位于深圳市南山区。",
                    "score": 0.91,
                    "dense_score": 0.88,
                    "keyword_score": 1.2,
                }
            ],
        },
    ]
    assert fake_agent.calls == [
        ("stream", "测试流式", make_stored_session_id("alice", "asgi-1"))
    ]


def test_asgi_json_chat_returns_citations_from_latest_turn(asgi_app):
    app, fake_agent = asgi_app
    citation = {
        "source": "faq.txt",
        "content": "办公地点位于深圳市南山区。",
        "score": 0.82,
        "dense_score": 0.82,
        "keyword_score": 0.0,
    }
    fake_agent.session_manager.sessions[
        make_stored_session_id("alice", "asgi-citation")
    ] = {
        "owner_user_id": 1,
        "title": "公司在哪里？",
        "messages": [
            {"role": "user", "content": "公司在哪里？"},
            {"role": "tool", "content": json.dumps({"citations": [citation]})},
        ],
    }

    async def make_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=ALICE_HEADERS,
        ) as client:
            return await client.post(
                "/chat",
                json={
                    "message": "公司在哪里？",
                    "session_id": "asgi-citation",
                },
            )

    response = anyio.run(make_request)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1]["type"] == "done"
    assert events[-1]["citations"] == [citation]


def test_asgi_chat_always_streams_and_validates_requests(asgi_app):
    app, fake_agent = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=ALICE_HEADERS,
        ) as client:
            chat = await client.post(
                "/chat",
                json={
                    "message": "兼容请求",
                    "session_id": "asgi-2",
                },
            )
            invalid = await client.post("/chat", json={"message": ""})
            return chat, invalid

    chat, invalid = anyio.run(make_requests)

    assert chat.status_code == 200
    assert chat.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[6:])
        for line in chat.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events == [
        {"type": "delta", "content": "第一段"},
        {"type": "delta", "content": "第二段"},
        {"type": "done", "session_id": "asgi-2", "citations": []},
    ]
    assert invalid.status_code == 422
    assert invalid.json()["code"] == 422
    assert fake_agent.calls == [
        ("stream", "兼容请求", make_stored_session_id("alice", "asgi-2"))
    ]


def test_asgi_session_routes_use_injected_memory_store(asgi_app):
    app, _ = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=ALICE_HEADERS,
        ) as client:
            created = await client.post("/sessions", json={"session_id": "asgi-3"})
            listed = await client.get("/sessions")
            deleted = await client.delete("/sessions/asgi-3")
            missing = await client.get("/sessions/asgi-3/messages")
            return created, listed, deleted, missing

    created, listed, deleted, missing = anyio.run(make_requests)

    assert created.status_code == 200
    assert listed.json()["data"] == [{"session_id": "asgi-3", "title": "新对话"}]
    assert deleted.status_code == 200
    assert missing.status_code == 404


def test_asgi_requires_authentication_except_health(asgi_app):
    app, _ = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            health = await client.get("/health")
            sessions = await client.get("/sessions")
            bad_token = await client.get(
                "/sessions",
                headers={"Authorization": "Bearer wrong-token"},
            )
            return health, sessions, bad_token

    health, sessions, bad_token = anyio.run(make_requests)
    assert health.status_code == 200
    assert sessions.status_code == 401
    assert sessions.headers["www-authenticate"] == "Bearer"
    assert bad_token.status_code == 401


def test_asgi_login_issues_jwt_and_rejects_invalid_credentials(asgi_app):
    app, _ = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            registered = await client.post(
                "/auth/register",
                json={"username": "new-user", "password": "new-password", "verification_code": "test-registration-code"},
            )
            duplicate = await client.post(
                "/auth/register",
                json={"username": "new-user", "password": "new-password", "verification_code": "test-registration-code"},
            )
            registered_login = await client.post(
                "/auth/token",
                json={"username": "new-user", "password": "new-password"},
            )
            success = await client.post(
                "/auth/token",
                json={"username": "alice", "password": "alice-password"},
            )
            failure = await client.post(
                "/auth/token",
                json={"username": "alice", "password": "wrong-password"},
            )
            logged_out = await client.post(
                "/auth/logout",
                headers={
                    "Authorization": (
                        "Bearer " + success.json()["data"]["access_token"]
                    )
                },
            )
            revoked_request = await client.get(
                "/sessions",
                headers={
                    "Authorization": (
                        "Bearer " + success.json()["data"]["access_token"]
                    )
                },
            )
            return (
                registered,
                duplicate,
                registered_login,
                success,
                failure,
                logged_out,
                revoked_request,
            )

    (
        registered,
        duplicate,
        registered_login,
        success,
        failure,
        logged_out,
        revoked_request,
    ) = anyio.run(make_requests)
    assert registered.status_code == 201
    assert registered.json()["data"]["username"] == "new-user"
    assert duplicate.status_code == 409
    assert registered_login.status_code == 200
    assert success.status_code == 200
    assert success.json()["data"]["token_type"] == "bearer"
    assert success.json()["data"]["expires_in"] == 1800
    assert success.json()["data"]["access_token"]
    assert failure.status_code == 401
    assert logged_out.status_code == 200
    assert revoked_request.status_code == 401


def test_asgi_session_ownership_and_list_are_scoped_per_user(asgi_app):
    app, fake_agent = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            created = await client.post(
                "/sessions",
                json={"session_id": "alice-private"},
                headers=ALICE_HEADERS,
            )
            alice_list = await client.get("/sessions", headers=ALICE_HEADERS)
            bob_list = await client.get("/sessions", headers=BOB_HEADERS)
            bob_read = await client.get(
                "/sessions/alice-private/messages",
                headers=BOB_HEADERS,
            )
            bob_clear = await client.post(
                "/sessions/alice-private/clear",
                headers=BOB_HEADERS,
            )
            bob_delete = await client.delete(
                "/sessions/alice-private",
                headers=BOB_HEADERS,
            )
            return created, alice_list, bob_list, bob_read, bob_clear, bob_delete

    created, alice_list, bob_list, bob_read, bob_clear, bob_delete = anyio.run(
        make_requests
    )
    assert created.status_code == 200
    assert [item["session_id"] for item in alice_list.json()["data"]] == [
        "alice-private"
    ]
    assert bob_list.json()["data"] == []
    assert bob_read.status_code == 404
    assert bob_clear.status_code == 404
    assert bob_delete.status_code == 404
    assert len(fake_agent.session_manager.sessions) == 1


def test_asgi_chat_without_session_id_creates_unique_user_sessions(asgi_app):
    app, fake_agent = asgi_app

    async def make_requests():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            alice = await client.post(
                "/chat",
                json={"message": "Alice 的新对话"},
                headers=ALICE_HEADERS,
            )
            bob = await client.post(
                "/chat",
                json={"message": "Bob 的新对话"},
                headers=BOB_HEADERS,
            )
            return alice, bob

    alice, bob = anyio.run(make_requests)
    def completed_session_id(response):
        return next(
            json.loads(line[6:])["session_id"]
            for line in response.text.splitlines()
            if line.startswith("data: ")
            and json.loads(line[6:]).get("type") == "done"
        )

    alice_session_id = completed_session_id(alice)
    bob_session_id = completed_session_id(bob)
    assert alice.status_code == 200
    assert bob.status_code == 200
    assert alice_session_id != bob_session_id
    assert len(fake_agent.session_manager.sessions) == 2
    assert all(call[2].count(":") == 1 for call in fake_agent.calls)


# 用真实 Agent 和检索上下文工具验证引用只编码一次且 SSE 正常完成。
@pytest.mark.parametrize("has_evidence", [True, False])
def test_real_rag_tool_reaches_http_citations(asgi_app, monkeypatch, has_evidence):
    from agent.agent import Agent
    import tools.knowledge as knowledge
    from test.test_agent import stream_chunk, stream_tool_call_delta

    app, fake_agent = asgi_app
    observed_owners = []

    # 保留真实工具协议，只替换昂贵的 embedding 检索。
    def search(query):
        observed_owners.append(knowledge._CURRENT_KNOWLEDGE_OWNER.get())
        return [{"source": "company.txt", "content": "公司在深圳。", "score": 0.9}] if has_evidence else []

    monkeypatch.setattr(knowledge, "knowledge_search", search)
    llm = MagicMock()
    llm.chat.completions.create.side_effect = [
        iter([stream_chunk(tool_calls=[stream_tool_call_delta(
            call_id="rag-call", name="knowledge_search_context", arguments='{"query":"公司在哪里"}',
        )])]),
        iter([stream_chunk(content="公司在深圳。" if has_evidence else "当前知识库无法确认。")]),
    ]
    real_agent = Agent(llm, [], {"knowledge_search_context": knowledge.knowledge_search_context})
    real_agent.session_manager = fake_agent.session_manager
    routes = importlib.import_module("api.routes.sessions")
    monkeypatch.setattr(routes, "agent", real_agent)

    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/chat", headers=ALICE_HEADERS, json={"message": "公司在哪里"})

    response = anyio.run(request)
    events = parse_sse_events(response)
    assert [event["type"] for event in events] == ["delta", "done"]
    assert observed_owners == [1]
    assert len(events[-1]["citations"]) == int(has_evidence)
    tool_messages = [m for record in fake_agent.session_manager.sessions.values() for m in record["messages"] if m["role"] == "tool"]
    assert isinstance(json.loads(tool_messages[0]["content"]), dict)


# 验证不同用户的作用域在每次线程池迭代中恢复，生成异常只有 error 终态。
def test_http_stream_scope_and_error_terminal(asgi_app, monkeypatch):
    import tools.knowledge as knowledge
    app, fake_agent = asgi_app
    observed = []

    # 在多次恢复生成器时记录用户范围，并模拟生成过程失败。
    def stream(message, session_id):
        observed.append(knowledge._CURRENT_KNOWLEDGE_OWNER.get())
        yield "片段一"
        observed.append(knowledge._CURRENT_KNOWLEDGE_OWNER.get())
        if message == "失败":
            raise RuntimeError("模拟模型故障")
        yield "片段二"
        observed.append(knowledge._CURRENT_KNOWLEDGE_OWNER.get())

    monkeypatch.setattr(fake_agent, "run_stream", stream)

    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            first = await client.post("/chat", headers=ALICE_HEADERS, json={"message": "成功"})
            second = await client.post("/chat", headers=BOB_HEADERS, json={"message": "失败"})
            return first, second

    first, second = anyio.run(request)
    assert [e["type"] for e in parse_sse_events(first)] == ["delta", "delta", "done"]
    assert [e["type"] for e in parse_sse_events(second)] == ["delta", "error"]
    assert observed == [1, 1, 1, 2, 2]
    assert knowledge._CURRENT_KNOWLEDGE_OWNER.get() is None


# 验证 async 文档接口将同步服务放到线程池执行。
@pytest.mark.parametrize("method", ["post", "put"])
def test_document_write_service_runs_outside_event_loop(asgi_app, monkeypatch, method):
    import threading
    app, _ = asgi_app
    routes = importlib.import_module("api.routes.documents")
    service_threads = []

    # 服务替身只记录执行线程，不接触真实文档和数据库。
    def service(*args):
        service_threads.append(threading.get_ident())
        return {"document_id": "test-document"}

    monkeypatch.setattr(routes, "create_document" if method == "post" else "replace_document_service", service)

    async def request():
        event_thread = threading.get_ident()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.request(method, "/knowledge/documents" + ("/test-document" if method == "put" else ""),
                headers=ALICE_HEADERS, files={"file": ("demo.txt", b"demo", "text/plain")})
            return event_thread, response

    event_thread, response = anyio.run(request)
    assert response.status_code == (201 if method == "post" else 200)
    assert len(service_threads) == 1
    assert service_threads[0] != event_thread
