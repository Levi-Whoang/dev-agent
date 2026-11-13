import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import modules.database as database
from agent.session import SessionManager
from modules.models import ChatMessage, ChatSession


@pytest.fixture
async def sqlite_database(monkeypatch):
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(database, "get_engine", lambda: engine)
    monkeypatch.setattr(database, "_session_factory", factory)
    monkeypatch.setattr(database, "_initialized", False)
    monkeypatch.setattr(database, "_initialization_lock", asyncio.Lock())
    yield engine
    await engine.dispose()


# 并发追加需要独立连接，因此使用文件型 SQLite 而不是共享连接的 :memory:。
@pytest.fixture
async def sqlite_file_database(tmp_path, monkeypatch):
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'session.db'}",
        connect_args={"timeout": 30},
    )
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(database, "get_engine", lambda: engine)
    monkeypatch.setattr(database, "_session_factory", factory)
    monkeypatch.setattr(database, "_initialized", False)
    monkeypatch.setattr(database, "_initialization_lock", asyncio.Lock())
    yield engine
    await engine.dispose()


async def test_session_manager_initializes_database_lazily(sqlite_database):
    manager = SessionManager()

    assert database._initialized is False
    assert (await manager.get_messages("missing")) == []
    assert database._initialized is True


async def test_session_manager_persists_unicode_messages_and_session_metadata(
    sqlite_database,
):
    manager = SessionManager()
    messages = [{"role": "user", "content": "你好，帮我查一下。"}]
    await manager.create("owner:session-1", owner_user_id=42, title="标题一")

    await manager.save("owner:session-1", messages)

    assert (await manager.exists("owner:session-1")) is True
    assert (await manager.get_messages("owner:session-1")) == messages
    sessions = await manager.list_sessions(owner_user_id=42)
    assert sessions[0]["session_id"] == "owner:session-1"
    assert sessions[0]["title"] == "标题一"
    assert sessions[0]["updated_at"] is not None


async def test_session_manager_scopes_list_and_rename_to_owner(sqlite_database):
    manager = SessionManager()
    await manager.create("alice:one", owner_user_id=1, title="旧标题")
    await manager.create("bob:one", owner_user_id=2, title="其他用户")

    assert (await manager.rename("alice:one", owner_user_id=1, title="新标题")) is True
    assert (await manager.rename("alice:one", owner_user_id=2, title="越权修改")) is False
    assert [item["session_id"] for item in (await manager.list_sessions(1))] == [
        "alice:one"
    ]
    assert (await manager.list_sessions(1))[0]["title"] == "新标题"


async def test_session_manager_clear_keeps_session_and_delete_removes_it(
    sqlite_database,
):
    manager = SessionManager()
    await manager.create("session-1", owner_user_id=1, title="保留标题")
    await manager.save("session-1", [{"role": "user", "content": "内容"}])

    await manager.clear("session-1")
    assert (await manager.exists("session-1")) is True
    assert (await manager.get_messages("session-1")) == []
    assert (await manager.list_sessions(1))[0]["title"] == "保留标题"

    await manager.delete("session-1")
    assert (await manager.exists("session-1")) is False


async def test_session_manager_summary_roundtrip_and_clear(sqlite_database):
    manager = SessionManager()
    await manager.create("session-1", owner_user_id=1, title="保留标题")
    await manager.save("session-1", [{"role": "user", "content": "内容"}])

    assert (await manager.get_summary("session-1")) is None

    await manager.save_summary("session-1", "第一版摘要", 2)
    assert (await manager.get_summary("session-1")) == {
        "content": "第一版摘要",
        "covered_messages": 2,
    }

    # 再次保存应更新同一条摘要，而不是插入重复记录。
    await manager.save_summary("session-1", "第二版摘要", 4)
    assert (await manager.get_summary("session-1")) == {
        "content": "第二版摘要",
        "covered_messages": 4,
    }

    await manager.clear("session-1")
    assert (await manager.get_summary("session-1")) is None
    assert (await manager.exists("session-1")) is True
    assert (await manager.get_messages("session-1")) == []
    assert (await manager.list_sessions(1))[0]["title"] == "保留标题"


async def test_session_manager_summary_creates_missing_session(sqlite_database):
    manager = SessionManager()

    await manager.save_summary("session-2", "摘要", 3)

    assert (await manager.exists("session-2")) is True
    assert (await manager.get_messages("session-2")) == []
    assert (await manager.get_summary("session-2")) == {
        "content": "摘要",
        "covered_messages": 3,
    }


async def test_session_manager_delete_removes_summary(sqlite_database):
    manager = SessionManager()
    await manager.create("session-3", owner_user_id=1, title="待删除")
    await manager.save_summary("session-3", "摘要", 2)

    await manager.delete("session-3")

    assert (await manager.exists("session-3")) is False
    assert (await manager.get_summary("session-3")) is None


async def test_session_manager_appends_without_overwriting_other_requests(
    sqlite_database,
):
    manager = SessionManager()
    await manager.create("append-session", owner_user_id=1, title="并发")

    # 模拟两个并发请求在写入前读到同一份空历史。
    assert (await manager.get_messages("append-session")) == []
    assert (await manager.get_messages("append-session")) == []

    await manager.append_message(
        "append-session", {"role": "user", "content": "请求 A"}
    )
    await manager.append_message(
        "append-session", {"role": "user", "content": "请求 B"}
    )

    messages = await manager.get_messages("append-session")
    assert [message["content"] for message in messages] == ["请求 A", "请求 B"]


async def test_session_manager_keeps_messages_appended_concurrently(
    sqlite_file_database,
):
    import asyncio

    manager = SessionManager()
    await manager.create("concurrent-session", owner_user_id=1)

    await asyncio.gather(*[
        manager.append_message(
            "concurrent-session", {"role": "user", "content": f"并发消息{index}"}
        )
        for index in range(10)
    ])

    messages = await manager.get_messages("concurrent-session")
    assert len(messages) == 10
    assert {message["content"] for message in messages} == {
        f"并发消息{index}" for index in range(10)
    }


async def test_session_manager_creates_missing_session_under_concurrent_appends(
    sqlite_file_database,
):
    import asyncio

    manager = SessionManager()

    await asyncio.gather(*[
        manager.append_message(
            "fresh-session", {"role": "user", "content": f"消息{index}"}
        )
        for index in range(5)
    ])

    assert (await manager.exists("fresh-session")) is True
    assert len(await manager.get_messages("fresh-session")) == 5


async def test_session_manager_appends_full_message_fields(sqlite_database):
    manager = SessionManager()
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "calculate", "arguments": '{"a": 1}'},
    }

    # 会话不存在时追加也应自动创建。
    await manager.append_message(
        "tool-session",
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
    )
    await manager.append_message(
        "tool-session",
        {"role": "tool", "tool_call_id": "call_1", "name": "calculate", "content": "1"},
    )

    messages = await manager.get_messages("tool-session")
    assert messages == [
        {"role": "assistant", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "call_1", "name": "calculate", "content": "1"},
    ]


async def test_session_manager_migrates_legacy_json_messages_once(sqlite_database):
    manager = SessionManager()
    legacy = [
        {"role": "system", "content": "旧系统提示"},
        {"role": "user", "content": "旧问题"},
        {"role": "assistant", "content": "旧回答"},
    ]
    async with database.session_scope() as db:
        db.add(ChatSession(session_id="legacy-session", messages=legacy))

    assert (await manager.get_messages("legacy-session")) == legacy
    # 再次读取不应重复插入迁移数据。
    assert (await manager.get_messages("legacy-session")) == legacy

    async with database.session_scope() as db:
        count = await db.scalar(
            select(func.count())
            .select_from(ChatMessage)
            .where(ChatMessage.session_id == "legacy-session")
        )
        raw = await db.scalar(
            select(ChatSession.messages).where(
                ChatSession.session_id == "legacy-session"
            )
        )
    assert count == 3
    assert raw == []


async def test_session_manager_clear_and_delete_remove_message_rows(sqlite_database):
    manager = SessionManager()
    await manager.create("rows-session", owner_user_id=1)
    await manager.append_message("rows-session", {"role": "user", "content": "内容"})

    async def message_count():
        async with database.session_scope() as db:
            return await db.scalar(
                select(func.count())
                .select_from(ChatMessage)
                .where(ChatMessage.session_id == "rows-session")
            )

    assert (await message_count()) == 1
    await manager.clear("rows-session")
    assert (await message_count()) == 0

    await manager.append_message("rows-session", {"role": "user", "content": "再写"})
    await manager.delete("rows-session")
    assert (await message_count()) == 0


pytestmark = pytest.mark.anyio

# 测试统一使用项目的asyncio运行时。
@pytest.fixture
def anyio_backend():
    return "asyncio"
