import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import modules.database as database
from agent.session import SessionManager


@pytest.fixture
def sqlite_database(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(database, "get_engine", lambda: engine)
    monkeypatch.setattr(database, "_session_factory", factory)
    monkeypatch.setattr(database, "_initialized", False)
    yield engine
    engine.dispose()


def test_session_manager_initializes_database_lazily(sqlite_database):
    manager = SessionManager()

    assert database._initialized is False
    assert manager.get_messages("missing") == []
    assert database._initialized is True


def test_session_manager_persists_unicode_messages_and_session_metadata(
    sqlite_database,
):
    manager = SessionManager()
    messages = [{"role": "user", "content": "你好，帮我查一下。"}]
    manager.create("owner:session-1", owner_user_id=42, title="标题一")

    manager.save("owner:session-1", messages)

    assert manager.exists("owner:session-1") is True
    assert manager.get_messages("owner:session-1") == messages
    sessions = manager.list_sessions(owner_user_id=42)
    assert sessions[0]["session_id"] == "owner:session-1"
    assert sessions[0]["title"] == "标题一"
    assert sessions[0]["updated_at"] is not None


def test_session_manager_scopes_list_and_rename_to_owner(sqlite_database):
    manager = SessionManager()
    manager.create("alice:one", owner_user_id=1, title="旧标题")
    manager.create("bob:one", owner_user_id=2, title="其他用户")

    assert manager.rename("alice:one", owner_user_id=1, title="新标题") is True
    assert manager.rename("alice:one", owner_user_id=2, title="越权修改") is False
    assert [item["session_id"] for item in manager.list_sessions(1)] == [
        "alice:one"
    ]
    assert manager.list_sessions(1)[0]["title"] == "新标题"


def test_session_manager_clear_keeps_session_and_delete_removes_it(
    sqlite_database,
):
    manager = SessionManager()
    manager.create("session-1", owner_user_id=1, title="保留标题")
    manager.save("session-1", [{"role": "user", "content": "内容"}])

    manager.clear("session-1")
    assert manager.exists("session-1") is True
    assert manager.get_messages("session-1") == []
    assert manager.list_sessions(1)[0]["title"] == "保留标题"

    manager.delete("session-1")
    assert manager.exists("session-1") is False
