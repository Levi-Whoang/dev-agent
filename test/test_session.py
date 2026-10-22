import json
import sys
from types import ModuleType

import pytest

from agent.session import SessionManager


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query, params=None):
        if query.lstrip().startswith("CREATE TABLE") and self.connection.fail_ddl:
            self.connection.fail_ddl = False
            raise RuntimeError("ddl failed")
        self.connection.queries.append((query, params))

    def fetchone(self):
        return self.connection.fetchone_result

    def fetchall(self):
        return self.connection.fetchall_result


class FakeConnection:
    def __init__(self, *, fetchone=None, fetchall=None, fail_ddl=False):
        self.queries = []
        self.fetchone_result = fetchone
        self.fetchall_result = fetchall or []
        self.fail_ddl = fail_ddl
        self.commit_count = 0
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commit_count += 1

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def install_fake_pymysql(monkeypatch, connection_factory):
    driver = ModuleType("pymysql")
    driver.connect = connection_factory
    monkeypatch.setitem(sys.modules, "pymysql", driver)
    return driver


def test_session_manager_constructor_does_not_connect(monkeypatch):
    connections = []

    def connect(**config):
        connection = FakeConnection()
        connections.append(connection)
        return connection

    install_fake_pymysql(monkeypatch, connect)

    manager = SessionManager()

    assert connections == []
    assert manager._initialized is False

    first = manager._connect()
    first.close()
    second = manager._connect()
    second.close()

    create_statements = [
        query
        for connection in connections
        for query, _ in connection.queries
        if query.lstrip().startswith("CREATE TABLE")
    ]
    assert len(create_statements) == 1
    assert manager._initialized is True


def test_save_serializes_unicode_and_commits(monkeypatch):
    connections = []

    def connect(**config):
        connection = FakeConnection()
        connections.append(connection)
        return connection

    install_fake_pymysql(monkeypatch, connect)
    manager = SessionManager()
    messages = [{"role": "user", "content": "你好"}]

    manager.save("session-1", messages)

    connection = connections[0]
    query, params = connection.queries[-1]
    assert "INSERT INTO sessions" in query
    assert params[0] == "session-1"
    assert json.loads(params[1]) == messages
    assert "你好" in params[1]
    assert connection.commit_count == 2  # 建表和保存各提交一次
    assert connection.closed is True


@pytest.mark.parametrize(
    "stored_messages",
    [
        json.dumps([{"role": "user", "content": "你好"}], ensure_ascii=False),
        [{"role": "user", "content": "你好"}],
    ],
    ids=["json-string", "decoded-json"],
)
def test_get_messages_accepts_driver_json_formats(monkeypatch, stored_messages):
    connection = FakeConnection(fetchone=(stored_messages,))
    install_fake_pymysql(monkeypatch, lambda **config: connection)
    manager = SessionManager()

    messages = manager.get_messages("session-1")

    assert messages == [{"role": "user", "content": "你好"}]
    assert connection.queries[-1][1] == ("session-1",)
    assert connection.closed is True


def test_list_sessions_queries_only_the_requested_database_user(monkeypatch):
    connection = FakeConnection(
        fetchall=[
            ("owner-hash:session-1", "标题一", None),
            ("owner-hash:session-2", "标题二", None),
        ]
    )
    install_fake_pymysql(monkeypatch, lambda **config: connection)
    manager = SessionManager()

    sessions = manager.list_sessions(owner_user_id=42)

    query, params = connection.queries[-1]
    assert "WHERE owner_user_id = %s" in query
    assert params == (42,)
    assert [session["session_id"] for session in sessions] == [
        "owner-hash:session-1",
        "owner-hash:session-2",
    ]
    assert [session["title"] for session in sessions] == ["标题一", "标题二"]


def test_failed_lazy_initialization_closes_connection_and_can_retry(monkeypatch):
    connections = []

    def connect(**config):
        connection = FakeConnection(fail_ddl=not connections)
        connections.append(connection)
        return connection

    install_fake_pymysql(monkeypatch, connect)
    manager = SessionManager()

    with pytest.raises(RuntimeError, match="ddl failed"):
        manager._connect()

    assert connections[0].closed is True
    assert manager._initialized is False

    connection = manager._connect()
    assert manager._initialized is True
    assert any(query.lstrip().startswith("CREATE TABLE") for query, _ in connection.queries)
    connection.close()
