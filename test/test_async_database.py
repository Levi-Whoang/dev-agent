"""异步数据库事务、认证与文档服务集成检查。"""
from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy import select
from modules.database import session_scope
from modules.models import AuthUser
from auth.repository import AuthRepository, DuplicateUsernameError
from test.test_session import sqlite_database


# 异步仓储重复用户名回滚后仍可继续使用新的Session。
async def test_repository_commit_and_rollback(sqlite_database):
    repo = AuthRepository()
    await repo.create_user("alice", "hash")
    with pytest.raises(DuplicateUsernameError):
        await repo.create_user("alice", "another")
    assert ((await repo.find_user("alice")))["password_hash"] == "hash"
    await repo.create_user("bob", "hash")
    assert (await repo.user_exists("bob"))

# 刷新令牌原子轮换后旧令牌不能再次使用，撤销可立即生效。
async def test_real_token_rotation(sqlite_database):
    repo = AuthRepository()
    uid = await repo.create_user("alice", "hash")
    expiry = datetime.now(timezone.utc)+timedelta(minutes=30)
    rows = [{"jti":name, "user_id":uid, "expires_at":expiry} for name in ["access1","refresh1"]]
    await repo.create_tokens(rows)
    new = [{**row,"jti":row["jti"].replace("1","2")} for row in rows]
    assert (await repo.rotate_tokens("refresh1","access1","alice",new))
    assert not (await repo.rotate_tokens("refresh1","access1","alice",new))
    assert not (await repo.is_token_active("access1","alice"))
    assert (await repo.is_token_active("access2","alice"))
    await repo.revoke_token("access2","alice","refresh2")
    assert not (await repo.is_token_active("refresh2","alice"))

# 异步事务发生异常时回滚未提交数据，之后仍可正常查询。
async def test_failed_transaction_rolls_back(sqlite_database):
    with pytest.raises(RuntimeError, match="模拟失败"):
        async with session_scope() as db:
            db.add(AuthUser(username="cancelled", password_hash="hash"))
            await db.flush()
            raise RuntimeError("模拟失败")
    assert (await AuthRepository().find_user("cancelled")) is None


# 取消事务任务时回滚未提交数据，后续数据库操作仍可执行。
async def test_cancelled_transaction_rolls_back(sqlite_database):
    import asyncio
    entered = asyncio.Event()

    # 暂停在flush之后，验证取消不会意外提交已经发送的SQL。
    async def transaction():
        async with session_scope() as db:
            db.add(AuthUser(username="cancelled-task", password_hash="hash"))
            await db.flush()
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(transaction())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await AuthRepository().find_user("cancelled-task") is None

# 两个作用域提供独立Session，不把会话对象共享给并发任务。
async def test_sessions_are_independent(sqlite_database):
    await AuthRepository().create_user("alice", "hash")
    async with session_scope() as first:
        async with session_scope() as second:
            assert first is not second
            assert (await second.scalar(select(AuthUser.username))) == "alice"

# 文档CRUD使用真实AsyncSession，所有者过滤与文件解析分工保持有效。
async def test_document_crud(sqlite_database, tmp_path, monkeypatch):
    from services import documents
    from tools import knowledge
    monkeypatch.setattr(documents,"KNOWLEDGE_DIR",tmp_path)
    monkeypatch.setattr(documents,"_FILES_DIR",tmp_path/".originals")
    monkeypatch.setattr(documents,"_TEXT_DIR",tmp_path/"documents")
    repo = AuthRepository()
    uid = await repo.create_user("alice","hash")
    other = await repo.create_user("bob","hash")
    result = await documents.create_document(uid,"demo.txt","公开示例".encode())
    docid = result["document_id"]
    assert len((await documents.list_documents(uid))) == 1
    assert (await documents.list_documents(other)) == []
    with pytest.raises(documents.DocumentNotFoundError):
        await documents.get_document(other,docid)
    assert ((await documents.rename_document(uid,docid,"new.txt")))["name"] == "new.txt"
    assert ((await documents.replace_document(uid,docid,"new.txt",b"updated")))["name"] == "new.txt"
    await documents.delete_document(uid,docid)
    assert (await documents.list_documents(uid)) == []
    assert not list((tmp_path/"documents").rglob("*.txt"))


# 注册、登录、刷新、登出经过真实异步仓储，同时检查审计落库。
async def test_auth_flow_with_repository(sqlite_database, monkeypatch):
    from auth import jwt_auth
    from fastapi.security import HTTPAuthorizationCredentials
    from fastapi import HTTPException
    from modules.models import AuthAuditEvent
    monkeypatch.setenv("JWT_SECRET_KEY", "async-test-secret-with-at-least-48-random-characters")
    monkeypatch.setenv("AUTH_REGISTRATION_CODE", "async-registration-code")
    monkeypatch.setattr(jwt_auth,"auth_repository",AuthRepository())
    user = await jwt_auth.register_user("alice", "test-password", "async-registration-code", "local")
    pair = await jwt_auth.login_user("alice","test-password","local")
    assert user["username"] == "alice"
    credentials = HTTPAuthorizationCredentials(scheme="Bearer",credentials=pair["access_token"])
    assert (await jwt_auth.get_current_user(credentials)) == "alice"
    renewed = await jwt_auth.refresh_token_pair(pair["refresh_token"],"local")
    with pytest.raises(HTTPException):
        await jwt_auth.get_current_user(credentials)
    await jwt_auth.revoke_access_token(renewed["access_token"],"alice","local")
    with pytest.raises(HTTPException):
        await jwt_auth.refresh_token_pair(renewed["refresh_token"],"local")
    async with session_scope() as db:
        events = ((await db.scalars(select(AuthAuditEvent)))).all()
        assert {event.event_type for event in events} >= {"register","login","refresh","logout"}


pytestmark = pytest.mark.anyio

# 测试统一使用项目的asyncio运行时。
@pytest.fixture
def anyio_backend():
    return "asyncio"
