"""pytest 公共夹具。

关键点：在**导入任何业务模块之前**把 DATABASE_URL 指到临时 SQLite，
否则测试会把数据写进开发库（dev_agent.db），跑一次脏一次。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 必须在 import config / app 之前设置
_TMP_DIR = tempfile.mkdtemp(prefix="dev-agent-test-")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TMP_DIR}/test.db"
os.environ["JWT_SECRET"] = "test-secret-not-for-production"
os.environ["LOG_LEVEL"] = "WARNING"


@pytest_asyncio.fixture
async def db_ready():
    """建好表，跑完清理连接池。"""
    from database import base

    await base.dispose_db()
    await base.init_db()
    yield
    await base.dispose_db()


@pytest_asyncio.fixture
async def client(db_ready):
    from httpx import ASGITransport, AsyncClient

    from app import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def auth_client(client):
    """已登录的客户端，带 Authorization 头。

    用户名带随机后缀：并发注册同名会因为唯一索引直接 409。
    """
    import uuid

    username = f"alice_{uuid.uuid4().hex[:8]}"
    res = await client.post(
        "/auth/register",
        json={"username": username, "password": "alice123456"},
    )
    assert res.status_code == 201, res.text
    token = res.json()["data"]["token"]["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    client.username = username
    return client


@pytest.fixture(scope="session")
def knowledge_dir() -> Path:
    from config import get_settings

    return get_settings().knowledge_dir
