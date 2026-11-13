"""单实例启动校验测试。"""

import pytest

from app.process_guard import ensure_single_process


def test_allows_single_process_defaults():
    ensure_single_process({}, [])


def test_rejects_multiple_workers_from_env():
    with pytest.raises(RuntimeError, match="仅支持单进程"):
        ensure_single_process({"WEB_CONCURRENCY": "2"}, [])
    with pytest.raises(RuntimeError, match="仅支持单进程"):
        ensure_single_process({"UVICORN_WORKERS": "4"}, [])
    with pytest.raises(RuntimeError, match="仅支持单进程"):
        ensure_single_process({"GUNICORN_WORKERS": "3"}, [])


def test_rejects_multiple_workers_from_argv():
    with pytest.raises(RuntimeError, match="仅支持单进程"):
        ensure_single_process({}, ["uvicorn", "app.main:app", "--workers", "2"])
    with pytest.raises(RuntimeError, match="仅支持单进程"):
        ensure_single_process({}, ["uvicorn", "app.main:app", "--workers=2"])
    with pytest.raises(RuntimeError, match="仅支持单进程"):
        ensure_single_process({}, ["gunicorn", "-w", "4", "app.main:app"])


def test_allows_explicit_single_worker_and_ignores_invalid_values():
    ensure_single_process({"WEB_CONCURRENCY": "1"}, ["uvicorn", "--workers", "1"])
    # 非法值不能被误判成多进程。
    ensure_single_process({"WEB_CONCURRENCY": "abc"}, ["uvicorn", "--workers", "many"])
