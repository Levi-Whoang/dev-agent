"""单实例启动校验。

生成停止、同会话互斥等能力目前保存在进程内，多 worker 部署会让这些能力
只对“处理该请求的进程”生效，因此启动阶段直接拒绝多 worker 配置。
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence

_WORKER_ENV_NAMES = ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS")


def _parse_workers(value: str) -> int | None:
    try:
        workers = int(value)
    except (TypeError, ValueError):
        return None
    return workers if workers > 0 else None


def _argv_workers(argv: Sequence[str]) -> int | None:
    """从命令行参数中读取 --workers/-w，支持空格和等号两种写法。"""
    for index, argument in enumerate(argv):
        for flag in ("--workers", "-w"):
            if argument == flag and index + 1 < len(argv):
                workers = _parse_workers(argv[index + 1])
                if workers is not None:
                    return workers
            elif argument.startswith(f"{flag}="):
                workers = _parse_workers(argument.split("=", 1)[1])
                if workers is not None:
                    return workers
            elif argument.startswith(flag) and argument != flag:
                workers = _parse_workers(argument[len(flag):])
                if workers is not None:
                    return workers
    return None


def ensure_single_process(
    environ: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    """发现多 worker 配置时拒绝启动。"""
    environ = os.environ if environ is None else environ
    argv = sys.argv if argv is None else argv

    workers = _argv_workers(argv)
    for name in _WORKER_ENV_NAMES:
        value = environ.get(name)
        if value is None:
            continue
        env_workers = _parse_workers(value)
        if env_workers is not None:
            workers = max(workers or 0, env_workers)

    if workers is not None and workers > 1:
        raise RuntimeError(
            "当前版本仅支持单进程运行：停止生成与会话互斥依赖进程内状态。"
            f"检测到 {workers} 个 worker，请改为 1 个进程，或先接入 "
            "Redis/粘性路由后再横向扩容。"
        )
