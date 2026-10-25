"""日志配置：控制台 + 文件双输出。

日志目录改成相对项目根（`BASE_DIR/logs`），
这样不管从哪个目录启动 `uvicorn app:app`，日志都落在同一个地方,
不会出现 “跑一次多一个 logs/ 文件夹”。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
    ],
)

# 第三方库的 INFO 太多，单独压一压
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logger = logging.getLogger("dev-agent")
