"""向量编码器：把 Sentence Transformer 包成异步接口。

为什么必须包一层：
`model.encode()` 是纯 CPU 的同步调用，一次可能要几百毫秒。
在 `async def` 路由里直接调用它，整个事件循环会被卡住——这期间所有请求都排队。
这里用 `asyncio.to_thread` 把它丢到线程池，事件循环继续处理别的连接。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger("dev-agent.rag")


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = asyncio.Lock()

    def _load_model(self):
        # 延迟到第一次真正需要时才加载：只跑 /health 不该下载几百 MB 模型
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("正在加载向量模型：%s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def _ensure_model(self):
        async with self._lock:
            if self._model is None:
                await asyncio.to_thread(self._load_model)
        return self._model

    async def encode(self, texts: Sequence[str]) -> np.ndarray:
        """返回 L2 归一化后的向量，之后点积就等于余弦相似度。"""
        model = await self._ensure_model()
        vectors = await asyncio.to_thread(
            model.encode,
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    async def encode_one(self, text: str) -> np.ndarray:
        vectors = await self.encode([text])
        return vectors[0]
