"""整体超时预算、剩余时间计算与可重试错误的有限退避。"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from openai import APIConnectionError, APITimeoutError, RateLimitError

logger = logging.getLogger(__name__)

# 这些状态码通常代表瞬时故障，可以安全地重试。
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})

T = TypeVar("T")


class AgentTimeoutError(RuntimeError):
    """本次回答耗尽了整体时间预算。"""

    code = "agent_timeout"

    def __init__(self, message: str = "本次回答超时，请稍后重试或缩短问题"):
        super().__init__(message)


@dataclass(frozen=True)
class TimeoutConfig:
    """一轮回答的总预算、单次请求上限与重试策略。"""

    total_seconds: float = 120.0
    request_seconds: float = 20.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    max_retry_backoff_seconds: float = 4.0

    @classmethod
    def from_env(cls) -> "TimeoutConfig":
        return cls(
            total_seconds=_float_env("CHAT_TOTAL_TIMEOUT_SECONDS", 120.0, 0.01),
            request_seconds=_float_env("LLM_REQUEST_TIMEOUT_SECONDS", 20.0, 0.01),
            max_retries=_int_env("LLM_MAX_RETRIES", 2, 0),
            retry_backoff_seconds=_float_env(
                "LLM_RETRY_BACKOFF_SECONDS", 0.5, 0.0
            ),
            max_retry_backoff_seconds=_float_env(
                "LLM_MAX_RETRY_BACKOFF_SECONDS", 4.0, 0.0
            ),
        )


@dataclass(frozen=True)
class Deadline:
    """以 monotonic 时钟记录的绝对截止时间。"""

    expires_at: float

    @classmethod
    def after(cls, seconds: float) -> "Deadline":
        return cls(time.monotonic() + max(0.0, float(seconds)))

    def remaining(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())

    def ensure(self, minimum: float = 0.05) -> float:
        """返回剩余秒数；预算已耗尽时抛出统一超时错误。"""
        remaining = self.remaining()
        if remaining <= minimum:
            raise AgentTimeoutError()
        return remaining


def is_retryable(error: Exception) -> bool:
    """只重试连接、限流、超时和常见瞬时服务端错误。"""
    if isinstance(error, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    if isinstance(error, TimeoutError):
        return True
    status_code = getattr(error, "status_code", None)
    return isinstance(status_code, int) and status_code in RETRYABLE_STATUS_CODES


async def run_with_retry(
    operation: Callable[[], Awaitable[T]],
    deadline: Deadline,
    config: TimeoutConfig,
    *,
    stop_event=None,
) -> T:
    """在整体预算内执行操作，可重试错误按指数退避有限重试。"""
    attempt = 0
    while True:
        deadline.ensure()
        try:
            return await operation()
        except AgentTimeoutError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not is_retryable(error) or attempt >= config.max_retries:
                raise
            attempt += 1
            remaining = deadline.remaining()
            if remaining <= 0.05:
                raise AgentTimeoutError() from error
            delay = min(
                config.retry_backoff_seconds * (2 ** (attempt - 1)),
                config.max_retry_backoff_seconds,
                max(0.0, remaining - 0.05),
            )
            logger.warning(
                "llm_retry attempt=%s delay=%.3f error=%s",
                attempt,
                delay,
                error,
            )
            if delay > 0:
                if stop_event is not None:
                    # 退避期间仍需及时响应用户停止。
                    from agent.control import await_or_stop

                    await await_or_stop(asyncio.sleep(delay), stop_event)
                else:
                    await asyncio.sleep(delay)


def _float_env(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("timeout_config_invalid name=%s value=%s", name, raw)
        return default
    if value < minimum:
        logger.warning("timeout_config_too_small name=%s value=%s", name, raw)
        return minimum
    return value


def _int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("timeout_config_invalid name=%s value=%s", name, raw)
        return default
    if value < minimum:
        logger.warning("timeout_config_too_small name=%s value=%s", name, raw)
        return minimum
    return value
