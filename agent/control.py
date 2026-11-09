"""生成任务与工具执行控制。"""
import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextvars import copy_context
from threading import Event, Lock, BoundedSemaphore
import logging
import time
logger = logging.getLogger(__name__)
_tasks = {}
_lock = Lock()
_pool = ThreadPoolExecutor(max_workers=4)
_slots = BoundedSemaphore(4)
# 登记生成任务，避免同会话并发生成。
def begin_generation(key):
    with _lock:
        if key in _tasks:
            raise ValueError("会话正在生成")
        event = Event()
        _tasks[key] = event
        return event
# 请求停止指定会话任务。
def stop_generation(key):
    with _lock:
        event = _tasks.get(key)
        if event is not None:
            event.set()
        return event is not None
# 清除对应的任务登记。
def finish_generation(key, event):
    with _lock:
        if _tasks.get(key) is event:
            del _tasks[key]
# 有界线程执行工具，超时或取消后停止等待，保留用户上下文。
async def execute_tool(tool, arguments, timeout, stop_event=None):
    if inspect.iscoroutinefunction(tool) or inspect.iscoroutinefunction(getattr(tool, "__call__", None)):
        started = time.monotonic()
        logger.info("tool_start tool=%s", tool.__name__)
        try:
            async with asyncio.timeout(timeout):
                result = await await_or_stop(tool(**arguments), stop_event)
            logger.info("tool_complete tool=%s duration=%.3f", tool.__name__, time.monotonic()-started)
            return result
        except GenerationStopped:
            return {"error": "生成已停止", "code": "cancelled"}
        except TimeoutError:
            logger.warning("tool_timeout tool=%s duration=%.3f", tool.__name__, time.monotonic()-started)
            return {"error": "工具超时", "code": "tool_timeout"}
        except Exception:
            logger.warning("tool_failed tool=%s duration=%.3f", tool.__name__, time.monotonic()-started)
            return {"error": "工具执行失败", "code": "tool_failed"}
    if not _slots.acquire(blocking=False):
        return {"error": "工具繁忙", "code": "tool_busy"}
    started = time.monotonic()
    logger.info("tool_start tool=%s", tool.__name__)
    future = _pool.submit(copy_context().run, tool, **arguments)
    future.add_done_callback(lambda _: _slots.release())
    wrapped = asyncio.wrap_future(future)
    # 超时返回后消费后台异常，避免未处理异常告警。
    wrapped.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                return {"error": "生成已停止", "code": "cancelled"}
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                logger.warning("tool_timeout tool=%s duration=%.3f", tool.__name__, time.monotonic()-started)
                return {"error": "工具超时", "code": "tool_timeout"}
            try:
                done, _ = await asyncio.wait({wrapped}, timeout=min(0.05, remaining))
                if not done:
                    continue
                result = wrapped.result()
                logger.info("tool_complete tool=%s duration=%.3f", tool.__name__, time.monotonic()-started)
                return result
            except TimeoutError:
                if future.done():
                    raise
    except Exception:
        logger.warning("tool_failed tool=%s duration=%.3f", tool.__name__, time.monotonic()-started)
        return {"error": "工具执行失败", "code": "tool_failed"}


class GenerationStopped(Exception):
    """用户主动停止生成。"""


# 停止信号到达时取消正在等待的异步网络操作。
async def await_or_stop(operation, event):
    task = asyncio.ensure_future(operation)
    try:
        while not task.done():
            if event is not None and event.is_set():
                raise GenerationStopped()
            await asyncio.wait({task}, timeout=0.05)
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
