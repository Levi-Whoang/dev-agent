"""封装聊天引用提取和 SSE 事件生成。"""

import json


# 从最近一轮会话工具结果中提取引用来源。
async def latest_turn_citations(session_manager, session_id: str) -> list[dict]:
    """从最近一轮知识检索工具结果提取可展示引用。"""
    messages = await session_manager.get_messages(session_id)
    latest_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        -1,
    )
    citations = []
    seen = set()

    for message in messages[latest_user_index + 1:]:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            result = json.loads(content)
            # 兼容修复前已经保存的双重编码工具消息。
            if isinstance(result, str):
                result = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict):
            continue
        for citation in result.get("citations", []):
            if not isinstance(citation, dict) or not citation.get("source"):
                continue
            key = (citation["source"], citation.get("content", ""))
            if key not in seen:
                seen.add(key)
                citations.append(citation)
    return citations


# 异步消费Agent正文，保证检索上下文与生成器在同一任务中关闭。
async def chat_event_stream(agent, logger, message, session_id, internal_session_id, owner_user_id=None, stop_event=None):
    import anyio
    import inspect
    from tools.knowledge import knowledge_owner_scope
    from agent.control import GenerationStopped
    kwargs = {"session_id": internal_session_id}
    if "stop_event" in inspect.signature(agent.run_stream).parameters:
        kwargs["stop_event"] = stop_event
    iterator = agent.run_stream(message, **kwargs)
    try:
        with knowledge_owner_scope(owner_user_id):
            try:
                async for content in iterator:
                    yield "data: " + json.dumps({"type": "delta", "content": content}, ensure_ascii=False) + "\n\n"
            except GenerationStopped:
                pass
            finally:
                with anyio.CancelScope(shield=True):
                    await iterator.aclose()
        citations = await latest_turn_citations(agent.session_manager, internal_session_id)
        yield "data: " + json.dumps({"type": "done", "session_id": session_id, "citations": citations,
                **({"stopped": True} if stop_event and stop_event.is_set() else {})}, ensure_ascii=False) + "\n\n"
    except Exception:
        logger.exception("流式聊天生成失败 session_id=%s", session_id)
        yield "data: " + json.dumps({"type": "error", "message": "生成回复时发生错误"}, ensure_ascii=False) + "\n\n"


# 管理异步SSE生命周期，断连时清理模型流并释放任务登记。
async def managed_chat_stream(agent, logger, message, session_id, internal_id, owner_id, event):
    import anyio
    import time
    from agent.control import finish_generation
    started = time.monotonic()
    logger.info("generation_start session_id=%s", session_id)
    iterator = chat_event_stream(agent, logger, message, session_id, internal_id, owner_id, event)
    try:
        async for item in iterator:
            yield item
    finally:
        stopped = event.is_set()
        event.set()
        try:
            with anyio.CancelScope(shield=True):
                await iterator.aclose()
        finally:
            finish_generation(internal_id, event)
            logger.info("generation_end session_id=%s stopped=%s duration=%.3f", session_id, stopped, time.monotonic()-started)
