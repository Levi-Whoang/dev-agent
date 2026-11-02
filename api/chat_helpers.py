"""封装聊天引用提取和 SSE 事件生成。"""

import json


# 从最近一轮会话工具结果中提取引用来源。
def latest_turn_citations(session_manager, session_id: str) -> list[dict]:
    """从最近一轮知识检索工具结果提取可展示引用。"""
    messages = session_manager.get_messages(session_id)
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


# 将 Agent 生成过程转换为 SSE 流式事件。
def chat_event_stream(
    agent,
    logger,
    message: str,
    session_id: str,
    internal_session_id: str,
    owner_user_id: int | None = None,
):
    """将 Agent 输出转换为 SSE 增量、完成或错误事件。"""
    try:
        from tools.knowledge import knowledge_owner_scope

        iterator = agent.run_stream(message, session_id=internal_session_id)
        exhausted = object()
        try:
            while True:
                # 每次 next 都在同一次线程池调用中设置和重置上下文。
                with knowledge_owner_scope(owner_user_id):
                    content = next(iterator, exhausted)
                if content is exhausted:
                    break
                yield "data: " + json.dumps(
                    {"type": "delta", "content": content},
                    ensure_ascii=False,
                ) + "\n\n"
        finally:
            with knowledge_owner_scope(owner_user_id):
                iterator.close()

        yield "data: " + json.dumps(
            {
                "type": "done",
                "session_id": session_id,
                "citations": latest_turn_citations(
                    agent.session_manager, internal_session_id
                ),
            },
            ensure_ascii=False,
        ) + "\n\n"
    except Exception:
        logger.exception("流式聊天生成失败 session_id=%s", session_id)
        yield "data: " + json.dumps(
            {"type": "error", "message": "生成回复时发生错误"},
            ensure_ascii=False,
        ) + "\n\n"
