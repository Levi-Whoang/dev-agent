from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from starlette.exceptions import HTTPException as StarletteHTTPException

from api.api_schemas import ChatResponse, ChatRequest, CreateSessionRequest,APIResponse
from api.api_response import error_response
from api.api_response import success_response
from auth.jwt_auth import authenticate_user, create_access_token, get_current_user
from auth.schemas import TokenData, TokenRequest


from main import agent
from uuid import uuid4
import hashlib
import json

import time
from logging_config import logger


app = FastAPI(title="Dev Agent API")
SESSION_ID_MAX_LENGTH = 75


def _session_prefix(user_id: str) -> str:
    """为认证用户生成不暴露身份的会话命名空间前缀。"""
    owner_key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
    return f"{owner_key}:"


def _stored_session_id(user_id: str, session_id: str) -> str:
    if not session_id or len(session_id) > SESSION_ID_MAX_LENGTH:
        raise HTTPException(status_code=422, detail="session_id 长度无效")
    return f"{_session_prefix(user_id)}{session_id}"


def _resolve_chat_session(user_id: str, requested_session_id: str | None):
    session_id = requested_session_id or uuid4().hex
    stored_session_id = _stored_session_id(user_id, session_id)
    if not agent.session_manager.exists(stored_session_id):
        agent.session_manager.create(stored_session_id)
    return session_id, stored_session_id


def _latest_turn_citations(session_id: str) -> list[dict]:
    """从最近一轮知识检索工具结果提取可展示引用。"""
    messages = agent.session_manager.get_messages(session_id)
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


@app.get("/health")
def health():
    """进程存活检查，不连接外部模型或数据库。"""
    return {"status": "ok"}


@app.post("/auth/token", response_model=APIResponse)
def issue_token(request: TokenRequest):
    """校验账号密码并签发短时效 JWT。"""
    if not authenticate_user(request.username, request.password):
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token, expires_in = create_access_token(request.username)
    return success_response(
        data=TokenData(access_token=access_token, expires_in=expires_in).model_dump()
    )

#日志（中间件）
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - start_time

        logger.info(
            "%s %s status=%s duration=%.3fs",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )

        return response

    except Exception:
        elapsed = time.perf_counter() - start_time

        logger.exception(
            "%s %s failed duration=%.3fs",
            request.method,
            request.url.path,
            elapsed,
        )
        raise

#在某会话进行
def chat_event_stream(message: str, session_id: str, stored_session_id: str):
    try:
        for content in agent.run_stream(message, session_id=stored_session_id):
            yield "data: " + json.dumps(
                {"type": "delta", "content": content},
                ensure_ascii=False,
            ) + "\n\n"

        yield "data: " + json.dumps(
            {
                "type": "done",
                "session_id": session_id,
                "citations": _latest_turn_citations(stored_session_id),
            },
            ensure_ascii=False,
        ) + "\n\n"
    except Exception:
        logger.exception("流式聊天生成失败 session_id=%s", session_id)
        yield "data: " + json.dumps(
            {"type": "error", "message": "生成回复时发生错误"},
            ensure_ascii=False,
        ) + "\n\n"


@app.post("/chat")
def chat(request: ChatRequest, current_user: str = Depends(get_current_user)):
    session_id, stored_session_id = _resolve_chat_session(
        current_user,
        request.session_id,
    )
    if not request.stream:
        answer = agent.run(
            request.message,
            session_id=stored_session_id,
        )

        return success_response(
            data={
                "answer": answer,
                "session_id": session_id,
                "citations": _latest_turn_citations(stored_session_id),
            }
        )

    return StreamingResponse(
        chat_event_stream(request.message, session_id, stored_session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

#清空会话
@app.post("/sessions/{session_id}/clear", response_model=APIResponse)
def clear_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    if not session_id.strip():
        raise HTTPException(
            status_code=400,
            detail="session_id 不能为空"
        )

    stored_session_id = _stored_session_id(current_user, session_id)
    if not agent.session_manager.exists(stored_session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    agent.session_manager.clear(stored_session_id)

    return success_response(
        data={"session_id": session_id},
        message="会话已清空"
    )

#新建会话
@app.post("/sessions", response_model=APIResponse)
def create_session(
    request: CreateSessionRequest,
    current_user: str = Depends(get_current_user),
):
    session_id = request.session_id or str(uuid4())
    stored_session_id = _stored_session_id(current_user, session_id)

    try:
        agent.session_manager.create(stored_session_id)
    except Exception as e:
        raise HTTPException(
            status_code=409,
            detail="会话创建失败，可能是 session_id 已存在"
        ) from e

    return success_response(
        data={"session_id": session_id},
        message="会话创建成功"
    )

#列出所有会话id
@app.get("/sessions", response_model=APIResponse)
def list_sessions(current_user: str = Depends(get_current_user)):
    prefix = _session_prefix(current_user)
    sessions = [
        {
            **session,
            "session_id": session["session_id"][len(prefix):],
        }
        for session in agent.session_manager.list_sessions(prefix=prefix)
        if session["session_id"].startswith(prefix)
    ]
    return success_response(data=sessions)

#获取会话信息
@app.get("/sessions/{session_id}/messages", response_model=APIResponse)
def get_session_messages(
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    stored_session_id = _stored_session_id(current_user, session_id)
    if not agent.session_manager.exists(stored_session_id):
        raise HTTPException(
            status_code=404,
            detail="会话不存在"
        )

    messages = agent.session_manager.get_messages(stored_session_id)

    return success_response(
        data={
            "session_id": session_id,
            "messages": messages
        }
    )

#删除会话
@app.delete("/sessions/{session_id}", response_model=APIResponse)
def delete_session(
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    stored_session_id = _stored_session_id(current_user, session_id)
    if not agent.session_manager.exists(stored_session_id):
        raise HTTPException(
            status_code=404,
            detail="会话不存在"
        )

    agent.session_manager.delete(stored_session_id)

    return success_response(
        data={"session_id": session_id},
        message="会话删除成功"
    )


#三个异常响应
#http异常处理器
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            code=exc.status_code,
            message=str(exc.detail)
        ),
        headers=exc.headers,
    )

#参数校验异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content=error_response(
            code=422,
            message="请求参数校验失败",
            data=exc.errors()
        ),
    )


#处理其他未知异常
@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
):
    return JSONResponse(
        status_code=500,
        content=error_response(
            code=500,
            message="服务器内部错误",
        ),
    )








if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
