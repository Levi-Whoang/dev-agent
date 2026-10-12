from fastapi import FastAPI, HTTPException,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from starlette.exceptions import HTTPException as StarletteHTTPException

from api.api_schemas import ChatResponse, ChatRequest, CreateSessionRequest,APIResponse
from api.api_response import error_response
from api.api_response import success_response


from main import agent
from uuid import uuid4

import time
from logging_config import logger


app = FastAPI(title="Dev Agent API")

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


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = agent.run(
        request.message,
        session_id=request.session_id
    )

    return ChatResponse(
        answer=answer,
        session_id=request.session_id
    )



@app.post("/sessions/{session_id}/clear")
def clear_session(session_id: str):
    if not session_id.strip():
        raise HTTPException(
            status_code=400,
            detail="session_id 不能为空"
        )

    agent.session_manager.clear(session_id)

    return {
        "message": "会话已清空",
        "session_id": session_id
    }



@app.post("/sessions")
def create_session(request: CreateSessionRequest):
    session_id = request.session_id or str(uuid4())

    try:
        agent.session_manager.create(session_id)
    except Exception as e:
        raise HTTPException(
            status_code=409,
            detail="会话创建失败，可能是 session_id 已存在"
        ) from e

    return {
        "session_id": session_id,
        "message": "会话创建成功"
    }


@app.get("/sessions")
def list_sessions():
    sessions = agent.session_manager.list_sessions()
    return success_response(data=sessions)



@app.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    if not agent.session_manager.exists(session_id):
        raise HTTPException(
            status_code=404,
            detail="会话不存在"
        )

    messages = agent.session_manager.get_messages(session_id)

    return {
        "session_id": session_id,
        "messages": messages
    }

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if not agent.session_manager.exists(session_id):
        raise HTTPException(
            status_code=404,
            detail="会话不存在"
        )

    agent.session_manager.clear(session_id)

    return {
        "message": "会话删除成功",
        "session_id": session_id
    }



@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.status_code,
            "message": str(exc.detail),
            "data": None,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "message": "请求参数校验失败",
            "data": exc.errors(),
        },
    )

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