import typing as Any

def success_response(data=None, message="success"):
    return {
        "code": 0,
        "message": message,
        "data": data,
    }

def error_response(
    code: int,
    message: str,
    data: Any = None,
) -> dict:
    return {
        "code": code,
        "message": message,
        "data": data,
    }