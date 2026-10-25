"""路由包。

别名故意不叫 `chat_router`：那正好和子模块同名，导入时会被子模块对象覆盖，
`import api.routers.chat_router as x` 就可能拿到模块而不是路由对象。
"""

from api.routers.auth_router import router as auth_routes
from api.routers.chat_router import router as chat_routes
from api.routers.session_router import router as session_routes

__all__ = ["auth_routes", "chat_routes", "session_routes"]
