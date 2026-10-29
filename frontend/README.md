# Dev Agent 前端

这是一个独立的 React + TypeScript 聊天前端，使用现有 FastAPI 认证和聊天 API。它支持带注册邀请码的账号注册、JWT 登录/退出、自动刷新令牌、SSE 流式回复、引用来源展示、新建会话、历史会话恢复和修改会话名称。

## 启动

确保 MySQL 和后端 API 已启动，并在后端 `.env` 配置 `JWT_SECRET_KEY`。新用户可直接在前端注册；账号密码会由后端写入 MySQL。然后在本目录运行：

```powershell
npm install
npm run dev
```

打开终端显示的本地网址。默认 Vite 代理会把 `/api` 请求转发至 `http://127.0.0.1:8000`，因此开发时无需额外配置后端 CORS。

如后端运行在其他地址，可创建 `.env.local` 并设置：

```dotenv
VITE_API_TARGET=http://127.0.0.1:8000
```

JWT 只保存在当前标签页的 `sessionStorage`，关闭标签页后会清除。退出登录会调用后端撤销当前令牌。

## 构建

```powershell
npm run build
npm run preview
```

生产环境默认请求同源 `/api`。请在 Web 服务器或反向代理中将 `/api` 转发到 FastAPI；若前后端部署在不同域名，可在构建时设置 `VITE_API_BASE_URL` 为后端 URL，并在后端配置允许的前端来源。
