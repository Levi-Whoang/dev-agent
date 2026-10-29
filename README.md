# Dev Agent — 企业知识库问答助手

一个基于 Python、FastAPI 和大语言模型的 Agent 应用项目，支持工具调用、企业知识库检索、对话历史持久化与 RESTful API。

项目重点是实践 Agent 应用开发中的核心流程：**LLM 决策 → Tool Calling → 工具执行 → 结果反馈 → 最终回答**，并结合 RAG 实现基于知识库证据的问答。

## 功能特性

- **Agent 工具调用**：根据用户问题决定是否调用工具，并处理工具参数、执行结果和异常。
- **RAG 混合检索**：使用 Sentence Transformers 做向量语义召回，并使用 BM25 做关键词召回；英文按词匹配，中文按重叠字片段匹配，再融合排序。
- **可解释检索分数**：返回知识片段、来源文件、向量分、关键词分和综合分，便于检查召回原因。
- **多轮对话**：维护会话消息历史，并将会话记录保存到 MySQL。
- **RESTful API**：基于 FastAPI 提供聊天、会话创建、历史查询、清空和删除接口。
- **统一异常响应**：处理 HTTP 异常、请求参数校验异常及未预期异常。
- **请求日志**：记录 API 请求路径、状态码与处理耗时。

## 技术栈

| 类别           | 技术                                               |
|----------------|----------------------------------------------------|
| 编程语言       | Python                                             |
| Web 框架       | FastAPI                                            |
| 大语言模型调用 | OpenAI 兼容 API                                    |
| Agent 核心     | Tool Calling、工具注册与参数校验                   |
| RAG            | Sentence Transformers、余弦相似度、BM25 关键词检索 |
| 参数校验       | Pydantic                                           |
| 数据库         | MySQL、SQLAlchemy ORM、PyMySQL 驱动                |
| 测试           | pytest                                             |

## 项目结构

```text
dev-agent/
├── agent/
│   ├── agent.py             # Agent 决策循环与工具调用
│   └── session.py           # MySQL 会话持久化
├── api/
│   ├── api_schemas.py       # 请求与响应模型
│   ├── api_response.py      # 统一响应格式
│   ├── chat_helpers.py      # 聊天接口辅助逻辑
│   └── session_helpers.py   # 会话接口辅助逻辑
├── app/
│   ├── main.py              # FastAPI 应用入口与路由注册
│   ├── users.py             # 注册、登录与认证相关路由
│   ├── sessions.py          # 聊天和会话管理路由
│   └── errors.py            # 全局异常处理
├── auth/
│   ├── jwt_auth.py          # JWT 认证与令牌验证
│   ├── passwords.py         # 密码哈希与校验
│   ├── repository.py        # 用户和令牌数据访问层
│   └── schemas.py           # 认证请求模型
├── modules/
│   ├── database.py          # SQLAlchemy engine、Session 与数据库初始化
│   └── models.py            # 用户、JWT 登记和聊天会话 ORM 模型
├── tools/
│   ├── registry.py          # 工具注册
│   ├── schema.py            # 工具参数模型与 Schema 生成
│   ├── knowledge.py         # 知识加载、向量/关键词混合检索与证据构建
│   ├── text_splitter.py     # 文本分块
│   ├── calculator.py        # 计算工具
│   ├── weather.py           # 天气工具
│   └── order.py             # 订单工具
├── frontend/                # React + TypeScript 前端
├── scripts/                 # 演示链路检查等脚本
├── logs/
│   └── logging_config.py     # 日志配置
├── test/                    # API、Agent、会话、检索和切分测试
├── demo_data/knowledge_base/ # 随仓库提供的虚构演示语料
├── app.py                   # 兼容入口；推荐使用 app.main:app
├── main.py                  # Agent 初始化入口
├── .mcp.json                # 本地 MCP 客户端配置示例
├── requirements.txt         # Python 依赖
└── README.md
```

## 环境要求

- Python 3.12
- MySQL
- 可用的大语言模型 API 及对应 API Key

## 安装与配置

### 1. 克隆项目

```bash
git clone https://github.com/Levi-Whoang/dev-agent.git
cd dev-agent
```

### 2. 创建虚拟环境

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，填写 OpenAI 兼容模型 API、MySQL 配置和随机 `JWT_SECRET_KEY`。账号通过前端注册或 `POST /auth/register` 创建，用户名和 PBKDF2 密码哈希保存在 MySQL 的 `auth_users` 表；JWT 会话登记保存在 `auth_tokens` 表，有效期固定为 30 分钟。首次注册/登录时应用会自动创建认证表。首次访问会话功能时，`sessions.owner_user_id` 会关联到 `auth_users.id`，并按旧会话的用户哈希前缀回填归属；未匹配到用户的旧会话保留为未归属记录。前端侧栏可查看并打开当前账号的历史会话。示例配置将知识库指向随仓库提供的 `demo_data/knowledge_base/` 虚构演示语料；换成自己的语料时，修改 `KNOWLEDGE_DIR`。

PowerShell：

```powershell
Copy-Item .env.example .env
notepad .env
```

生成随机 JWT 签名密钥并填入 `.env`：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

将命令输出填入 `JWT_SECRET_KEY`。用户名和密码由用户注册接口提交，后端只把密码哈希写入 MySQL。`.env.example` 中密钥是占位值，必须替换。不要把 `.env` 或真实签名密钥提交到仓库。

确保 MySQL 服务已启动，且配置的数据库已经创建。账号需要在目标库中创建表和读写数据的权限；应用首次使用时通过 SQLAlchemy ORM 自动创建缺失的 `sessions`、`auth_users` 和 `auth_tokens` 表。`create_all` 不会修改已有表结构；如果数据库来自旧版本且结构不兼容，请配置一个空的新数据库，或自行迁移后再启动。连接默认读取 `MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_USER`、`MYSQL_PASSWORD` 和 `MYSQL_DATABASE`；也可设置 SQLAlchemy 格式的 `DATABASE_URL` 覆盖默认连接配置。

### 5. 演示语料与模型

克隆仓库即可使用 `demo_data/knowledge_base/` 中的虚构演示资料。检索模块读取 `KNOWLEDGE_DIR` 下的 `.txt` 文件。Sentence Transformers 模型首次检索时需要从 Hugging Face 下载；后续运行使用本机缓存。

演示语料中的公司、职位和地址均为虚构数据。自己的知识库可以通过 `.env` 中的 `KNOWLEDGE_DIR` 指向本地目录；请勿将公司内部资料、个人隐私或其他敏感信息提交到公开仓库。

`.mcp.json` 是本地开发工具的 MCP 客户端配置示例，当前项目的 Agent 尚未接入 MCP Server；没有配置或启动对应的本地 MCP 服务时，该文件不会影响 API、聊天或 RAG 功能。

### 6. 运行离线测试

```powershell
python -m pytest test/ -v
```

HTTP/ASGI 测试使用内存 Agent 和会话存储，不读取 API Key、不连接 MySQL，也不调用真实模型。

## 启动服务

在项目根目录运行：

```bash
uvicorn app.main:app --reload
```

服务启动后访问：

- API 文档：`http://127.0.0.1:8000/docs`
- OpenAPI Schema：`http://127.0.0.1:8000/openapi.json`
- 存活检查：`http://127.0.0.1:8000/health`

### 实际服务链路检查

配置 `.env`、启动 MySQL 和 API 服务后，在另一个终端运行：

```powershell
python scripts/smoke_demo.py
```

该检查会先提示输入已注册账号，然后通过真实 HTTP 流程调用 LLM 和检索工具、确认 MySQL 中写入了工具结果及完整回答，最后删除专用临时会话。它会产生一次模型 API 调用，并在 MySQL 中短暂创建一条演示会话；检查脚本不会打印密码、令牌或完整回答。

## React + TypeScript 前端

前端代码位于 `frontend/`，支持注册、JWT 登录、SSE 流式聊天、引用来源展示和新建会话。注册数据写入后端 MySQL。启动后端 API，再在项目根目录运行：

```powershell
cd frontend
npm install
npm run dev
```

默认通过 Vite 代理将 `/api` 请求转发至 `http://127.0.0.1:8000`。生产构建运行 `npm run build`；部署时需让同源 `/api` 转发到 FastAPI，或配置 `VITE_API_BASE_URL` 与后端 CORS。更多前端说明见 [frontend/README.md](frontend/README.md)。

## API 示例

### 1. 发送聊天请求

`POST /chat`

请求体：

```json
{
  "message": "请介绍一下知识库中的相关岗位要求",
  "session_id": "demo-session"
}
```

`POST /chat` 始终返回 `Content-Type: text/event-stream` 的 SSE 流：`delta` 事件逐段包含 `content`，结束时 `done` 事件包含 `session_id` 和本轮检索得到的 `citations`；发生生成错误时发送 `error` 事件。请求方应持续读取响应流并拼接 `delta.content`。每条引用包含来源文件、证据片段和检索分数。没有调用知识检索工具时引用列表为空。实际回答取决于模型配置和知识库内容。

注册新用户时调用 `POST /auth/register`：

```json
{"username": "demo-user", "password": "至少 8 位的密码"}
```

注册成功后调用 `POST /auth/token` 提交同一用户名和密码登录，响应中的 `data.access_token` 是 JWT。令牌有效期为 30 分钟，并在 MySQL 的 `auth_tokens` 表登记令牌 ID 和过期时间；退出登录 `POST /auth/logout` 会撤销当前令牌。除 `GET /health`、注册和登录接口外，所有 API 路由都需要 `Authorization: Bearer <access_token>`。密码以 PBKDF2 哈希保存在 `auth_users` 表。会话仍按认证用户名隔离。省略 `session_id` 会为该次请求生成新会话，后续请求应带回响应中的 ID 以继续对话。会话列表只返回当前认证用户的会话。

会话列表通过数据库中的 `owner_user_id` 关联当前用户。旧版本数据不会自动迁移或回填归属；若旧表结构不兼容，请按安装说明改用空数据库。

### 2. 创建会话

`POST /sessions`

```json
{
  "session_id": "demo-session"
}
```

也可以不提供 `session_id`，由服务生成会话 ID。

### 3. 查询会话列表

`GET /sessions`

### 4. 查询会话消息

`GET /sessions/demo-session/messages`

### 5. 修改会话名称

`PATCH /sessions/demo-session/title`，请求体示例：

```json
{"title": "招聘政策问题"}
```

### 6. 清空会话消息

`POST /sessions/demo-session/clear`

### 7. 删除会话

`DELETE /sessions/demo-session`

## Agent 工作流程

```text
用户请求
   ↓
FastAPI 接收并校验参数
   ↓
加载对应会话历史
   ↓
LLM 分析问题并决定是否调用工具
   ↓
工具参数校验
   ↓
执行工具（例如知识库检索）
   ↓
将工具结果反馈给 LLM
   ↓
生成最终回答
   ↓
保存会话历史并返回响应
```

当涉及知识库的问题时，Agent 可以调用知识检索工具获取相关文本，再结合检索结果生成回答。文本切分优先沿段落、句子和换行边界进行；遇到超长句或无标点文本时，再按空白或字符长度切分，并保留相邻片段重叠。检索同时使用向量语义分和 BM25 关键词分：默认权重分别为 0.7 和 0.3，默认向量相似度阈值为 0.60；通用词不参与关键词分，关键词召回还需达到有效查询词覆盖门槛。没有充分关键词证据时，仅由向量阈值决定是否返回。结果中的 `score` 是综合分，`dense_score` 和 `keyword_score` 可用于观察两路召回。

`test/retrieval_dataset.py` 当前是 `rag-eval-2.0`，包含 80 条按真实提问方式编写的样例：50 条可回答、10 条一般无答案、20 条与已有知识相近但缺少目标事实的干扰问题。正例标注标准证据词和允许来源。运行 `python test/evaluate_retrieval.py` 比较分块、阈值和 Top-K 配置，查看 Recall@K、MRR、误召回率和分类结果。每次运行会在 `test/evaluation_results/` 保存 JSON 报告，其中包含评估集版本与 SHA-256 指纹、配置、指标及失败问题；报告不复制检索片段正文，只记录来源、分数和正文摘要哈希。该评估集仍是开发集，指标不能代表通用检索准确率。

## 运行测试

安装依赖后，在项目根目录运行：

```bash
python -m pytest test/ -v
```

测试结果应以本地实际运行输出为准。单元测试通过不等于真实模型 API、MySQL 和知识库的端到端集成测试全部通过。

## 项目改进方向

- 将本地 MySQL 和模型依赖整理为可部署的服务配置，并完善部署文档。
- 持续扩大 RAG 评估集，补充问答正确性、引用准确性和无答案拒答质量评估。
- 为知识库变更、索引缓存和检索性能增加可重复的评估与观测。
- 根据目标运行环境补充数据库迁移、备份和生产级运维方案。

## 项目目标

通过实现一个可运行的 Agent 应用，系统实践大模型工具调用、RAG 检索、会话持久化、API 设计、异常处理和自动化测试等应用开发能力。

在历史会话列表中可修改会话名称或删除会话。删除操作会请求后端 DELETE /sessions/{session_id}；删除当前会话后，聊天面板会清空。
