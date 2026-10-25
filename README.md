# Dev Agent — 企业知识库问答助手

一个基于 **FastAPI + 异步 SQLAlchemy + 混合检索 RAG + 流式输出** 的 Agent 应用。

核心链路：**LLM 决策 → Tool Calling → 工具执行 → 结果回灌 → 流式回答**，
其中知识类问题由「向量召回 + BM25 召回 + RRF 融合」的混合检索提供证据。

## 功能特性

| 能力 | 说明 |
|---|---|
| 用户体系 | 注册 / 登录 / JWT 鉴权，会话按用户隔离 |
| Agent 工具调用 | 自动决定是否调工具，参数经 Pydantic 校验后执行 |
| 混合检索 RAG | 稠密向量（bge-small-zh）+ 稀疏 BM25 + RRF 融合排序 |
| 拒答机制 | 证据不足返回 `NO_RELEVANT_EVIDENCE`，禁止模型猜测 |
| 流式输出 | SSE 逐字返回，工具调用过程也作为事件推送 |
| 全异步 | 异步路由 + 异步数据库 + 异步 LLM 客户端 + 线程池跑向量编码 |
| 前端页面 | 原生 HTML/CSS/JS，单页实现登录、会话列表、打字机效果 |
| 统一响应 | `{code, message, data}`，异常处理器统一兜底 |

## 项目结构

```text
dev-agent/
├── app.py                  # FastAPI 装配：路由 / 中间件 / 异常 / 静态页
├── main.py                 # 命令行入口（不启服务，直接问一句）
├── config.py               # 统一读取环境变量（唯一配置源）
├── logging_config.py       # 日志配置
│
├── database/               # 数据层
│   ├── base.py             # 异步 engine / 会话工厂 / get_db 依赖
│   └── models.py           # User / ChatSession / ChatMessage
│
├── auth/                   # 鉴权
│   ├── passwords.py        # PBKDF2 密码哈希
│   ├── tokens.py           # JWT 签发与校验
│   ├── schemas.py          # 注册 / 登录模型
│   ├── service.py          # 用户业务逻辑
│   └── dependencies.py     # get_current_user 依赖
│
├── api/                    # HTTP 层
│   ├── api_schemas.py      # 请求 / 响应模型
│   ├── api_response.py     # 统一响应格式
│   ├── deps.py             # DbSession / CurrentUser 类型别名
│   ├── exceptions.py       # 异常处理器
│   ├── middleware.py       # 纯 ASGI 请求日志中间件
│   └── routers/
│       ├── auth_router.py      # /auth/*
│       ├── session_router.py   # /sessions/*
│       └── chat_router.py      # /chat, /chat/stream
│
├── agent/                  # Agent 核心
│   ├── agent.py            # 异步决策循环 + stream()
│   ├── events.py           # SSE 事件定义
│   ├── prompts.py          # 系统提示词
│   ├── store.py            # 会话 / 消息读写（SQLAlchemy）
│   ├── tools_runner.py     # 工具参数校验与执行
│   └── factory.py          # 按请求装配 Agent
│
├── rag/                    # 检索
│   ├── loader.py           # 文档加载 + 缓存
│   ├── splitter.py         # 三级切分：段落 → 句子 → 词/字
│   ├── embedder.py         # 向量编码（异步封装）
│   ├── bm25.py             # BM25 + 中英混合分词
│   ├── hybrid.py           # RRF 融合
│   └── retriever.py        # 检索入口（召回 → 融合 → 门槛）
│
├── tools/                  # 工具定义（全部 async）
│   ├── registry.py         # 注册表
│   ├── schema.py           # 函数签名 → Tool Schema
│   ├── knowledge.py        # 知识检索工具（薄封装 rag/）
│   ├── calculator.py / weather.py / order.py
│   └── text_splitter.py    # 兼容层，指向 rag/splitter.py
│
├── web/                    # 前端页面（由 FastAPI 直接托管）
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── test/                   # 测试（pytest + pytest-asyncio）
├── TXT/                    # 变更记录（对话总结 / 代码修改）
└── requirements.txt
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

cp .env.example .env              # 填入 LLM_API_KEY 等
python scripts/generate_knowladge_base.py   # 生成示例知识库
uvicorn app:app --reload
```

浏览器打开 <http://127.0.0.1:8000/>：

- 首页：登录 / 注册 + 聊天界面
- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/api/health>

### 数据库

默认使用项目根目录的 **SQLite**（`dev_agent.db`），不需要装任何数据库服务。
首次启动会自动建表。

要换成 MySQL：

```dotenv
DATABASE_URL=mysql+aiomysql://root:password@127.0.0.1:3306/agent_db?charset=utf8mb4
```

并安装异步驱动 `pip install aiomysql`。
（如果只配了 `MYSQL_HOST` 却没装 aiomysql，程序会打印警告并自动降级到 SQLite。）

### 环境变量

| 变量 | 说明 |
|---|---|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | OpenAI 兼容接口配置 |
| `DATABASE_URL` | 数据库连接串，留空则用 SQLite |
| `JWT_SECRET` | JWT 签名密钥，**部署前必须替换** |
| `JWT_EXPIRE_MINUTES` | token 有效期（默认 720 分钟） |
| `KNOWLEDGE_DIR` | 知识库目录，相对路径按项目根解析 |
| `RAG_DENSE_THRESHOLD` | 稠密检索“算作有证据”的门槛（默认 0.55） |
| `RAG_DENSE_FLOOR` | 只靠关键词命中时要求的最低语义分（默认 0.45） |
| `RAG_RRF_K` | RRF 融合参数（默认 60） |

完整清单见 `.env.example`。

## API

| 方法 | 路径 | 说明 | 需要登录 |
|---|---|---|---|
| POST | `/auth/register` | 注册并返回 token | 否 |
| POST | `/auth/login` | 登录换取 token | 否 |
| GET | `/auth/me` | 当前用户 | 是 |
| GET | `/sessions` | 我的会话列表 | 是 |
| POST | `/sessions` | 新建会话 | 是 |
| PATCH | `/sessions/{id}` | 重命名会话（标题 1~60 字，自动洗空白） | 是 |
| GET | `/sessions/{id}/messages` | 会话消息 | 是 |
| POST | `/sessions/{id}/clear` | 清空消息 | 是 |
| DELETE | `/sessions/{id}` | 删除会话 | 是 |
| POST | `/chat` | 提问，等完整回答 | 是 |
| POST | `/chat/stream` | 提问，SSE 流式回答 | 是 |

调用示例：

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123456"}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['data']['token']['access_token'])")

curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message":"公司在哪里？"}'
```

### 流式事件协议

```text
event: meta        {"session_id": "...", "model": "..."}
event: tool_start  {"name": "knowledge_search_context", "arguments": "{...}"}
event: tool_result {"name": "...", "result": "..."}
event: delta       {"text": "增量文本"}
event: done        {"answer": "完整回答", "session_id": "...", "steps": 2}
event: error       {"message": "出错原因"}
```

前端用 `fetch + ReadableStream` 解析，不用 `EventSource`
（后者只能发 GET，也带不了 `Authorization` 头）。

## 文本怎么切（三级，不是硬切）

```text
段落（空行）
  └─ 超过 300 字 → 句子（。！？；!?）
        └─ 单句还超长 → 分句（，、：）
              └─ 分句还超长 → 词/字（英文按单词、中文按单字）窗口兜底
```

三条规则：

- **短块吸附**：小于 60 字的段落（标题、单句）并入相邻段，避免"公司介绍"
  这种标题单独成片——它单独成块时向量里几乎没有信息。
- **整句重叠**：`overlap=50` 是预算，回退时按整句取，不会把一句话切成两半。
- **降级才到词**：只有句子切不开时才降到词/字；英文单词始终完整。

为什么不给中文做真正的分词：那需要几十 MB 的词典依赖，
而切点落在词中间时重叠窗口已经把上下文补回来了，收益远小于成本。

## 混合检索是怎么做的

```text
提问
 ├─ 稠密：bge-small-zh 编码 → 余弦相似度 → 取前 N
 └─ 稀疏：BM25（中文按二元组切词）→ 取前 N
        ↓
    RRF 融合：score = Σ 1/(60 + rank)
        ↓
    门槛过滤：语义分 < 0.55 且不是“关键词强命中” → 丢弃
        ↓
    取 top_k 片段作为证据
```

- **为什么要 BM25**：向量检索擅长语义相近，但会把错误码、型号、人名这类
  “必须原样命中”的东西糊掉；BM25 靠词频补上这块。
- **为什么用 RRF 而不是加权求和**：余弦是 0~1，BM25 可以到 30+，
  直接相加等于让 BM25 单方面决定排序；RRF 只看名次，天然免疫量纲差异。
- **为什么还要门槛**：只靠关键词命中容易把“出现了‘公司’两个字”的无关段落
  塞进证据里，把拒答能力搞坏。

## 测试

```bash
pytest test/ -v
```

- `test_agent.py` / `test_agent_stream.py`：决策循环与流式事件（假客户端）
- `test_rag_agent.py`：证据如何进入第二轮模型请求
- `test_knowledge.py`：检索行为 + 拒答（需真实向量模型，缺知识库会自动跳过）
- `test_api_auth.py`：注册登录、鉴权、越权隔离、SSE
- `test/evaluate_retrieval.py`：检索参数扫参（命中率 / 误召回率）
- `test/evaluate_rag_answers.py`：真实模型端到端评估

## 已知限制

- 建表用 `create_all`，生产环境应换成 Alembic 迁移。
- 会话历史只取最近 40 条，长对话没有做摘要压缩。
- JWT 签发后在过期前无法单独作废。
- 知识库索引在进程内存里，重启需重建（多实例部署需要外部向量库）。
- 检索参数（阈值 / top_k）是在很小的示例语料上调的，换语料要重新评估。
