# Dev Agent — 企业知识库问答助手

一个基于 **FastAPI + 异步 SQLAlchemy + 混合检索 RAG + 流式输出** 的 Agent 应用。

核心链路：一个显式的 **ReAct 循环** —— `Thought → Action → Observation` 反复推进，
直到模型不再调工具、直接给出答案。知识类问题由「向量召回 + BM25 召回 + RRF 融合」
的混合检索提供证据。

## 功能特性

| 能力 | 说明 |
|---|---|
| 用户体系 | 注册 / 登录 / JWT 鉴权，会话按用户隔离 |
| ReAct 循环 | 思考 → 调工具 → 看结果，循环推进；带步数预算与重复动作守卫 |
| Agent 工具调用 | 自动决定是否调工具，参数经 Pydantic 校验后执行 |
| 三层记忆 | 长期记忆 + 会话摘要 + 最近几轮原文 |
| 混合检索 RAG | 稠密向量（bge-small-zh）+ 稀疏 BM25 + RRF 融合排序 |
| 拒答机制 | 证据不足返回 `NO_RELEVANT_EVIDENCE`，禁止模型猜测 |
| 流式输出 | SSE 逐字返回，思考与工具调用过程也作为事件推送 |
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
│   ├── agent.py            # ReAct 循环（run / stream 共用一条 _cycle）
│   ├── react.py            # 循环策略层：轨迹 / 卡死守卫 / 观测压缩（纯逻辑）
│   ├── events.py           # SSE 事件定义
│   ├── prompts.py          # 系统提示词
│   ├── store.py            # 会话 / 消息读写（SQLAlchemy）
│   ├── tools_runner.py     # 工具参数校验与执行
│   ├── factory.py          # 按请求装配 Agent
│   └── memory/             # 三层记忆
│       ├── tokens.py       # token 估算（不引 tiktoken）
│       ├── window.py       # 按「轮」裁剪，保证工具配对完整
│       ├── summarizer.py   # 滚动摘要
│       └── longterm.py     # 跨会话用户记忆
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
│   ├── memory_tool.py      # remember：写入长期记忆（闭包绑用户）
│   └── calculator.py / weather.py / order.py
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
| `MEMORY_WINDOW_TOKENS` / `MEMORY_MAX_TURNS` | 最近几轮原文的预算（默认 3000 token / 12 轮） |
| `MEMORY_SUMMARY_ENABLED` | 早期对话是否压成摘要（默认开） |
| `MEMORY_LONGTERM_ENABLED` | 跨会话长期记忆 + `remember` 工具（默认开） |
| `REACT_MAX_ITERATIONS` | 一次提问最多几轮「思考→动作→观测」（默认 10） |
| `REACT_LOOP_THRESHOLD` | 同一工具同一参数重复到第几次开始提醒（默认 2） |
| `REACT_OBSERVATION_LIMIT` | 单条工具结果回灌前的截断长度（默认 1200 字符） |

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
event: delta       {"text": "增量文本"}          # 模型的正文增量
event: thought     {"text": "我要先查报销制度", "step": 1}
event: tool_start  {"name": "knowledge_search_context", "arguments": "{...}", "step": 1}
event: tool_result {"name": "...", "result": "...", "ok": true, "step": 1}
event: done        {"answer": "...", "termination": "answer", "steps": 2,
                    "tool_calls": 1, "thoughts": 1, "trace": [...]}
event: error       {"message": "出错原因", "termination": "loop"}
```

两条约定：

- **`done` 永远是最后一个事件**，终止原因统一放在 `termination` 里
  （`answer` / `max_steps` / `loop` / `error`）。客户端只需要认这一个终态，
  不必靠"连接断了"去猜结果。
- **`thought` 只描述推理，不携带答案。** 它是给不渲染增量的客户端
  （脚本、日志回放）用的；页面本身靠 `delta` 就能看到同样的文字。

前端用 `fetch + ReadableStream` 解析，不用 `EventSource`
（后者只能发 GET，也带不了 `Authorization` 头）。

## ReAct 循环是怎么做的

ReAct = **Rea**soning + **Act**ing。一问一答不再是一次模型调用，而是一个循环：

```text
        ┌─────────────────────────────┐
        │  Thought  说明"我要查什么"    │
        │     ↓                        │
        │  Action   发起工具调用        │
        │     ↓                        │
        │  Observation  工具结果回灌     │
        └─────────────┬───────────────┘
                      ↓
        没有工具调用 = 给出最终答案，结束
```

**三条设计主线**（代码见 `agent/agent.py`、策略层在 `agent/react.py`）：

1. **只有一份循环。** 原来的 `run()` 和 `stream()` 各写了一遍决策逻辑，
   改一处必须记得改两处，迟早分叉。现在两者都消费同一个 `_cycle()` 异步生成器：
   流式模式把增量吐成事件，非流式模式吞掉增量只取最终答案。
   "要不要流式"只在调模型那一步分叉。

2. **动作走 function calling，只有思考用自然语言。** 论文原版要求模型输出
   `Action: search({"q": "..."})` 再由正则解析——参数里多一个引号就崩，
   工具名拼错要到运行时才发现。这里用结构化调用（schema 校验 + 调用 id 配对），
   只把「思考」这一步显式化：拿论文的好处，不接它的脆弱。

3. **循环必须能停下来。** 光有 `max_iterations` 不够——模型会拿一模一样的参数
   反复重试同一个失败的工具，把预算烧光。所以还有 `LoopGuard`：

   | 同一工具 + 同一参数 | 处理 |
   | --- | --- |
   | 第 1 次 | 正常执行 |
   | 第 `threshold` 次（默认 2） | 在观测结果前贴一句提醒，让模型换思路 |
   | 再重复一次 | 直接中断，`termination=loop` |

   参数比较前先做归一化（`json.dumps(sort_keys=True)`），否则
   `{"a":1,"b":2}` 和 `{"b":2,"a":1}` 会被当成两个不同动作，守卫形同虚设。

另外两处细节：

- **观测结果压缩**：工具输出超过 1200 字就截断（`REACT_OBSERVATION_LIMIT`）。
  一条检索结果两三千字，原样回灌两次就吃满窗口。
- **思考不伪造**：模型不说就直接调工具时，`extract_thought` 返回空字符串，
  不会编一句"用户想了解…"塞进轨迹——伪造的推理比没有更糟。

`done` 事件里的 `trace` 是完整的推理轨迹（每步的思考、动作、观测、是否终止），
前端和历史回放都靠它区分「思考」和「正式答案」。

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

## 记忆是怎么做的

三层，各管一段，失败互不影响：

| 层 | 存在哪 | 管什么 | 失效时 |
|---|---|---|---|
| 长期记忆 | `user_memories` 表 | 这个人是谁（职业、偏好、常驻地点） | 跳过，不影响回答 |
| 会话摘要 | `chat_sessions.summary` | 这段对话早期聊到哪了 | 沿用旧摘要 |
| 最近原文 | `chat_messages` | 刚才具体说了什么 | 无 |

```text
system prompt
 ├─ 长期记忆块（按相关度挑，≤320 token）
 ├─ 【早期对话摘要】（挤出去的轮次压成 300 字）
 └─ 最近 N 轮原文（≤12 轮 / ≤3000 token）
```

**为什么按「轮」裁剪，不按「条」裁剪** —— 这是整个模块最容易写错的地方。

OpenAI 协议要求：任何 `role="tool"` 的消息，前面必须紧跟一条带 `tool_calls`
的 `assistant`。而一次工具调用会写入**两行**。原来用
`ORDER BY id DESC LIMIT 40` 取最近 40 行，边界有可能正好落在一行 `tool` 上，
于是送出去的对话以一条孤立的 tool 消息开头，模型直接返回 400。

实测：历史 41/43/45 行正常，42/44/46 行全部报错 —— 一半概率触发，
开发时极难复现。

修法不是"少取一条"或者"跳过 tool"，而是换最小单位：

```text
一轮 = 一条 user 消息 + 它之后直到下一条 user 之前的所有消息
```

按轮切，`assistant(tool_calls)` 和它的 `tool` 结果天然在同一边，不可能被切开。
`test_memory.py::test_no_orphan_tool_at_any_boundary` 扫了 90 种边界组合钉死这一点。

**为什么摘要要"攒够了再压"**：摘要是一次额外的模型调用。每轮都压，成本和延迟都翻倍。
所以挤出去的内容累积到 `MEMORY_SUMMARY_TRIGGER_TOKENS`（默认 800）才压一次，
而且只压"尚未被摘要覆盖"的部分（靠 `summary_upto_id` 记录进度）。

**为什么长期记忆不做向量检索**：记忆条数是几十条量级，为几十条数据加载
embedding 模型，冷启动几百毫秒，收益远小于成本。用字面二元组重合度排序即可。
真到几千条再换。

记忆由 `remember` 工具写入 —— 让模型自己判断"这条值得记"，比每轮自动抽取省得多。
工具用闭包绑定了 `db` 和 `user_id`，所以模型看到的参数只有 `content` / `kind`，
拿不到也改不了身份。

## 测试

```bash
pytest test/ -v
```

- `test_agent.py` / `test_agent_stream.py`：决策循环与流式事件（假客户端）
- `test_rag_agent.py`：证据如何进入第二轮模型请求
- `test_memory.py`：窗口切分（90 种边界）、token 估算、摘要、长期记忆、remember 工具
- `test_knowledge.py`：检索行为 + 拒答 + 消融开关（需真实向量模型，缺知识库会自动跳过）
- `test_api_auth.py`：注册登录、鉴权、越权隔离、SSE
- `test/evaluate_rag_answers.py`：真实模型端到端评估

## 检索效果评测

```bash
python test/evaluate_retrieval.py           # 四组消融
python test/evaluate_retrieval.py --sweep   # 额外做阈值扫描 + 留出集验证
```

评测集在 `test/retrieval_eval_set.py`：**103 条**（83 可回答 / 20 不可回答），
覆盖全部 36 篇文档，按难度分 easy / medium / hard（hard 刻意避开文档原词）。
不可回答里有 10 条是**强干扰**——和知识库共享词汇，专门测门槛是否真的在拦。

同一份标注集跑四遍，对比 `dense` / `sparse` / `hybrid` / `hybrid 关门槛`：

```text
配置               Recall@3   MRR@10   拒答准确   误拒率
仅向量 dense         96.4%    0.903     45.0%    0.0%
仅关键词 BM25        95.2%    0.932     45.0%    0.0%
混合 RRF（生产）      97.6%    0.950     45.0%    0.0%
混合 · 关门槛        98.8%    0.955      0.0%    0.0%
```

**但这组百分比不能直接用。** 配对比较（逐条比而不是比总分）之后：

```text
对比                     混合赢   混合输    打平    净差
混合 vs 仅向量 dense          1       0      82      +1
混合 vs 仅关键词 BM25          2       0      81      +2
混合 vs 混合·关门槛            0       1      82      -1

全部四种配置都命中：78/83   全部失败：1/83   真正能区分优劣：4/83
```

三条要知道的：

- **在这个语料上，混合检索的优势测不出来。** 83 条样本里 1 条 = 1.2pp，
  而 78/83 是"怎么检都中"——天花板效应。所以这里不声称"混合更好"。
  要比较三种检索方式，得先把评测集做难：扩语料、加同主题不同答案的干扰文档、
  问题改写成口语短句。
- **拒答是短板。** 20 条负样本里 11 条给了证据。诊断显示它们**不是**从
  "关键词兜底"通道进来的，而是自己就越过了向量门槛——embedding 衡量的是
  主题相关，不是答案蕴含。
- **调阈值是零和交易。** 留出集上把拒答从 71% 拉到 100%，召回从 100% 掉到 78.6%。
  所以默认参数没动。

Recall@3 按**不同文档**去重计算，否则一篇切成多个片段会独占前几名。

## 已知限制

- 建表用 `create_all` + 一个幂等的补列逻辑（`database/base.py:_add_missing_columns`），
  够单人开发用；多人协作或需要回滚时应换 Alembic。
- 摘要触发阈值以下被挤出窗口的内容，在攒够之前模型看不到（原文仍在库里）。
- 长期记忆条数上限 60 条，超出后 `remember` 会拒绝写入，需要先手动删。
- JWT 签发后在过期前无法单独作废。
- 知识库索引在进程内存里，重启需重建（多实例部署需要外部向量库）。
- 检索参数（阈值 / top_k）是在很小的示例语料上调的，换语料要重新评估。
