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

| 类别 | 技术 |
|---|---|
| 编程语言 | Python |
| Web 框架 | FastAPI |
| 大语言模型调用 | OpenAI 兼容 API |
| Agent 核心 | Tool Calling、工具注册与参数校验 |
| RAG | Sentence Transformers、余弦相似度、BM25 关键词检索 |
| 参数校验 | Pydantic |
| 数据库 | MySQL、PyMySQL |
| 测试 | pytest |

## 项目结构

```text
dev-agent/
├── agent/
│   ├── agent.py             # Agent 决策循环与工具调用
│   └── session.py           # MySQL 会话持久化
├── api/
│   ├── api_schemas.py       # 请求与响应模型
│   └── api_response.py      # 统一响应格式
├── tools/
│   ├── registry.py          # 工具注册
│   ├── schema.py            # 工具参数模型与 Schema 生成
│   ├── knowledge.py         # 知识加载、向量/关键词混合检索与证据构建
│   ├── text_splitter.py     # 文本分块
│   ├── calculator.py        # 计算工具
│   ├── weather.py           # 天气工具
│   └── order.py             # 订单工具
├── test/                    # 测试代码
├── app.py                   # FastAPI 应用与路由
├── main.py                  # Agent 初始化入口
├── logging_config.py        # 日志配置
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
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 4. 配置环境变量

在项目根目录创建 `.env` 文件，根据 `main.py` 中实际读取的模型配置项，以及 `agent/session.py` 中的数据库配置项填写参数。

数据库配置项：

```dotenv
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=你的数据库密码
MYSQL_DATABASE=agent_db
```

请同时配置 `main.py` 实际使用的大模型 API Key、API Base URL 和模型名称等变量。变量名以项目代码为准，不要直接照抄其他模型平台的配置。

确保 MySQL 服务已启动，并且数据库账号有创建表和读写数据的权限。应用导入和启动时不会连接 MySQL；首次访问会话数据（例如调用 `/chat` 或会话管理接口）时，才会连接数据库并按需创建 `sessions` 表。

### 5. 准备知识库

知识检索模块会从 `knowledge_base/` 目录读取 `.txt` 文件。请创建该目录并放入自己的知识文档。

注意：知识库目录可能不包含在 Git 提交中，需要自行准备测试数据；请勿将公司内部资料、个人隐私或其他敏感信息提交到公开仓库。

Sentence Transformers 模型会在首次实际执行知识检索时加载；首次加载时可能需要下载模型文件，请确保模型能够正常获取。

## 启动服务

在项目根目录运行：

```bash
uvicorn app:app --reload
```

服务启动后访问：

- API 文档：`http://127.0.0.1:8000/docs`
- OpenAPI Schema：`http://127.0.0.1:8000/openapi.json`

## API 示例

### 1. 发送聊天请求

`POST /chat`

请求体：

```json
{
  "message": "请介绍一下知识库中的相关岗位要求",
  "session_id": "demo-session",
  "stream": true
}
```

`stream` 默认为 `true`。流式模式返回 `Content-Type: text/event-stream`，采用 SSE 格式：`delta` 事件逐段包含 `content`，结束时发送带 `session_id` 的 `done` 事件；发生生成错误时发送 `error` 事件。请求方应持续读取响应流并拼接 `delta.content`。

需要兼容原有 JSON 响应时传入 `"stream": false`，返回统一响应结构 `data.answer` 和 `data.session_id`。实际回答取决于模型配置和知识库内容。

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

### 5. 清空会话消息

`POST /sessions/demo-session/clear`

### 6. 删除会话

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

`test/retrieval_dataset.py` 包含 45 条按知识库主题整理的问题（35 条可回答、10 条无答案）；正例标注了标准证据词和允许来源。运行 `python test/evaluate_retrieval.py` 可比较分块、阈值和 Top-K 配置，查看 Recall@K、按相关来源排名计算的 MRR、无答案误召回率和问题类别命中率。当前数据集仍是小规模开发集，评估结果不能代表通用检索准确率。

## 运行测试

安装依赖后，在项目根目录运行：

```bash
python -m pytest test/ -v
```

测试结果应以本地实际运行输出为准。单元测试通过不等于真实模型 API、MySQL 和知识库的端到端集成测试全部通过。

## 项目改进方向

- 增加更完整的 Agent 单元测试与 API 集成测试。
- 统一知识库检索的无证据返回协议。
- 增加检索评估与问答评估数据集。
- 完善部署说明、错误追踪与运行日志。
- 优化知识库更新、检索效果和生产环境配置。

## 项目目标

通过实现一个可运行的 Agent 应用，系统实践大模型工具调用、RAG 检索、会话持久化、API 设计、异常处理和自动化测试等应用开发能力。
