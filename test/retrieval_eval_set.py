"""检索评测集：100 条标注问题（80 可回答 + 20 不可回答）。

标注规则（很重要，直接影响数字的可信度）：

1. **标到文档级，不标片段级。** `docs` 是"能回答这个问题的那几篇文档"，
   命中任意一篇即算相关。片段级标注需要人工指定第几段，
   而这种知识库每篇才几百字，切片边界本身就有歧义 —— 标不准的数字比没数字更糟。

2. **`docs` 必须真的是答案出处。** 写之前逐篇核对过原文，
   不写"大概是这篇"。答非所问的标注会把 recall 变成噪音。

3. **难度分层，故意制造不利于 BM25 的样本。**
   - easy   直接复用文档里的词（BM25 的舒适区）
   - medium 轻度改写
   - hard   刻意避开文档原词（"函数能调"而不是"工具 Schema"），
            以及需要跨段落归纳的问法。这是检验向量检索价值的地方。

4. **负样本里有一半是"强干扰"。** 比如"数据库连接池设多大"、
   "部署到 Kubernetes 怎么配"—— 和知识库存在词汇重叠，
   专门用来测相似度门槛是不是真的在拦东西，而不是靠"完全不相关"蒙过去。

5. **不写能靠常识作答的问题。** 否则测的是模型的先验知识，不是检索。
"""

from __future__ import annotations

CASES: list[dict] = [
    # ================= 公司类（虚构示例文档） =================
    {"id": "ov-1", "query": "公司在哪里？",
     "docs": ["demo_company_overview.txt"], "difficulty": "easy"},
    {"id": "ov-2", "query": "团队有多少人？",
     "docs": ["demo_company_overview.txt"], "difficulty": "easy"},
    {"id": "ov-3", "query": "来访需要提前做什么？",
     "docs": ["demo_company_overview.txt"], "difficulty": "easy"},
    {"id": "ov-4", "query": "公司用什么开发方式推进项目？",
     "docs": ["demo_company_overview.txt"], "difficulty": "medium"},

    {"id": "dep-1", "query": "测试部负责什么？",
     "docs": ["demo_company_departments.txt"], "difficulty": "easy"},
    {"id": "dep-2", "query": "跨部门需求要记录哪些信息？",
     "docs": ["demo_company_departments.txt"], "difficulty": "easy"},
    {"id": "dep-3", "query": "部门职责变了之后文档要怎么办？",
     "docs": ["demo_company_departments.txt"], "difficulty": "medium"},

    {"id": "faq-1", "query": "知识库助手能回答所有公司问题吗？",
     "docs": ["demo_company_faq.txt"], "difficulty": "easy"},
    {"id": "faq-2", "query": "发现文档内容写错了该怎么修？",
     "docs": ["demo_company_faq.txt"], "difficulty": "medium"},

    {"id": "sec-1", "query": "密码和 API Key 能不能写进代码仓库？",
     "docs": ["demo_company_security.txt"], "difficulty": "easy"},
    {"id": "sec-2", "query": "提交代码前要检查哪些东西？",
     "docs": ["demo_company_security.txt"], "difficulty": "easy"},
    {"id": "sec-3", "query": "发现凭据泄露了第一步做什么？",
     "docs": ["demo_company_security.txt"], "difficulty": "medium"},

    {"id": "wf-1", "query": "研发任务开始前要明确什么？",
     "docs": ["demo_company_workflow.txt"], "difficulty": "easy"},
    {"id": "wf-2", "query": "代码评审意见应该怎么写？",
     "docs": ["demo_company_workflow.txt"], "difficulty": "easy"},
    {"id": "wf-3", "query": "需求变了之后要同步更新什么？",
     "docs": ["demo_company_workflow.txt"], "difficulty": "medium"},

    # ================= Agent 类 =================
    {"id": "loop-1", "query": "Agent 的决策循环是怎么执行的？",
     "docs": ["demo_agent_loop.txt"], "difficulty": "easy"},
    {"id": "loop-2", "query": "达到最大迭代次数之后会发生什么？",
     "docs": ["demo_agent_loop.txt"], "difficulty": "medium"},

    {"id": "fail-1", "query": "工具调用失败了应该怎么处理？",
     "docs": ["demo_agent_failures.txt"], "difficulty": "easy"},
    {"id": "fail-2", "query": "外部 API 超时导致工具报错，要不要把错误隐藏起来？",
     "docs": ["demo_agent_failures.txt"], "difficulty": "medium"},

    {"id": "tools-1", "query": "工具注册表是怎么把名字映射到函数的？",
     "docs": ["demo_agent_tools.txt"], "difficulty": "easy"},
    {"id": "tools-2", "query": "模型要求调用一个没注册的工具怎么办？",
     "docs": ["demo_agent_tools.txt"], "difficulty": "medium"},

    {"id": "hist-1", "query": "多轮对话的历史消息有哪些角色？",
     "docs": ["demo_agent_history.txt"], "difficulty": "easy"},
    {"id": "hist-2", "query": "为什么 assistant 的 tool_calls 必须和 tool 结果关联起来？",
     "docs": ["demo_agent_history.txt"], "difficulty": "medium"},
    {"id": "hist-3", "query": "不同用户的对话历史能混在一起吗？",
     "docs": ["demo_agent_history.txt"], "difficulty": "medium"},

    {"id": "schema-1", "query": "工具参数的 Schema 里包含哪些信息？",
     "docs": ["demo_agent_schema.txt"], "difficulty": "easy"},
    {"id": "schema-2", "query": "模型给的参数能不能直接信任？",
     "docs": ["demo_agent_schema.txt"], "difficulty": "medium"},

    # ================= HTTP / 数据库 / 部署 =================
    {"id": "api-1", "query": "REST 接口对不存在的资源应该怎么处理？",
     "docs": ["demo_api_rest.txt"], "difficulty": "easy"},
    {"id": "api-2", "query": "删除一个会话应该用什么 HTTP 方法和路径？",
     "docs": ["demo_api_rest.txt"], "difficulty": "medium"},

    {"id": "db-1", "query": "会话历史怎么存到 MySQL 里？",
     "docs": ["demo_database_mysql.txt"], "difficulty": "easy"},
    {"id": "db-2", "query": "怎么降低 SQL 注入的风险？",
     "docs": ["demo_database_mysql.txt"], "difficulty": "easy"},
    {"id": "db-3", "query": "消息数组适合用什么字段类型保存？",
     "docs": ["demo_database_mysql.txt"], "difficulty": "medium"},

    {"id": "tran-1", "query": "事务能解决所有并发问题吗？",
     "docs": ["demo_database_transactions.txt"], "difficulty": "medium"},
    {"id": "tran-2", "query": "为什么不要在事务里做长时间的网络调用？",
     "docs": ["demo_database_transactions.txt"], "difficulty": "medium"},

    {"id": "depr-1", "query": "环境配置应该写在哪里？",
     "docs": ["demo_deployment_config.txt"], "difficulty": "easy"},
    {"id": "depr-2", "query": ".env 文件能不能提交到公开仓库？",
     "docs": ["demo_deployment_config.txt"], "difficulty": "easy"},
    {"id": "depr-3", "query": "启动前要验证哪些东西？",
     "docs": ["demo_deployment_config.txt"], "difficulty": "medium"},

    # ================= FastAPI 类 =================
    {"id": "err-1", "query": "怎么统一处理 FastAPI 的异常？",
     "docs": ["demo_fastapi_errors.txt"], "difficulty": "easy"},
    {"id": "err-2", "query": "统一响应格式是不是意味着所有错误都返回 200？",
     "docs": ["demo_fastapi_errors.txt"], "difficulty": "medium"},

    {"id": "mw-1", "query": "中间件能做什么？",
     "docs": ["demo_fastapi_middleware.txt"], "difficulty": "easy"},
    {"id": "mw-2", "query": "请求日志里应该记录哪些字段？",
     "docs": ["demo_fastapi_middleware.txt", "demo_observability_logs.txt"],
     "difficulty": "medium"},

    {"id": "rt-1", "query": "FastAPI 怎么声明一个路由？",
     "docs": ["demo_fastapi_routes.txt"], "difficulty": "easy"},
    {"id": "rt-2", "query": "接口文档是怎么自动生成的？",
     "docs": ["demo_fastapi_routes.txt"], "difficulty": "medium"},

    {"id": "val-1", "query": "怎么限制请求字段的长度？",
     "docs": ["demo_fastapi_validation.txt"], "difficulty": "easy"},
    {"id": "val-2", "query": "参数校验应该在业务逻辑之前还是之后？",
     "docs": ["demo_fastapi_validation.txt"], "difficulty": "medium"},

    {"id": "obs-1", "query": "可观测性一般记录哪些指标？",
     "docs": ["demo_observability_logs.txt"], "difficulty": "easy"},

    # ================= Python 基础 =================
    {"id": "async-1", "query": "什么场景适合用异步？",
     "docs": ["demo_python_async.txt"], "difficulty": "easy"},
    {"id": "exc-1", "query": "try except else finally 分别什么时候执行？",
     "docs": ["demo_python_exceptions.txt"], "difficulty": "easy"},
    {"id": "mod-1", "query": "Python 的模块和包有什么区别？",
     "docs": ["demo_python_modules.txt"], "difficulty": "easy"},
    {"id": "test-1", "query": "pytest 里的 fixture 是干什么的？",
     "docs": ["demo_python_testing.txt"], "difficulty": "easy"},
    {"id": "type-1", "query": "想让外部数据强制校验该用什么？",
     "docs": ["demo_python_types.txt", "demo_fastapi_validation.txt"],
     "difficulty": "medium"},

    # ================= RAG 类 =================
    {"id": "chunk-1", "query": "分块太小或者太大会有什么问题？",
     "docs": ["demo_rag_chunking.txt"], "difficulty": "medium"},
    {"id": "chunk-2", "query": "重叠分块解决了什么问题？",
     "docs": ["demo_rag_chunking.txt"], "difficulty": "medium"},
    {"id": "chunk-3", "query": "分块大小应该怎么定？",
     "docs": ["demo_rag_chunking.txt"], "difficulty": "medium"},

    {"id": "emb-1", "query": "向量相似度高就代表内容一定正确吗？",
     "docs": ["demo_rag_embeddings.txt"], "difficulty": "medium"},

    {"id": "eval-1", "query": "Recall@K 和 MRR 分别衡量什么？",
     "docs": ["demo_rag_evaluation.txt"], "difficulty": "medium"},
    {"id": "eval-2", "query": "评估集里为什么要有无答案的问题？",
     "docs": ["demo_rag_evaluation.txt"], "difficulty": "medium"},

    {"id": "pipe-1", "query": "RAG 的完整流程包括哪些步骤？",
     "docs": ["demo_rag_pipeline.txt"], "difficulty": "medium"},
    {"id": "pipe-2", "query": "RAG 能保证答案一定正确吗？",
     "docs": ["demo_rag_pipeline.txt"], "difficulty": "medium"},

    {"id": "ref-1", "query": "检索不到证据的时候应该怎么回答？",
     "docs": ["demo_rag_refusal.txt"], "difficulty": "medium"},
    {"id": "ref-2", "query": "怎么避免靠调高阈值刷出好看的拒答率？",
     "docs": ["demo_rag_refusal.txt"], "difficulty": "medium"},

    {"id": "sim-1", "query": "余弦相似度是用来做什么的？",
     "docs": ["demo_rag_similarity.txt"], "difficulty": "easy"},
    {"id": "sim-2", "query": "Top-K 的 K 太大或太小会怎样？",
     "docs": ["demo_rag_similarity.txt"], "difficulty": "medium"},
    {"id": "sim-3", "query": "相似度阈值应该怎么确定？",
     "docs": ["demo_rag_similarity.txt"], "difficulty": "medium"},

    # ================= 招聘 / 验收 / 简历 =================
    {"id": "rec-1", "query": "Agent 开发实习岗位要求哪些能力？",
     "docs": ["demo_recruitment_agent.txt"], "difficulty": "medium"},
    {"id": "rec-2", "query": "如果项目里有 RAG，面试时要能解释哪些点？",
     "docs": ["demo_recruitment_agent.txt"], "difficulty": "medium"},
    {"id": "rec-3", "query": "后端实习岗位要求哪些基础？",
     "docs": ["demo_recruitment_backend.txt"], "difficulty": "medium"},
    {"id": "rec-4", "query": "后端岗位的加分项有哪些？",
     "docs": ["demo_recruitment_backend.txt"], "difficulty": "medium"},
    {"id": "rec-5", "query": "实习项目验收要检查什么？",
     "docs": ["demo_recruitment_evaluation.txt"], "difficulty": "medium"},
    {"id": "rec-6", "query": "只展示一次成功的聊天结果能证明什么？",
     "docs": ["demo_recruitment_evaluation.txt"], "difficulty": "medium"},
    {"id": "rec-7", "query": "准备项目演示时应该能说明什么？",
     "docs": ["demo_recruitment_interview.txt"], "difficulty": "medium"},
    {"id": "rec-8", "query": "简历里的项目描述应该包含什么？",
     "docs": ["demo_recruitment_resume.txt"], "difficulty": "medium"},
    {"id": "rec-9", "query": "计划中的功能能写成已完成吗？",
     "docs": ["demo_recruitment_resume.txt"], "difficulty": "medium"},
    {"id": "rec-10", "query": "写性能提升时要注明什么？",
     "docs": ["demo_recruitment_resume.txt"], "difficulty": "medium"},

    {"id": "tm-1", "query": "单元测试、接口测试、集成测试分别测什么？",
     "docs": ["demo_testing_metrics.txt"], "difficulty": "medium"},
    {"id": "tm-2", "query": "评估报告应该记录哪些信息？",
     "docs": ["demo_testing_metrics.txt"], "difficulty": "medium"},

    # ================= hard：刻意避开文档原词 =================
    {"id": "hd-1", "query": "模型是怎么知道有哪些函数能调、每个函数要传什么东西的？",
     "docs": ["demo_agent_schema.txt"], "difficulty": "hard"},
    {"id": "hd-2", "query": "助手一直反复调同一个函数停不下来，怎么办？",
     "docs": ["demo_agent_loop.txt"], "difficulty": "hard"},
    {"id": "hd-3", "query": "模型自己知道的事情能不能当公司规定来回答？",
     "docs": ["demo_rag_refusal.txt"], "difficulty": "hard"},
    {"id": "hd-4", "query": "打日志的时候能把认证令牌一起打出来吗？",
     "docs": ["demo_company_security.txt", "demo_fastapi_middleware.txt"],
     "difficulty": "hard"},
    {"id": "hd-5", "query": "员工是一周都得到公司上班吗？",
     "docs": ["demo_company_overview.txt"], "difficulty": "hard"},
    {"id": "hd-6", "query": "把文档切碎了对检索有什么坏处？",
     "docs": ["demo_rag_chunking.txt"], "difficulty": "hard"},
    {"id": "hd-7", "query": "两个人同时往同一个会话写消息，会互相覆盖吗？",
     "docs": ["demo_database_transactions.txt"], "difficulty": "hard"},
    {"id": "hd-8", "query": "怎么限制用户输入不超过四千字？",
     "docs": ["demo_fastapi_validation.txt"], "difficulty": "hard"},

    # ================= 负样本：知识库里没有答案 =================
    # 前 10 条与知识库毫无词汇关联
    {"id": "neg-1", "query": "今天股市大盘怎么样？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-2", "query": "明天北京天气如何？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-3", "query": "公司去年的营业收入是多少？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-4", "query": "员工一年有几天年假？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-5", "query": "报销流程是什么？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-6", "query": "公积金缴纳比例是多少？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-7", "query": "公司的 CEO 是谁？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-8", "query": "食堂几点开饭？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-9", "query": "怎么申请调岗？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-10", "query": "加班费怎么算？",
     "docs": [], "difficulty": "negative"},
    # 后 10 条是强干扰：和知识库共享词汇，专门测门槛
    {"id": "neg-11", "query": "公司的股票代码是多少？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-12", "query": "公司上市了吗？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-13", "query": "深圳到北京的机票多少钱？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-14", "query": "前端框架应该用 React 还是 Vue？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-15", "query": "数据库连接池大小应该设多少？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-16", "query": "这个项目部署到 Kubernetes 要怎么配置？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-17", "query": "系统能支持多少用户同时在线？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-18", "query": "公司的服务器有多少台？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-19", "query": "培训预算有多少钱？",
     "docs": [], "difficulty": "negative"},
    {"id": "neg-20", "query": "办公用品去哪里领？",
     "docs": [], "difficulty": "negative"},
]


def positives() -> list[dict]:
    return [c for c in CASES if c["docs"]]


def negatives() -> list[dict]:
    return [c for c in CASES if not c["docs"]]


def by_difficulty(level: str) -> list[dict]:
    return [c for c in CASES if c["difficulty"] == level]


if __name__ == "__main__":
    from collections import Counter

    counts = Counter(c["difficulty"] for c in CASES)
    print(f"总用例：{len(CASES)}")
    print(f"可回答：{len(positives())}   不可回答：{len(negatives())}")
    for level in ("easy", "medium", "hard", "negative"):
        print(f"  {level:9s} {counts[level]}")

    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)), "用例 id 重复"
    covered = {d for c in CASES for d in c["docs"]}
    print(f"覆盖文档：{len(covered)} 篇")
