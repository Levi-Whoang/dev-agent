# 定义检索评估问题及标准答案
# 正例：34 条；无答案案例：8 条；总计：42 条

EVALUATION_CASES = [
    # ==================== 招聘与岗位要求 ====================
    {
        "query": "这个实习岗位需要掌握哪些技术？",
        "expected": ["Python", "FastAPI", "LLM API", "RAG"],
        "sources": ["company.txt"],
    },
    {
        "query": "后端开发实习生需要掌握什么？",
        "expected": ["Python", "Java", "REST API", "关系型数据库"],
        "sources": ["demo_recruitment_backend.txt"],
    },
    {
        "query": "Agent 应用开发岗位看重哪些能力？",
        "expected": ["Prompt 设计", "Tool Calling", "JSON 参数校验"],
        "sources": ["demo_recruitment_agent.txt"],
    },
    {
        "query": "后端实习生有哪些加分项？",
        "expected": ["Git", "Docker", "FastAPI", "Spring Boot"],
        "sources": ["demo_recruitment_backend.txt"],
    },
    {
        "query": "Agent 方向的技术面试可能考什么？",
        "expected": ["模型为什么调用工具", "消息历史如何保存"],
        "sources": ["demo_recruitment_interview.txt"],
    },
    {
        "query": "简历应该如何描述项目经历？",
        "expected": ["项目目标", "技术栈", "个人负责的模块"],
        "sources": ["demo_recruitment_resume.txt"],
    },

    # ==================== Agent 与 Tool Calling ====================
    {
        "query": "Agent 的基本决策循环是什么？",
        "expected": ["解析参数", "验证参数", "执行工具"],
        "sources": ["demo_agent_loop.txt"],
    },
    {
        "query": "Agent 如何根据工具名称找到对应函数？",
        "expected": ["工具注册表", "Python 可调用对象"],
        "sources": ["demo_agent_tools.txt"],
    },
    {
        "query": "工具注册表中保存的是什么？",
        "expected": ["工具名称", "Python 可调用对象"],
        "sources": ["demo_agent_tools.txt"],
    },
    {
        "query": "为什么需要给大模型提供工具参数 Schema？",
        "expected": ["参数字段", "参数类型", "Pydantic"],
        "sources": ["demo_agent_schema.txt"],
    },
    {
        "query": "模型生成的 JSON 参数不合法怎么办？",
        "expected": ["格式错误", "参数验证错误"],
        "sources": ["demo_agent_schema.txt", "demo_agent_failures.txt"],
    },
    {
        "query": "Agent 为什么需要限制最大迭代次数？",
        "expected": ["最大迭代次数", "无限调用循环"],
        "sources": ["demo_agent_loop.txt"],
    },
    {
        "query": "Agent 如何管理多轮对话的消息？",
        "expected": ["system", "user", "assistant", "tool"],
        "sources": ["demo_agent_history.txt"],
    },
    {
        "query": "为什么不同用户的对话历史需要隔离？",
        "expected": ["按会话隔离", "不同用户的对话混合"],
        "sources": ["demo_agent_history.txt"],
    },
    {
        "query": "工具执行失败后应该如何处理？",
        "expected": ["参数解析错误", "参数验证错误", "工具运行时错误"],
        "sources": ["demo_agent_failures.txt"],
    },
    {
        "query": "Agent 应该如何处理未知工具？",
        "expected": ["未知工具", "明确错误"],
        "sources": ["demo_agent_tools.txt"],
    },

    # ==================== RAG 检索 ====================
    {
        "query": "RAG 的完整处理流程包括哪些步骤？",
        "expected": ["文档加载", "分块", "向量编码", "相关内容检索"],
        "sources": ["demo_rag_pipeline.txt"],
    },
    {
        "query": "RAG 为什么需要对文档进行分块？",
        "expected": ["长文档", "文本分块"],
        "sources": ["demo_rag_chunking.txt"],
    },
    {
        "query": "文本分块过小会有什么问题？",
        "expected": ["丢失上下文"],
        "sources": ["demo_rag_chunking.txt"],
    },
    {
        "query": "重叠分块有什么作用？",
        "expected": ["重要信息", "边界两侧", "丢失风险"],
        "sources": ["demo_rag_chunking.txt"],
    },
    {
        "query": "Embedding 模型的作用是什么？",
        "expected": ["数值向量", "语义相关"],
        "sources": ["demo_rag_embeddings.txt"],
    },
    {
        "query": "向量相似度能否证明答案正确？",
        "expected": ["相关性的估计", "事实正确性"],
        "sources": ["demo_rag_embeddings.txt"],
    },
    {
        "query": "余弦相似度检索如何排序？",
        "expected": ["相似度", "降序排列", "前 K 个结果"],
        "sources": ["demo_rag_similarity.txt"],
    },
    {
        "query": "Top-K 参数会影响什么？",
        "expected": ["候选片段", "噪声", "上下文长度"],
        "sources": ["demo_rag_similarity.txt"],
    },
    {
        "query": "Recall@K 衡量什么？",
        "expected": ["相关证据"],
        "sources": ["demo_rag_evaluation.txt"],
    },
    {
        "query": "MRR 关注哪个检索指标？",
        "expected": ["第一个相关结果出现的位置"],
        "sources": ["demo_rag_evaluation.txt"],
    },
    {
        "query": "知识库没有足够证据时应该怎么办？",
        "expected": ["无法确认答案", "自行编造"],
        "sources": ["demo_rag_refusal.txt"],
    },
    {
        "query": "RAG 为什么不能保证回答正确？",
        "expected": ["过时的信息", "低质量的分块", "检索参数"],
        "sources": ["demo_rag_pipeline.txt"],
    },

    # ==================== Python 与 FastAPI ====================
    {
        "query": "Python 类型标注有什么作用？",
        "expected": ["代码可读性", "静态检查工具"],
        "sources": ["demo_python_types.txt"],
    },
    {
        "query": "Python 的 finally 在什么时候执行？",
        "expected": ["无论成功或失败", "清理逻辑"],
        "sources": ["demo_python_exceptions.txt"],
    },
    {
        "query": "Python 模块和包如何组织代码？",
        "expected": [".py 文件", "组织多个相关模块"],
        "sources": ["demo_python_modules.txt"],
    },
    {
        "query": "FastAPI 的 GET 和 POST 路由如何声明？",
        "expected": ["@app.get", "@app.post"],
        "sources": ["demo_fastapi_routes.txt"],
    },
    {
        "query": "FastAPI 如何校验请求参数长度？",
        "expected": ["Field(min_length=1, max_length=4000)"],
        "sources": ["demo_fastapi_validation.txt"],
    },
    {
        "query": "API 为什么需要统一异常响应？",
        "expected": ["code", "message", "data"],
        "sources": ["demo_fastapi_errors.txt"],
    },
    {
        "query": "FastAPI 中间件可以记录哪些信息？",
        "expected": ["HTTP 方法", "状态码", "处理耗时"],
        "sources": ["demo_fastapi_middleware.txt"],
    },

    # ==================== 数据库、测试与部署 ====================
    {
        "query": "MySQL 会话持久化需要注意什么？",
        "expected": ["session_id", "参数化 SQL", "事务"],
        "sources": ["demo_database_mysql.txt"],
    },

    # ==================== 无答案问题 ====================
    {
        "query": "星河科技的真实老板叫什么名字？",
        "expected": None,
        "category": "no_answer",
    },
    {
        "query": "星河科技的真实注册资本是多少？",
        "expected": None,
        "category": "no_answer",
    },
    {
        "query": "星河科技的真实成立日期是什么时候？",
        "expected": None,
        "category": "no_answer",
    },
    {
        "query": "这个岗位承诺的具体月薪是多少元？",
        "expected": None,
        "category": "no_answer",
    },
    {
        "query": "公司 HR 的真实手机号码是什么？",
        "expected": None,
        "category": "no_answer",
    },
    {
        "query": "星河科技真实办公楼的具体门牌号是多少？",
        "expected": None,
        "category": "no_answer",
    },
    {
        "query": "公司今年实际招聘了多少名实习生？",
        "expected": None,
        "category": "no_answer",
    },
    {
        "query": "公司今年实际营业收入是多少？",
        "expected": None,
        "category": "no_answer",
    },
]
