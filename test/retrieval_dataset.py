"""RAG 检索评估集：问题、相关证据词和允许的知识来源。"""

EVALUATION_CASES = [
    # 企业与岗位信息：包括改写、细节查询和相近问题。
    {"query": "这家公司的办公地点在哪个区？", "expected": ["深圳市南山区"], "sources": ["company.txt", "faq.txt"], "category": "paraphrase"},
    {"query": "公司目前在招什么职位？", "expected": ["Agent 应用开发实习生"], "sources": ["company.txt", "recruitment.txt"], "category": "paraphrase"},
    {"query": "这个实习岗位要求掌握哪些技术栈？", "expected": ["Python", "FastAPI", "LLM API", "RAG"], "sources": ["company.txt", "recruitment.txt"], "category": "multi_fact"},
    {"query": "候选人还需要了解哪些模型交互和检索能力？", "expected": ["Tool Calling", "提示词设计", "知识检索"], "sources": ["company.txt", "recruitment.txt"], "category": "paraphrase"},
    {"query": "岗位名称是什么？", "expected": ["Agent 应用开发实习生"], "sources": ["company.txt", "recruitment.txt"], "category": "keyword"},

    # Agent 执行与工具调用。
    {"query": "工具调用参数应该如何防止模型生成错误？", "expected": ["Pydantic", "独立校验"], "sources": ["demo_agent_schema.txt"], "category": "paraphrase"},
    {"query": "Agent 怎么避免陷入无限次调用工具？", "expected": ["最大迭代次数", "最大轮数限制"], "sources": ["demo_agent_loop.txt"], "category": "paraphrase"},
    {"query": "不同用户的对话记录如何避免串在一起？", "expected": ["按会话隔离"], "sources": ["demo_agent_history.txt"], "category": "paraphrase"},
    {"query": "工具执行失败后，错误应该怎样反馈？", "expected": ["失败不应被伪装成成功结果", "重试策略"], "sources": ["demo_agent_failures.txt"], "category": "paraphrase"},
    {"query": "工具注册表负责什么？", "expected": ["工具名称", "可调用对象"], "sources": ["demo_agent_tools.txt"], "category": "keyword"},

    # RAG 流程与检索质量。
    {"query": "长文档为什么要切片并保留重叠？", "expected": ["文本分块", "重要信息", "重叠分块"], "sources": ["demo_rag_chunking.txt"], "category": "paraphrase"},
    {"query": "Embedding 向量相似是否能证明事实正确？", "expected": ["不等于事实正确性"], "sources": ["demo_rag_embeddings.txt"], "category": "paraphrase"},
    {"query": "Recall@K 和 MRR 分别衡量什么？", "expected": ["Recall@K", "MRR"], "sources": ["demo_rag_evaluation.txt"], "category": "keyword"},
    {"query": "RAG 从原始文档到生成答案有哪些主要步骤？", "expected": ["文档加载", "文本清洗", "向量编码"], "sources": ["demo_rag_pipeline.txt"], "category": "multi_fact"},
    {"query": "资料里没有答案时，知识库助手应怎么做？", "expected": ["资料缺失", "编造公司政策"], "sources": ["demo_company_faq.txt"], "category": "paraphrase"},
    {"query": "余弦相似度怎么用于取 Top-K 片段？", "expected": ["余弦相似度", "Top-K"], "sources": ["demo_rag_similarity.txt"], "category": "keyword"},

    # Python、API 和数据库基础主题。
    {"query": "pytest 里 fixture 和 monkeypatch 有什么用途？", "expected": ["fixture", "monkeypatch"], "sources": ["demo_python_testing.txt"], "category": "keyword"},
    {"query": "Python 的 finally 代码块什么时候运行？", "expected": ["finally", "无论成功或失败"], "sources": ["demo_python_exceptions.txt"], "category": "keyword"},
    {"query": "在异步 FastAPI 路由中直接执行阻塞函数有什么影响？", "expected": ["同步阻塞操作", "事件循环"], "sources": ["demo_python_async.txt"], "category": "paraphrase"},
    {"query": "Python 类型标注能在运行时自动校验输入吗？", "expected": ["不会在运行时自动拒绝"], "sources": ["demo_python_types.txt"], "category": "paraphrase"},
    {"query": "GET 和 POST 会话接口分别用来做什么？", "expected": ["GET /sessions", "POST /sessions"], "sources": ["demo_api_rest.txt"], "category": "keyword"},
    {"query": "FastAPI 的请求字段长度限制可以用什么定义？", "expected": ["Field(min_length=1, max_length=4000)"], "sources": ["demo_fastapi_validation.txt"], "category": "keyword"},
    {"query": "FastAPI 出错时如何区分资源不存在和服务端故障？", "expected": ["合理的状态码", "HTTPException"], "sources": ["demo_fastapi_errors.txt"], "category": "paraphrase"},
    {"query": "请求中间件日志适合记录哪些信息？", "expected": ["HTTP 方法", "状态码", "处理耗时"], "sources": ["demo_fastapi_middleware.txt"], "category": "multi_fact"},
    {"query": "数据库事务提交或回滚解决什么问题？", "expected": ["整体提交或回滚", "数据一致性"], "sources": ["demo_database_transactions.txt"], "category": "paraphrase"},
    {"query": "保存 MySQL 会话消息时，参数化 SQL 有什么好处？", "expected": ["参数化 SQL", "SQL 注入"], "sources": ["demo_database_mysql.txt"], "category": "keyword"},

    # 组织、安全、招聘和项目交付。
    {"query": "研发任务开始前需要先明确什么？", "expected": ["问题背景", "输入输出", "验收标准"], "sources": ["demo_company_workflow.txt"], "category": "multi_fact"},
    {"query": "项目里应该怎样保存 API Key？", "expected": ["环境变量", "密钥管理机制"], "sources": ["demo_company_security.txt", "demo_deployment_config.txt"], "category": "paraphrase"},
    {"query": "测试部门主要负责哪些工作？", "expected": ["测试用例", "缺陷复现", "回归验证"], "sources": ["demo_company_departments.txt"], "category": "multi_fact"},
    {"query": "星河科技主要开发哪些类型的软件？", "expected": ["企业内部管理系统", "数据分析工具", "智能问答应用"], "sources": ["demo_company_overview.txt"], "category": "multi_fact"},
    {"query": "后端实习生需要掌握哪些基础？", "expected": ["Python 或 Java", "HTTP 请求与响应", "关系型数据库"], "sources": ["demo_recruitment_backend.txt"], "category": "multi_fact"},
    {"query": "Agent 方向面试可能会问哪些工程问题？", "expected": ["工具执行失败", "消息历史", "检索结果"], "sources": ["demo_recruitment_interview.txt"], "category": "multi_fact"},
    {"query": "简历中可以随意写一个性能提升百分比吗？", "expected": ["不要使用没有实际测试依据"], "sources": ["demo_recruitment_resume.txt"], "category": "paraphrase"},
    {"query": "RAG 项目验收应该记录哪些检索结果？", "expected": ["正确证据是否被检索", "无答案"], "sources": ["demo_recruitment_evaluation.txt"], "category": "paraphrase"},
    {"query": "Agent 项目日志还可以观测哪些耗时和用量？", "expected": ["检索耗时", "模型调用耗时", "token 使用量"], "sources": ["demo_observability_logs.txt"], "category": "multi_fact"},

    # 无答案样本：语义相近但知识库没有给出所问事实。
    {"query": "公司老板的姓氏是什么？", "expected": None, "category": "no_answer"},
    {"query": "公司的具体月薪范围是多少？", "expected": None, "category": "no_answer"},
    {"query": "公司是哪一年成立的？", "expected": None, "category": "no_answer"},
    {"query": "公司总部的街道和门牌号是什么？", "expected": None, "category": "no_answer"},
    {"query": "公司目前有多少名员工？", "expected": None, "category": "no_answer"},
    {"query": "公司的产品今年收入是多少？", "expected": None, "category": "no_answer"},
    {"query": "这家公司的正式法定名称是什么？", "expected": None, "category": "no_answer"},
    {"query": "实习岗位每周需要到岗几天？", "expected": None, "category": "no_answer"},
    {"query": "公司是否提供餐补和住房补贴？", "expected": None, "category": "no_answer"},
    {"query": "公司老板的手机号码是多少？", "expected": None, "category": "no_answer"},
]
