# 定义检索评估问题及标准答案
#
# expected 为 None 表示“知识库里没有答案”，用来测误召回/拒答；
# 其余情况要检查所有关键词是否都出现在召回内容里。
#
# 关键词的写法有讲究：必须能落在**同一个合理片段**里。
# 比如问“实习岗位需要哪些技术”写 ["Python", "FastAPI", "RAG", "LLM API"]，
# 这四个词分散在三篇文档、四个片段里，任何 top_k 都不可能同时覆盖，
# 那测的就不是检索质量，而是数据集的错。

EVALUATION_CASES = [
    {
        "query": "这个实习岗位需要掌握哪些技术？",
        "expected": ["Python"],
    },
    {
        "query": "公司在哪里？",
        "expected": ["深圳市南山区"],
    },
    {
        "query": "工具调用的参数 Schema 是干什么的？",
        "expected": ["Schema"],
    },
    {
        "query": "MySQL 里怎么保存会话历史？",
        "expected": ["session_id"],
    },
    {
        "query": "余弦相似度是怎么计算 Top-K 的？",
        "expected": ["余弦相似度"],
    },
    # ---------- 无答案，应该拒答 ----------
    {"query": "今天股市大盘怎么样？", "expected": None},
    # 已知误召回：语义上确实贴近“公司介绍”，靠阈值拦不干净，
    # 最终由模型层的拒答提示词兜住
    {"query": "公司的股票代码是多少？", "expected": None},
    {"query": "明天北京天气如何？", "expected": None},
]
