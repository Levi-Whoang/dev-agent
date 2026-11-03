# RAG Evaluation

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m evaluation.retrieval
```

默认只读取 `demo_data/knowledge_base` 顶层公共 TXT，避免混入上传文档；可用 `--corpus` 指定公共语料、`--output` 指定结果目录。首次使用 embedding 模型需要下载，模型有缓存时可设置 `HF_HUB_OFFLINE=1` 离线运行，不需要 LLM 密钥或数据库。

使用相同的切分、模型、向量阈值、Top-K 与去重，比较纯向量策略重建（dense）和当前向量/BM25 加权融合（hybrid）。默认应用检索仍为 hybrid，没有用修改权重的方式伪装纯向量基线。纯向量模式不通过关键词通道补召回。

JSON/TXT 历史报告保存到 `evaluation/results`，最新 TXT 同时写入 `TXT/RAG检索对比评估结果.txt`。报告包括问题集/语料指纹、Git 提交及 dirty 状态、相关源文件指纹、模型修订与依赖环境、逐题结果、失败题目、分类指标和成对变化。TXT 给出指标定义和本次结论。

证据覆盖命中率基于标准词与允许来源，不能直接称为严格片段 Recall；来源 MRR 不代表证据正确，单片段证据 MRR 要求全部标准词在单片段内。无答案误召回以负例总数为分母。开发集的词面标注可能缺少同义词，也可能与语料版本不符；gold_audit 会列出异常，但不从主要指标中静默删除。当前体系评估检索，不包含生成答案质量、人工引用支持性或独立留出集。

运行指标及检索回归测试：`python -m pytest test/test_rag_evaluation.py test/test_knowledge.py -q`。


## 五项指标与公司业务评估集

新增 `evaluation/business_dataset.json`（business-eval-1.0）：20 道虚构星桥云业务问题，12 道可回答、8 道无答案；配套 5 份 `demo_data/knowledge_base/business_*.txt`。标注包含证据 ID、来源、完整引文和参考答案。运行前校验证据能否映射到当前分块；缺失时停止评估，避免悄悄改变分母。

```powershell
python -m evaluation.quality
python -m evaluation.quality --generate --generation-limit 6
```

第一条离线执行全部检索指标；第二条额外调用已配置的模型，按正负题交替抽样，6 题即 3 正例+3负例，每种模式生成6份答案并裁判，最多24次请求。`--generation-limit 20` 覆盖全部20题。需要 `.env` 中 LLM 配置，会产生模型调用费用；不连接 MySQL、不读取上传文档。

- Recall@K：召回证据 ID 数 / 全部标注证据 ID 数，按可回答问题宏平均。
- MRR：首个含正确标注证据的片段倒数排名，按可回答问题宏平均。
- Precision@K：含正确标注证据的片段数 / K，不足K的空位也占分母，分别报告全题及可回答题均值。
- Faithfulness：裁判认为有证据支持的原子事实数 / 有效判定的原子事实总数；纯拒答没有事实时为 N/A，同时报告失败数与判定陈述数。
- 无答案准确率：生成抽样负例中正确拒答数 / 抽样负例数；模型或裁判失败算失败。另报检索为空的实际拒答率和仅检索阶段的无答案准确率，不混淆三者。

Precision/Recall 相关性使用来源+完整证据引文规则，便于复核，但不能覆盖语义等价改写；复杂资料可扩展人工证据标注。忠实度/拒答由模型裁判，不是人工真值；生成与裁判目前使用同一配置模型，有同模型偏差。报告保留答案与原子判断供人工复查。该链路评估固定检索后生成，不评估 Agent 是否自主选择检索工具。

输出：`TXT/RAG五项指标评估结果.txt` 及 `evaluation/results/rag_quality_*.json/.txt`。旧80题检索对照报告继续保留；新资料改变语料版本，新业务评估不能与旧版本的绝对分数直接比较。
