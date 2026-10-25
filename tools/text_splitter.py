"""兼容层：分块逻辑已迁移到 `rag/splitter.py`。

保留这个文件是为了不让旧代码 `from tools.text_splitter import split_text` 直接报错。
新代码请从 `rag.splitter` 导入——分块属于检索层，不属于工具层。
"""

from rag.splitter import split_by_paragraph, split_text

__all__ = ["split_text", "split_by_paragraph"]
