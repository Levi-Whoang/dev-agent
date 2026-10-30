"""知识库文档的解析、文件存储和数据库业务逻辑。"""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from modules.database import session_scope
from modules.document_parser import ParsedDocument, parse_document
from modules.models import KnowledgeDocument
from tools.knowledge import KNOWLEDGE_DIR, refresh_knowledge_cache

_FILES_DIR = KNOWLEDGE_DIR / ".originals"
_TEXT_DIR = KNOWLEDGE_DIR / "documents"


class DocumentNotFoundError(Exception):
    """当前用户没有找到指定文档时抛出的业务异常。"""


class DocumentParseError(Exception):
    """上传文件无法解析或不包含文本时抛出的业务异常。"""


# 按文档 ID 和所有者 ID 构造文档查询条件。
def _document_query(document_id: str, owner_user_id: int):
    """按文档 ID 和所有者 ID 构造文档查询条件。"""
    return select(KnowledgeDocument).where(
        KnowledgeDocument.document_id == document_id,
        KnowledgeDocument.owner_user_id == owner_user_id,
    )


# 将文档 ORM 记录转换为服务层返回的数据字典。
def _serialize(document: KnowledgeDocument) -> dict:
    """将文档 ORM 记录转换为服务层返回的数据字典。"""
    return {
        "document_id": document.document_id,
        "name": document.name,
        "format": document.file_format,
        "uploaded_by": document.owner_user_id,
        "uploaded_at": document.created_at.isoformat() if document.created_at else None,
        "modified_at": document.modified_at.isoformat() if document.modified_at else None,
    }


# 生成原始文件与解析文本的存储路径。
def _paths(owner_user_id: int, document_id: str, filename: str) -> tuple[Path, Path]:
    """生成原始文件与解析文本的存储路径。"""
    safe_name = Path(filename).name.replace("/", "_").replace("\\", "_")
    original = _FILES_DIR / str(owner_user_id) / f"{document_id}{Path(safe_name).suffix.lower()}"
    extracted = _TEXT_DIR / str(owner_user_id) / f"{document_id}--{safe_name}.txt"
    return original, extracted


# 将知识库内的文件路径转换为相对路径。
def _relative(path: Path) -> str:
    """将知识库内的文件路径转换为相对路径。"""
    return str(path.resolve().relative_to(KNOWLEDGE_DIR.resolve()))


# 安全删除知识库目录中的指定文件。
def _remove_stored_file(relative_path: str) -> None:
    """安全删除知识库目录中的指定文件。"""
    path = (KNOWLEDGE_DIR / relative_path).resolve()
    if path.is_relative_to(KNOWLEDGE_DIR.resolve()):
        path.unlink(missing_ok=True)


# 解析文档并保存原始文件及用于检索的纯文本。
def _parse_and_save(
    owner_user_id: int, document_id: str, filename: str, content: bytes
) -> tuple[ParsedDocument, str, str]:
    """解析文档并保存原始文件及用于检索的纯文本。"""
    try:
        parsed = parse_document(filename, content)
    except ValueError as error:
        raise DocumentParseError(str(error)) from error
    original, extracted = _paths(owner_user_id, document_id, filename)
    original.parent.mkdir(parents=True, exist_ok=True)
    extracted.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(content)
    extracted.write_text(parsed.text, encoding="utf-8")
    return parsed, _relative(original), _relative(extracted)


# 创建用户文档记录并将解析文本加入知识库。
def create_document(owner_user_id: int, filename: str, content: bytes) -> dict:
    """创建用户文档记录并将解析文本加入知识库。"""
    document_id = str(uuid4())
    parsed, original_path, extracted_path = _parse_and_save(
        owner_user_id, document_id, filename, content
    )
    try:
        with session_scope() as db:
            document = KnowledgeDocument(
                document_id=document_id,
                owner_user_id=owner_user_id,
                name=parsed.name,
                file_format=parsed.file_format,
                original_path=original_path,
                extracted_path=extracted_path,
            )
            db.add(document)
            db.flush()
            result = _serialize(document)
    except Exception:
        _remove_stored_file(original_path)
        _remove_stored_file(extracted_path)
        raise
    refresh_knowledge_cache()
    return result


# 按上传用户列出文档并转换为响应数据。
def list_documents(owner_user_id: int) -> list[dict]:
    """按上传用户列出文档并转换为响应数据。"""
    with session_scope() as db:
        documents = db.scalars(
            select(KnowledgeDocument)
            .where(KnowledgeDocument.owner_user_id == owner_user_id)
            .order_by(KnowledgeDocument.created_at.desc())
        ).all()
        return [_serialize(document) for document in documents]


# 读取指定用户文档元数据，未找到时抛出业务异常。
def get_document(owner_user_id: int, document_id: str) -> dict:
    """读取指定用户文档元数据，未找到时抛出业务异常。"""
    with session_scope() as db:
        document = db.scalar(_document_query(document_id, owner_user_id))
        if document is None:
            raise DocumentNotFoundError(document_id)
        return _serialize(document)


# 替换当前用户文档内容、解析结果和修改时间。
def replace_document(owner_user_id: int, document_id: str, filename: str, content: bytes) -> dict:
    """替换当前用户文档内容、解析结果和修改时间。"""
    with session_scope() as db:
        document = db.scalar(_document_query(document_id, owner_user_id))
        if document is None:
            raise DocumentNotFoundError(document_id)
        old_original, old_extracted = document.original_path, document.extracted_path
        parsed, original_path, extracted_path = _parse_and_save(
            owner_user_id, document_id, filename, content
        )
        document.name = parsed.name
        document.file_format = parsed.file_format
        document.original_path = original_path
        document.extracted_path = extracted_path
        document.modified_at = datetime.utcnow()
        result = _serialize(document)
    if old_original != original_path:
        _remove_stored_file(old_original)
    if old_extracted != extracted_path:
        _remove_stored_file(old_extracted)
    refresh_knowledge_cache()
    return result


# 修改文档显示名称并同步更新检索文本来源名称。
def rename_document(owner_user_id: int, document_id: str, name: str) -> dict:
    """修改文档显示名称并同步更新检索文本来源名称。"""
    name = Path(name.strip().replace("\\", "/")).name
    if not name:
        raise ValueError("文档名称不能为空")
    with session_scope() as db:
        document = db.scalar(_document_query(document_id, owner_user_id))
        if document is None:
            raise DocumentNotFoundError(document_id)
        old_path = (KNOWLEDGE_DIR / document.extracted_path).resolve()
        new_path = old_path.with_name(f"{document_id}--{name}.txt")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if old_path.exists():
            old_path.replace(new_path)
        document.name = name
        document.extracted_path = _relative(new_path)
        document.modified_at = datetime.utcnow()
        result = _serialize(document)
    refresh_knowledge_cache()
    return result


# 删除当前用户的文档记录、原文件和检索文本。
def delete_document(owner_user_id: int, document_id: str) -> None:
    """删除当前用户的文档记录、原文件和检索文本。"""
    with session_scope() as db:
        document = db.scalar(_document_query(document_id, owner_user_id))
        if document is None:
            raise DocumentNotFoundError(document_id)
        original_path, extracted_path = document.original_path, document.extracted_path
        db.delete(document)
    _remove_stored_file(original_path)
    _remove_stored_file(extracted_path)
    refresh_knowledge_cache()
