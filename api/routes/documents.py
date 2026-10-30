"""知识库文档 HTTP 路由及请求校验。"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.api_response import success_response
from api.api_schemas import RenameDocumentRequest
from api.session_helpers import database_user_id
from auth.jwt_auth import get_current_user
from modules.document_parser import SUPPORTED_FORMATS
from services.documents import (
    DocumentNotFoundError,
    DocumentParseError,
    create_document,
    delete_document as delete_document_service,
    get_document as get_document_service,
    list_documents as list_documents_service,
    rename_document as rename_document_service,
    replace_document as replace_document_service,
)

router = APIRouter(prefix="/knowledge/documents", tags=["知识库文档"])
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024


# 读取上传文件并校验文件名、格式和大小。
async def _read_upload(upload: UploadFile) -> tuple[str, bytes]:
    """读取上传文件并校验文件名、格式和大小。"""
    filename = Path((upload.filename or "").replace("\\", "/")).name.strip()
    if not filename or len(filename) > 255:
        raise HTTPException(status_code=422, detail="文件名无效")
    if Path(filename).suffix.lower() not in SUPPORTED_FORMATS:
        formats = ", ".join(sorted(SUPPORTED_FORMATS))
        raise HTTPException(status_code=415, detail=f"文件格式不支持，可上传：{formats}")
    content = await upload.read(MAX_DOCUMENT_BYTES + 1)
    if not content:
        raise HTTPException(status_code=422, detail="不能上传空文件")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=413, detail="单个文件不能超过 20 MB")
    return filename, content


# 查询已认证用户的数据库主键。
def _owner_id(username: str) -> int:
    """获取当前登录用户的数据库主键。"""
    return database_user_id(username)


# 把文档未找到业务异常转换成 HTTP 404。
def _not_found(error: DocumentNotFoundError) -> HTTPException:
    """将文档服务的未找到异常转换为 HTTP 404。"""
    return HTTPException(status_code=404, detail="文档不存在")


# 把文档解析业务异常转换成 HTTP 422。
def _parse_error(error: DocumentParseError) -> HTTPException:
    """将文档服务的解析异常转换为 HTTP 422。"""
    return HTTPException(status_code=422, detail=str(error))


# 接收上传文件并调用文档服务创建知识库条目。
@router.post("", status_code=201)
async def upload_document(
    file: UploadFile = File(...), current_user: str = Depends(get_current_user)
):
    """上传文档并委托服务层完成解析、存储和元数据创建。"""
    filename, content = await _read_upload(file)
    try:
        result = create_document(_owner_id(current_user), filename, content)
    except DocumentParseError as error:
        raise _parse_error(error) from error
    return success_response(result, "文档上传并加入知识库")


# 返回当前用户的文档列表。
@router.get("")
def list_documents(current_user: str = Depends(get_current_user)):
    """列出当前登录用户上传的文档。"""
    return success_response(list_documents_service(_owner_id(current_user)))


# 返回指定文档的元数据。
@router.get("/{document_id}")
def get_document(document_id: str, current_user: str = Depends(get_current_user)):
    """读取当前用户文档的元数据。"""
    try:
        result = get_document_service(_owner_id(current_user), document_id)
    except DocumentNotFoundError as error:
        raise _not_found(error) from error
    return success_response(result)


# 接收新文件并替换指定文档的内容。
@router.put("/{document_id}")
async def replace_document(
    document_id: str,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    """替换当前用户文档内容并返回更新后的元数据。"""
    filename, content = await _read_upload(file)
    try:
        result = replace_document_service(
            _owner_id(current_user), document_id, filename, content
        )
    except DocumentNotFoundError as error:
        raise _not_found(error) from error
    except DocumentParseError as error:
        raise _parse_error(error) from error
    return success_response(result, "文档已更新")


# 校验并更新文档显示名称。
@router.patch("/{document_id}")
def rename_document(
    document_id: str,
    request: RenameDocumentRequest,
    current_user: str = Depends(get_current_user),
):
    """修改当前用户文档名称。"""
    try:
        result = rename_document_service(
            _owner_id(current_user), document_id, request.name
        )
    except DocumentNotFoundError as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return success_response(result, "文档名称已更新")


# 删除指定文档及服务层管理的文件内容。
@router.delete("/{document_id}")
def delete_document(document_id: str, current_user: str = Depends(get_current_user)):
    """删除当前用户的文档及相关文件。"""
    try:
        delete_document_service(_owner_id(current_user), document_id)
    except DocumentNotFoundError as error:
        raise _not_found(error) from error
    return success_response({"document_id": document_id}, "文档已删除")
