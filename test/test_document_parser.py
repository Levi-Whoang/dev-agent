"""文档解析的大小、页数、字符数与独立进程超时检查。"""

import io
import zipfile

import pytest

from modules.document_parser import (
    DEFAULT_MAX_ARCHIVE_ENTRIES,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_UNCOMPRESSED_BYTES,
    DocumentLimitError,
    DocumentParseTimeoutError,
    ParserLimits,
    parse_document,
)


# 构造用于测试的 Office 压缩包。
def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return buffer.getvalue()


# 构造包含指定页数的空白 PDF。
def _pdf_bytes(page_count: int) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# 构造最小 Word 文档。
def _docx_bytes(text: str) -> bytes:
    from docx import Document

    document = Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# 构造最小 Excel 工作簿。
def _xlsx_bytes(text: str) -> bytes:
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active["A1"] = text
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


# 构造最小 PowerPoint 演示文稿。
def _pptx_bytes(text: str) -> bytes:
    from pptx import Presentation

    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.shapes.add_textbox(0, 0, 800, 200).text_frame.text = text
    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


# 普通文本通过独立进程解析，中文内容保持不变。
def test_parse_document_runs_in_isolated_process():
    parsed = parse_document("demo.txt", "第一段\n\n\n第二段".encode("utf-8"))

    assert parsed.name == "demo.txt"
    assert parsed.file_format == "txt"
    assert parsed.text == "第一段\n\n第二段"


# Word 文档正文能被提取。
def test_parse_docx_extracts_text():
    parsed = parse_document("demo.docx", _docx_bytes("合同金额：100 元"))

    assert parsed.file_format == "docx"
    assert "合同金额" in parsed.text


# Excel 单元格文本能被提取。
def test_parse_xlsx_extracts_cells():
    parsed = parse_document("demo.xlsx", _xlsx_bytes("服务器预算 12000"))

    assert parsed.file_format == "xlsx"
    assert "服务器预算" in parsed.text


# PowerPoint 文本框能被提取。
def test_parse_pptx_extracts_text():
    parsed = parse_document("demo.pptx", _pptx_bytes("第一页要点"))

    assert parsed.file_format == "pptx"
    assert "第一页要点" in parsed.text


# 文本超过字符数上限时拒绝解析。
def test_document_text_char_limit():
    limits = ParserLimits(max_chars=5)

    with pytest.raises(DocumentLimitError, match="字符限制"):
        parse_document("demo.txt", "123456".encode(), limits=limits)


# PDF 页数超过上限时拒绝解析。
def test_document_pdf_page_limit():
    limits = ParserLimits(max_pages=2)

    with pytest.raises(DocumentLimitError, match="页数超过"):
        parse_document("demo.pdf", _pdf_bytes(3), limits=limits)


# Office 压缩包条目数超过上限时拒绝解析。
def test_document_archive_entry_limit():
    content = _zip_bytes([("a.txt", b"a"), ("b.txt", b"b"), ("c.txt", b"c")])
    limits = ParserLimits(max_archive_entries=2)

    with pytest.raises(DocumentLimitError, match="条目数超过"):
        parse_document("demo.docx", content, limits=limits)


# Office 压缩包声明解压后总大小超过上限时拒绝解析。
def test_document_archive_total_size_limit():
    content = _zip_bytes([("big.txt", b"x" * 4096)])
    limits = ParserLimits(max_uncompressed_bytes=1024)

    with pytest.raises(DocumentLimitError, match="总大小超过"):
        parse_document("demo.xlsx", content, limits=limits)


# 解析进程超时会被终止并返回明确异常。
def test_document_parse_timeout(monkeypatch):
    monkeypatch.setenv("DOCUMENT_PARSE_TIMEOUT_SECONDS", "0.01")

    with pytest.raises(DocumentParseTimeoutError, match="解析超过"):
        parse_document("demo.txt", b"hello")


# 环境变量可覆盖限制，非法值回落到默认值。
def test_parser_limits_from_env(monkeypatch):
    monkeypatch.setenv("DOCUMENT_MAX_ARCHIVE_ENTRIES", "12")
    monkeypatch.setenv("DOCUMENT_MAX_UNCOMPRESSED_BYTES", "2048")
    monkeypatch.setenv("DOCUMENT_MAX_PAGES", "8")
    monkeypatch.setenv("DOCUMENT_MAX_CHARS", "99")
    monkeypatch.setenv("DOCUMENT_PARSE_TIMEOUT_SECONDS", "1.5")

    limits = ParserLimits.from_env()

    assert limits.max_archive_entries == 12
    assert limits.max_uncompressed_bytes == 2048
    assert limits.max_pages == 8
    assert limits.max_chars == 99
    assert limits.timeout_seconds == 1.5


# 非法数值使用默认值，不会让配置错误阻断上传。
def test_parser_limits_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("DOCUMENT_MAX_ARCHIVE_ENTRIES", "not-a-number")
    monkeypatch.setenv("DOCUMENT_MAX_UNCOMPRESSED_BYTES", "-1")
    monkeypatch.setenv("DOCUMENT_MAX_PAGES", "0")
    monkeypatch.setenv("DOCUMENT_MAX_CHARS", "")
    monkeypatch.setenv("DOCUMENT_PARSE_TIMEOUT_SECONDS", "bad")

    limits = ParserLimits.from_env()

    assert limits.max_archive_entries == DEFAULT_MAX_ARCHIVE_ENTRIES
    assert limits.max_uncompressed_bytes == DEFAULT_MAX_UNCOMPRESSED_BYTES
    assert limits.max_pages == DEFAULT_MAX_PAGES
    assert limits.max_chars == DEFAULT_MAX_CHARS
    assert limits.timeout_seconds == 30.0
