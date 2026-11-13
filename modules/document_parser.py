"""将支持的上传文件统一解析为纯文本。"""

import csv
from dataclasses import dataclass
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


SUPPORTED_FORMATS = {".txt", ".md", ".csv", ".json", ".html", ".htm", ".pdf", ".docx", ".xlsx", ".pptx"}
_ARCHIVE_FORMATS = {".docx", ".xlsx", ".pptx"}
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MAX_ARCHIVE_ENTRIES = 2000
DEFAULT_MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
DEFAULT_MAX_PAGES = 300
DEFAULT_MAX_CHARS = 2_000_000
DEFAULT_PARSE_TIMEOUT_SECONDS = 30.0


class DocumentLimitError(ValueError):
    """文档超过解压大小、条目数、页数或字符数限制。"""


class DocumentParseTimeoutError(ValueError):
    """文档解析进程超时。"""


@dataclass(frozen=True)
class ParserLimits:
    """解析限制；默认值可通过环境变量覆盖，避免异常文件耗尽资源。"""

    max_archive_entries: int = DEFAULT_MAX_ARCHIVE_ENTRIES
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES
    max_pages: int = DEFAULT_MAX_PAGES
    max_chars: int = DEFAULT_MAX_CHARS
    timeout_seconds: float = DEFAULT_PARSE_TIMEOUT_SECONDS

    # 从环境变量读取限制，非法值回落到默认值。
    @classmethod
    def from_env(cls) -> "ParserLimits":
        return cls(
            max_archive_entries=_env_number(
                "DOCUMENT_MAX_ARCHIVE_ENTRIES", DEFAULT_MAX_ARCHIVE_ENTRIES, int, 1
            ),
            max_uncompressed_bytes=_env_number(
                "DOCUMENT_MAX_UNCOMPRESSED_BYTES", DEFAULT_MAX_UNCOMPRESSED_BYTES, int, 1024
            ),
            max_pages=_env_number("DOCUMENT_MAX_PAGES", DEFAULT_MAX_PAGES, int, 1),
            max_chars=_env_number("DOCUMENT_MAX_CHARS", DEFAULT_MAX_CHARS, int, 1),
            timeout_seconds=_env_number(
                "DOCUMENT_PARSE_TIMEOUT_SECONDS", DEFAULT_PARSE_TIMEOUT_SECONDS, float, 0.01
            ),
        )


# 读取数值环境变量，缺失、非法或低于最小值时使用默认值。
def _env_number(name: str, default, cast, minimum):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        return default
    if value < minimum:
        return default
    return value


@dataclass(frozen=True)
class ParsedDocument:
    """文档解析结果，供上传入库流程统一消费。"""

    name: str
    file_format: str
    text: str


class _TextOnlyHTMLParser(HTMLParser):
    """提取 HTML 可见文本，并跳过脚本和样式内容。"""

    # 初始化 HTML 可见文本收集器。
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    # 处理 HTML 开始标签并跳过脚本样式内容。
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif tag.lower() in {"p", "div", "li", "br", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    # 处理 HTML 结束标签并维护忽略区域状态。
    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag.lower() in {"p", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    # 收集 HTML 中可见的文本内容。
    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self.parts.append(data.strip())


# 统一换行并清理解析文本中的空白字符。
def _normalize(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", text)).strip()


# 校验 Office 压缩包的条目数和解压后总大小，防止压缩炸弹。
def _validate_archive(content: bytes, limits: ParserLimits) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            if len(infos) > limits.max_archive_entries:
                raise DocumentLimitError(f"压缩包条目数超过 {limits.max_archive_entries} 个限制")
            total_size = sum(info.file_size for info in infos)
            if total_size > limits.max_uncompressed_bytes:
                raise DocumentLimitError(
                    f"压缩包解压后总大小超过 {_format_size(limits.max_uncompressed_bytes)} 限制"
                )
    except zipfile.BadZipFile as error:
        raise ValueError("文件不是有效的 Office 压缩文档") from error


# 把字节数格式化为易读的大小，用于限制提示。
def _format_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):g} MB"
    if size >= 1024:
        return f"{size / 1024:g} KB"
    return f"{size} B"


# 校验页数上限，PDF 和 PPTX 共用。
def _check_page_limit(page_count: int, limits: ParserLimits) -> None:
    if page_count > limits.max_pages:
        raise DocumentLimitError(f"文档页数超过 {limits.max_pages} 页限制")


# 在当前进程内完成实际解析；只由独立解析进程调用。
def _parse_document_bytes(filename: str, content: bytes, limits: ParserLimits) -> ParsedDocument:
    """解析字节内容并应用大小、页数和字符数限制。"""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"暂不支持 {suffix or '无扩展名'} 格式")
    if suffix in _ARCHIVE_FORMATS:
        _validate_archive(content, limits)
    try:
        if suffix in {".txt", ".md"}:
            text = content.decode("utf-8-sig")
        elif suffix == ".csv":
            rows = csv.reader(io.StringIO(content.decode("utf-8-sig")))
            text = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
        elif suffix == ".json":
            text = json.dumps(json.loads(content.decode("utf-8-sig")), ensure_ascii=False, indent=2)
        elif suffix in {".html", ".htm"}:
            parser = _TextOnlyHTMLParser()
            parser.feed(content.decode("utf-8-sig"))
            text = "\n".join(parser.parts)
        elif suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            _check_page_limit(len(reader.pages), limits)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == ".docx":
            from docx import Document
            document = Document(io.BytesIO(content))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            for table in document.tables:
                text += "\n" + "\n".join(" | ".join(cell.text for cell in row.cells) for row in table.rows)
        elif suffix == ".xlsx":
            from openpyxl import load_workbook
            workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
            try:
                text = "\n".join(
                    f"工作表：{sheet.title}\n" + "\n".join(
                        " | ".join("" if value is None else str(value) for value in row)
                        for row in sheet.iter_rows(values_only=True)
                    )
                    for sheet in workbook.worksheets
                )
            finally:
                workbook.close()
        else:
            from pptx import Presentation
            presentation = Presentation(io.BytesIO(content))
            _check_page_limit(len(presentation.slides), limits)
            text = "\n".join(
                shape.text for slide in presentation.slides for shape in slide.shapes if shape.has_text_frame
            )
    except DocumentLimitError:
        raise
    except Exception as error:
        raise ValueError("文件内容无法解析，请确认文件未损坏且文本编码为 UTF-8") from error

    normalized = _normalize(text)
    if not normalized:
        raise ValueError("文档中没有可检索的文本内容")
    if len(normalized) > limits.max_chars:
        raise DocumentLimitError(f"文档文本超过 {limits.max_chars:,} 字符限制")
    return ParsedDocument(name=Path(filename).name, file_format=suffix.lstrip("."), text=normalized)


# 独立解析进程入口：读取输入文件，把结果或错误写入 JSON 输出文件。
def _worker_main(input_path: str, output_path: str, filename: str) -> None:
    try:
        content = Path(input_path).read_bytes()
        parsed = _parse_document_bytes(filename, content, ParserLimits.from_env())
        payload = {"name": parsed.name, "file_format": parsed.file_format, "text": parsed.text}
    except DocumentLimitError as error:
        payload = {"error": str(error), "error_type": "limit"}
    except Exception as error:
        payload = {"error": str(error) or "文档解析失败"}
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# 把解析限制写入子进程环境变量，保证父子进程使用同一组限制。
def _limits_env(limits: ParserLimits) -> dict[str, str]:
    return {
        "DOCUMENT_MAX_ARCHIVE_ENTRIES": str(limits.max_archive_entries),
        "DOCUMENT_MAX_UNCOMPRESSED_BYTES": str(limits.max_uncompressed_bytes),
        "DOCUMENT_MAX_PAGES": str(limits.max_pages),
        "DOCUMENT_MAX_CHARS": str(limits.max_chars),
        "DOCUMENT_PARSE_TIMEOUT_SECONDS": str(limits.timeout_seconds),
    }


# 在独立进程中解析文档；超时会终止子进程并抛出明确异常。
def parse_document(filename: str, content: bytes, limits: ParserLimits | None = None) -> ParsedDocument:
    """按扩展名解析文档；解析在独立进程执行并受超时约束。"""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"暂不支持 {suffix or '无扩展名'} 格式")
    limits = limits or ParserLimits.from_env()
    # 清理失败（如杀毒软件短暂占用）不应掩盖已经完成的解析结果。
    with tempfile.TemporaryDirectory(prefix="document-parse-", ignore_cleanup_errors=True) as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        output_path = Path(temp_dir) / "result.json"
        input_path.write_bytes(content)
        env = os.environ.copy()
        env.update(_limits_env(limits))
        python_path = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(_PROJECT_ROOT) + (os.pathsep + python_path if python_path else "")
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "modules.document_parser", str(input_path), str(output_path), filename],
                cwd=str(_PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(0.01, limits.timeout_seconds),
            )
        except subprocess.TimeoutExpired as error:
            raise DocumentParseTimeoutError(
                f"文档解析超过 {limits.timeout_seconds:g} 秒，已终止"
            ) from error
        if completed.returncode != 0:
            raise ValueError("文档解析进程异常退出")
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("文档解析结果读取失败") from error
        if not isinstance(payload, dict):
            raise ValueError("文档解析结果格式无效")
        if "error" in payload:
            message = str(payload["error"])
            if payload.get("error_type") == "limit":
                raise DocumentLimitError(message)
            raise ValueError(message)
        return ParsedDocument(
            name=str(payload.get("name") or Path(filename).name),
            file_format=str(payload.get("file_format") or suffix.lstrip(".")),
            text=str(payload.get("text") or ""),
        )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(2)
    _worker_main(sys.argv[1], sys.argv[2], sys.argv[3])
