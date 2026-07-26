"""Safe corpus scanning and structure-aware text chunking."""

from __future__ import annotations

import ast
import hashlib
import os
import re
from pathlib import Path

from .config import SUPPORTED_NAMES, SUPPORTED_SUFFIXES, RagConfig
from .types import DocumentChunk, ScanResult, SourceDocument

TARGET_CHARS = 1_500
OVERLAP_CHARS = 200
MAX_CHARS = 3_000
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _is_supported(path: Path) -> bool:
    return (
        path.suffix.lower() in SUPPORTED_SUFFIXES
        or path.name.lower() in SUPPORTED_NAMES
    )


def _is_hidden(relative: Path) -> bool:
    return any(part.startswith(".") for part in relative.parts)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def scan_resources(config: RagConfig) -> ScanResult:
    """Read eligible files without following symlinks outside the corpus."""
    result = ScanResult()
    root = config.resources_dir
    if not root.exists():
        return result
    if root.is_symlink():
        result.warnings.append("resources/ is a symlink; corpus scan refused")
        return result

    root_real = root.resolve()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        safe_dirs: list[str] = []
        for dirname in dirnames:
            candidate = current / dirname
            relative = candidate.relative_to(root)
            if _is_hidden(relative) or candidate.is_symlink():
                result.skipped += 1
                continue
            try:
                if not _inside(candidate.resolve(), root_real):
                    result.skipped += 1
                    continue
            except OSError:
                result.skipped += 1
                continue
            safe_dirs.append(dirname)
        dirnames[:] = safe_dirs

        for filename in filenames:
            path = current / filename
            relative_path = path.relative_to(root)
            relative = (Path("resources") / relative_path).as_posix()
            if (
                _is_hidden(relative_path)
                or path.is_symlink()
                or not _is_supported(path)
            ):
                result.skipped += 1
                continue
            try:
                resolved = path.resolve(strict=True)
                if not _inside(resolved, root_real):
                    result.skipped += 1
                    continue
                if path.stat().st_size > config.max_file_bytes:
                    result.skipped += 1
                    continue
                raw = path.read_bytes()
            except OSError as exc:
                result.skipped += 1
                result.failed_paths.add(relative)
                result.warnings.append(f"{relative}: read failed ({exc})")
                continue
            if b"\x00" in raw[:8192]:
                result.skipped += 1
                continue
            text: str | None = None
            for encoding in ("utf-8-sig", "gb18030"):
                try:
                    text = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if text is None:
                result.skipped += 1
                result.failed_paths.add(relative)
                result.warnings.append(f"{relative}: unsupported text encoding")
                continue
            result.documents.append(
                SourceDocument(
                    path=relative,
                    text=text,
                    file_hash=hashlib.sha256(raw).hexdigest(),
                )
            )
    result.documents.sort(key=lambda item: item.path)
    return result


def _window_span(
    text: str,
    source_path: str,
    file_hash: str,
    base_line: int,
    symbol: str,
) -> list[DocumentChunk]:
    if not text.strip():
        return []
    chunks: list[DocumentChunk] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + TARGET_CHARS)
        if end < length:
            newline = text.rfind("\n", start + TARGET_CHARS // 2, end)
            if newline > start:
                end = newline + 1
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        trailing_text = raw.rstrip()
        if trailing_text:
            content_start = start + leading
            content_end = start + len(trailing_text)
            content = text[content_start:content_end]
            start_line = base_line + text.count("\n", 0, content_start)
            end_line = base_line + text.count("\n", 0, content_end)
            identity = (
                f"{source_path}\0{file_hash}\0{start_line}\0"
                f"{end_line}\0{symbol}\0{content}"
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest(),
                    text=content[:MAX_CHARS],
                    source_path=source_path,
                    start_line=start_line,
                    end_line=end_line,
                    symbol=symbol,
                    file_hash=file_hash,
                )
            )
        if end >= length:
            break
        start = max(start + 1, end - OVERLAP_CHARS)
        while start < length and text[start] in "\r\n":
            start += 1
    return chunks


def _span_text(lines: list[str], start_line: int, end_line: int) -> str:
    return "".join(lines[start_line - 1 : end_line])


def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    starts = [getattr(node, "lineno", 1)]
    starts.extend(getattr(item, "lineno", starts[0]) for item in decorators)
    return min(starts)


def _python_chunks(document: SourceDocument) -> list[DocumentChunk]:
    try:
        tree = ast.parse(document.text)
    except (SyntaxError, ValueError):
        return _window_span(
            document.text,
            document.path,
            document.file_hash,
            1,
            "",
        )

    lines = document.text.splitlines(keepends=True)
    total_lines = max(1, len(lines))
    spans: list[tuple[int, int, str]] = []

    def add_function(node: ast.AST, prefix: str = "") -> None:
        name = str(getattr(node, "name", ""))
        symbol = f"{prefix}.{name}" if prefix else name
        spans.append(
            (
                _node_start(node),
                int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
                symbol,
            )
        )

    def add_class(node: ast.ClassDef, prefix: str = "") -> None:
        class_name = f"{prefix}.{node.name}" if prefix else node.name
        structural: list[tuple[int, int]] = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add_function(child, class_name)
                structural.append(
                    (
                        _node_start(child),
                        int(getattr(child, "end_lineno", child.lineno)),
                    )
                )
            elif isinstance(child, ast.ClassDef):
                add_class(child, class_name)
                structural.append(
                    (
                        _node_start(child),
                        int(getattr(child, "end_lineno", child.lineno)),
                    )
                )
        cursor = _node_start(node)
        class_end = int(getattr(node, "end_lineno", node.lineno))
        for child_start, child_end in sorted(structural):
            if cursor < child_start:
                spans.append((cursor, child_start - 1, class_name))
            cursor = max(cursor, child_end + 1)
        if cursor <= class_end:
            spans.append((cursor, class_end, class_name))

    top_level_structural: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_function(node)
            top_level_structural.append(
                (
                    _node_start(node),
                    int(getattr(node, "end_lineno", node.lineno)),
                )
            )
        elif isinstance(node, ast.ClassDef):
            add_class(node)
            top_level_structural.append(
                (
                    _node_start(node),
                    int(getattr(node, "end_lineno", node.lineno)),
                )
            )

    cursor = 1
    for start_line, end_line in sorted(top_level_structural):
        if cursor < start_line:
            spans.append((cursor, start_line - 1, ""))
        cursor = max(cursor, end_line + 1)
    if cursor <= total_lines:
        spans.append((cursor, total_lines, ""))

    chunks: list[DocumentChunk] = []
    for start_line, end_line, symbol in sorted(
        spans, key=lambda item: (item[0], item[1], item[2])
    ):
        chunks.extend(
            _window_span(
                _span_text(lines, start_line, end_line),
                document.path,
                document.file_hash,
                start_line,
                symbol,
            )
        )
    return chunks


def _markdown_chunks(document: SourceDocument) -> list[DocumentChunk]:
    lines = document.text.splitlines(keepends=True)
    headings: list[tuple[int, str]] = []
    for index, line in enumerate(lines, start=1):
        match = _MARKDOWN_HEADING.match(line.rstrip("\r\n"))
        if match:
            headings.append((index, match.group(2).strip()))
    if not headings:
        return _window_span(
            document.text,
            document.path,
            document.file_hash,
            1,
            "",
        )

    chunks: list[DocumentChunk] = []
    if headings[0][0] > 1:
        chunks.extend(
            _window_span(
                _span_text(lines, 1, headings[0][0] - 1),
                document.path,
                document.file_hash,
                1,
                "",
            )
        )
    for index, (start_line, title) in enumerate(headings):
        end_line = (
            headings[index + 1][0] - 1
            if index + 1 < len(headings)
            else len(lines)
        )
        chunks.extend(
            _window_span(
                _span_text(lines, start_line, end_line),
                document.path,
                document.file_hash,
                start_line,
                title,
            )
        )
    return chunks


def chunk_document(document: SourceDocument) -> list[DocumentChunk]:
    suffix = Path(document.path).suffix.lower()
    if suffix == ".py":
        return _python_chunks(document)
    if suffix in {".md", ".rst"}:
        return _markdown_chunks(document)
    return _window_span(
        document.text,
        document.path,
        document.file_hash,
        1,
        "",
    )
