from __future__ import annotations

from pathlib import Path

from mini_claude_code.rag.chunking import chunk_document, scan_resources
from mini_claude_code.rag.config import RagConfig
from mini_claude_code.rag.types import SourceDocument


def make_document(path: str, text: str) -> SourceDocument:
    return SourceDocument(path=path, text=text, file_hash="a" * 64)


def make_config(project: Path) -> RagConfig:
    return RagConfig(
        project_dir=project,
        enabled=True,
        milvus_uri="http://127.0.0.1:19530",
        milvus_token="root:Milvus",
    )


def test_python_chunks_preserve_functions_methods_and_lines() -> None:
    text = (
        "MODULE_VALUE = 1\n"
        "\n"
        "class Greeter:\n"
        "    language = 'zh'\n"
        "\n"
        "    def hello(self, name):\n"
        "        return f'你好 {name}'\n"
        "\n"
        "def top_level():\n"
        "    return 42\n"
    )
    chunks = chunk_document(make_document("sample.py", text))
    by_symbol = {chunk.symbol: chunk for chunk in chunks if chunk.symbol}

    assert "Greeter" in by_symbol
    assert "Greeter.hello" in by_symbol
    assert "top_level" in by_symbol
    assert by_symbol["Greeter.hello"].start_line == 6
    assert by_symbol["Greeter.hello"].end_line == 7
    assert by_symbol["top_level"].start_line == 9
    assert all(len(chunk.text) <= 3_000 for chunk in chunks)


def test_markdown_chunks_use_heading_symbols_and_line_numbers() -> None:
    text = "intro\n\n# 安装\n第一步\n\n## 配置\n设置密钥\n"
    chunks = chunk_document(make_document("guide.md", text))
    symbols = {chunk.symbol: chunk for chunk in chunks}

    assert symbols["安装"].start_line == 3
    assert symbols["配置"].start_line == 6
    assert "设置密钥" in symbols["配置"].text


def test_plain_text_windows_overlap() -> None:
    text = "".join(str(index % 10) for index in range(3_500))
    chunks = chunk_document(make_document("notes.txt", text))

    assert len(chunks) == 3
    assert chunks[0].text[-200:] == chunks[1].text[:200]
    assert chunks[1].text[-200:] == chunks[2].text[:200]
    assert all(len(chunk.text) <= 1_500 for chunk in chunks)


def test_scan_filters_files_and_decodes_gb18030(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    resources.mkdir()
    (resources / "ok.txt").write_text("正常文本", encoding="utf-8")
    (resources / "legacy.md").write_bytes("中文旧编码".encode("gb18030"))
    (resources / ".hidden.txt").write_text("secret", encoding="utf-8")
    (resources / "binary.py").write_bytes(b"abc\x00def")
    (resources / "large.txt").write_bytes(b"x" * (1024 * 1024 + 1))
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        (resources / "escape.txt").symlink_to(outside)
    except OSError:
        pass

    result = scan_resources(make_config(tmp_path))

    assert [document.path for document in result.documents] == [
        "resources/legacy.md",
        "resources/ok.txt",
    ]
    assert result.documents[0].text == "中文旧编码"
    assert result.skipped >= 3
