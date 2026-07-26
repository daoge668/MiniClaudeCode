from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mini_claude_code.config import ConfigurationError, Settings


PROJECT = Path(__file__).resolve().parents[1]


def test_import_has_no_runtime_side_effects(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("MODEL_ID", None)
    environment["PYTHONPATH"] = str(PROJECT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import mini_claude_code; "
                "print(mini_claude_code.__version__)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1.0.0"
    assert list(tmp_path.iterdir()) == []


def test_settings_loads_project_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "MODEL_ID=from-file\nFALLBACK_MODEL_ID=backup\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MODEL_ID", raising=False)
    loaded = Settings.from_env(tmp_path)
    assert loaded.model_id == "from-file"
    assert loaded.fallback_model_id == "backup"
    assert loaded.workdir == tmp_path.resolve()


def test_settings_requires_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MODEL_ID", raising=False)
    with pytest.raises(ConfigurationError, match="MODEL_ID"):
        Settings.from_env(tmp_path)


def test_cli_help_does_not_require_configuration(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.pop("MODEL_ID", None)
    result = subprocess.run(
        [sys.executable, str(PROJECT / "code.py"), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0
    assert "mini-claude-code" in result.stdout
    assert "--workdir" in result.stdout
