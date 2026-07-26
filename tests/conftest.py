from __future__ import annotations

from pathlib import Path

import pytest

from mini_claude_code.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        project_dir=tmp_path,
        workdir=tmp_path,
        model_id="test-model",
        fallback_model_id="fallback-model",
        base_delay_ms=0,
    )
