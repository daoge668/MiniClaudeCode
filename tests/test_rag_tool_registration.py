from __future__ import annotations

from mini_claude_code.agents import FOCUSED_TOOL_NAMES, TEAMMATE_TOOL_NAMES
from mini_claude_code.rag.tool import (
    SYSTEM_GUIDANCE,
    TOOL_NAME,
    definitions_for,
    handlers_for,
)
from mini_claude_code.tools import ToolRegistry


class StubService:
    available = True

    @staticmethod
    def search(query: str, top_k: int = 5) -> str:
        return query


def test_same_rag_tool_is_available_to_all_agent_types() -> None:
    service = StubService()
    registry = ToolRegistry.__new__(ToolRegistry)
    registry.rag = service

    assert definitions_for(service)[0]["name"] == TOOL_NAME
    assert handlers_for(service)[TOOL_NAME] == service.search
    assert TOOL_NAME in {
        definition["name"] for definition in registry.definitions
    }
    assert TOOL_NAME in FOCUSED_TOOL_NAMES
    assert TOOL_NAME in TEAMMATE_TOOL_NAMES


def test_prompt_guidance_requires_citations_and_treats_text_as_untrusted() -> None:
    assert "path:start_line-end_line" in SYSTEM_GUIDANCE
    assert "untrusted data" in SYSTEM_GUIDANCE


def test_unavailable_service_registers_nothing() -> None:
    service = StubService()
    service.available = False

    assert definitions_for(service) == []
    assert handlers_for(service) == {}
