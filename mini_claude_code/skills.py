"""Skill discovery and system-prompt assembly."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

from .config import Settings


class SkillRegistry:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self._skills: dict[str, dict] = {}

    @staticmethod
    def parse_frontmatter(text: str) -> tuple[dict, str]:
        if not text.startswith("---"):
            return {}, text
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text
        try:
            metadata = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            metadata = {}
        return metadata, parts[2].strip()

    def scan(self) -> None:
        self._skills.clear()
        if not self.skills_dir.exists():
            return
        for directory in sorted(self.skills_dir.iterdir()):
            manifest = directory / "SKILL.md"
            if not directory.is_dir() or not manifest.exists():
                continue
            raw = manifest.read_text(encoding="utf-8")
            metadata, _ = self.parse_frontmatter(raw)
            name = metadata.get("name", directory.name)
            description = metadata.get(
                "description", raw.split("\n")[0].lstrip("#").strip()
            )
            self._skills[name] = {
                "name": name,
                "description": description,
                "content": raw,
            }

    def list_text(self) -> str:
        if not self._skills:
            return "(no skills found)"
        return "\n".join(
            f"- {skill['name']}: {skill['description']}"
            for skill in self._skills.values()
        )

    def load(self, name: str) -> str:
        skill = self._skills.get(name)
        if not skill:
            available = ", ".join(self._skills) or "(none)"
            return f"Skill not found: {name}. Available: {available}"
        return str(skill["content"])


class PromptAssembler:
    def __init__(
        self,
        settings: Settings,
        skills: SkillRegistry,
        tool_names: Callable[[], list[str]],
        mcp_names: Callable[[], list[str]],
    ):
        self.settings = settings
        self.skills = skills
        self.tool_names = tool_names
        self.mcp_names = mcp_names

    def assemble(self, context: dict) -> str:
        names = ", ".join(self.tool_names())
        sections = [
            "You are a coding agent. Act, don't explain.",
            f"Available tools: {names}.",
            f"Working directory: {self.settings.workdir}",
            f"Current time: {datetime.now().isoformat(timespec='seconds')}",
            "Skills catalog:\n"
            + self.skills.list_text()
            + "\nUse load_skill(name) when a skill is relevant.",
        ]
        if context.get("memories"):
            sections.append(f"Relevant memories:\n{context['memories']}")
        connected = self.mcp_names()
        if connected:
            sections.append(
                f"Connected MCP servers: {', '.join(connected)}"
            )
        return "\n\n".join(sections)
