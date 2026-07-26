"""Late-bound mock MCP servers and tool discovery."""

from __future__ import annotations

import re
import threading
from typing import Callable


class MCPClient:
    def __init__(self, name: str):
        self.name = name
        self.tools: list[dict] = []
        self._handlers: dict[str, Callable[..., str]] = {}

    def register(
        self,
        tool_defs: list[dict],
        handlers: dict[str, Callable[..., str]],
    ) -> None:
        self.tools = tool_defs
        self._handlers = handlers

    def call_tool(self, tool_name: str, args: dict) -> str:
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"MCP error: unknown tool '{tool_name}'"
        try:
            return str(handler(**args))
        except Exception as exc:
            return f"MCP error: {exc}"


_DISALLOWED_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def normalize_mcp_name(name: str) -> str:
    return _DISALLOWED_CHARS.sub("_", name)


def _mock_server_docs() -> MCPClient:
    client = MCPClient("docs")
    client.register(
        [
            {
                "name": "search",
                "description": "Search documentation. (readOnly)",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_version",
                "description": "Get API version. (readOnly)",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ],
        {
            "search": lambda query: f"[docs] Found 3 results for '{query}'",
            "get_version": lambda: "[docs] API v2.1.0",
        },
    )
    return client


def _mock_server_deploy() -> MCPClient:
    client = MCPClient("deploy")
    client.register(
        [
            {
                "name": "trigger",
                "description": (
                    "Trigger a deployment. "
                    "(destructive — requires approval in real CC)"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            },
            {
                "name": "status",
                "description": "Check deployment status. (readOnly)",
                "inputSchema": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            },
        ],
        {
            "trigger": lambda service: f"[deploy] Triggered: {service}",
            "status": lambda service: f"[deploy] {service}: running (v1.4.2)",
        },
    )
    return client


class MCPRegistry:
    def __init__(self, printer: Callable[[str], None] = print):
        self.printer = printer
        self._clients: dict[str, MCPClient] = {}
        self._factories: dict[str, Callable[[], MCPClient]] = {
            "docs": _mock_server_docs,
            "deploy": _mock_server_deploy,
        }
        self._lock = threading.RLock()

    @property
    def names(self) -> list[str]:
        with self._lock:
            return list(self._clients)

    def connect(self, name: str) -> str:
        with self._lock:
            if name in self._clients:
                return f"MCP server '{name}' already connected"
            factory = self._factories.get(name)
            if not factory:
                available = ", ".join(self._factories)
                return (
                    f"Unknown server '{name}'. Available: {available}"
                )
            client = factory()
            self._clients[name] = client
        tool_names = [tool["name"] for tool in client.tools]
        self.printer(
            f"  \033[31m[mcp] connected: {name} → {tool_names}\033[0m"
        )
        return (
            f"Connected to MCP server '{name}'. Discovered "
            f"{len(client.tools)} tools: {', '.join(tool_names)}"
        )

    def tool_pool(self) -> tuple[list[dict], dict[str, Callable[..., str]]]:
        definitions: list[dict] = []
        handlers: dict[str, Callable[..., str]] = {}
        with self._lock:
            clients = list(self._clients.items())
        for server_name, client in clients:
            safe_server = normalize_mcp_name(server_name)
            for tool in client.tools:
                safe_tool = normalize_mcp_name(tool["name"])
                prefixed = f"mcp__{safe_server}__{safe_tool}"
                definitions.append(
                    {
                        "name": prefixed,
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("inputSchema", {}),
                    }
                )

                def invoke(
                    *,
                    _client: MCPClient = client,
                    _tool: str = tool["name"],
                    **kwargs,
                ) -> str:
                    return _client.call_tool(_tool, kwargs)

                handlers[prefixed] = invoke
        return definitions, handlers
