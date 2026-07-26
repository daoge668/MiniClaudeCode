"""Model gateway and retry policy."""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Protocol

from .config import ConfigurationError, Settings
from .models import RecoveryState


class ModelGateway(Protocol):
    def create_message(self, **kwargs: Any) -> Any:
        """Create one model response."""


class AnthropicGateway:
    """Small adapter around the Anthropic SDK.

    Importing the SDK and constructing its client are deferred until explicit
    application startup.
    """

    def __init__(self, client: Any):
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnthropicGateway":
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - installation guidance
            raise ConfigurationError(
                "Missing dependency 'anthropic'. Run: pip install -e ."
            ) from exc

        kwargs: dict[str, Any] = {}
        if settings.anthropic_base_url:
            kwargs["base_url"] = settings.anthropic_base_url
        try:
            return cls(Anthropic(**kwargs))
        except Exception as exc:
            raise ConfigurationError(
                f"Could not initialize the Anthropic client: {exc}"
            ) from exc

    def create_message(self, **kwargs: Any) -> Any:
        return self._client.messages.create(**kwargs)


def block_type(block: Any) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def extract_text(content: Any) -> str:
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if block_type(block) != "text":
            continue
        value = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
        if value:
            parts.append(str(value))
    return "\n".join(parts).strip()


def has_tool_use(content: Any) -> bool:
    return isinstance(content, list) and any(
        block_type(block) == "tool_use" for block in content
    )


def is_prompt_too_long_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        ("prompt" in message and "long" in message)
        or "context_length_exceeded" in message
        or "max_context_window" in message
    )


class RetryController:
    def __init__(
        self,
        settings: Settings,
        printer: Callable[[str], None] = print,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings
        self.printer = printer
        self.sleeper = sleeper

    def retry_delay(self, attempt: int) -> float:
        base = min(self.settings.base_delay_ms * (2**attempt), 32_000) / 1_000
        return base + random.uniform(0, base * 0.25)

    def run(self, fn: Callable[[], Any], state: RecoveryState) -> Any:
        for attempt in range(self.settings.max_retries):
            try:
                result = fn()
                state.consecutive_529 = 0
                return result
            except Exception as exc:
                name = type(exc).__name__.lower()
                message = str(exc).lower()
                if "ratelimit" in name or "429" in message:
                    delay = self.retry_delay(attempt)
                    self.printer(
                        f"  \033[33m[429] retry {attempt + 1}/"
                        f"{self.settings.max_retries} after {delay:.1f}s\033[0m"
                    )
                    self.sleeper(delay)
                    continue
                if (
                    "overloaded" in name
                    or "529" in message
                    or "overloaded" in message
                ):
                    state.consecutive_529 += 1
                    if (
                        state.consecutive_529
                        >= self.settings.max_consecutive_529
                        and self.settings.fallback_model_id
                    ):
                        state.current_model = self.settings.fallback_model_id
                        state.consecutive_529 = 0
                        self.printer(
                            "  \033[31m[529] switching to "
                            f"{self.settings.fallback_model_id}\033[0m"
                        )
                    delay = self.retry_delay(attempt)
                    self.printer(
                        f"  \033[33m[529] retry {attempt + 1}/"
                        f"{self.settings.max_retries} after {delay:.1f}s\033[0m"
                    )
                    self.sleeper(delay)
                    continue
                raise
        raise RuntimeError(
            f"Max retries ({self.settings.max_retries}) exceeded"
        )
