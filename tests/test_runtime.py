from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from mini_claude_code.runtime import Application


@dataclass
class FakeGateway:
    events: list

    def __post_init__(self) -> None:
        self.calls: list[dict] = []

    def create_message(self, **kwargs):
        self.calls.append(kwargs)
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event


def response(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def text(value: str) -> dict:
    return {"type": "text", "text": value}


def test_application_lifecycle_and_text_turn(settings) -> None:
    gateway = FakeGateway([response([text("hello")])])
    application = Application(
        settings, gateway=gateway, printer=lambda _text: None
    )
    application.start()
    thread = application.cron._thread
    application.start()
    assert application.cron._thread is thread
    assert application.run("hi") == ["hello"]
    application.close()
    assert not application.cron.is_running


def test_tool_call_uses_unified_dispatcher(settings) -> None:
    gateway = FakeGateway(
        [
            response(
                [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "todo_write",
                        "input": {
                            "todos": [
                                {
                                    "content": "test",
                                    "status": "in_progress",
                                }
                            ]
                        },
                    }
                ],
                stop_reason="tool_use",
            ),
            response([text("done")]),
        ]
    )
    with Application(
        settings, gateway=gateway, printer=lambda _text: None
    ) as application:
        assert application.run("work") == ["done"]
        assert application.todos.items[0]["content"] == "test"


def test_permission_denial_reaches_model(settings) -> None:
    gateway = FakeGateway(
        [
            response(
                [
                    {
                        "type": "tool_use",
                        "id": "unsafe",
                        "name": "bash",
                        "input": {"command": "sudo reboot"},
                    }
                ],
                stop_reason="tool_use",
            ),
            response([text("blocked")]),
        ]
    )
    with Application(
        settings, gateway=gateway, printer=lambda _text: None
    ) as application:
        assert application.run("unsafe") == ["blocked"]
        assert "Permission denied" in str(application.history)


def test_rate_limit_retry(settings) -> None:
    gateway = FakeGateway(
        [RuntimeError("429 rate limit"), response([text("recovered")])]
    )
    with Application(
        settings, gateway=gateway, printer=lambda _text: None
    ) as application:
        assert application.run("retry") == ["recovered"]
    assert len(gateway.calls) == 2
