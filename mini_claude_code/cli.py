"""Interactive command-line interface."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Sequence

from .config import ConfigurationError, Settings
from .runtime import Application


class Terminal:
    """Print background messages without permanently erasing pending input."""

    PROMPT = "\033[36mMiniClaudeCode >> \033[0m"

    def __init__(self):
        self.active = False
        try:
            import readline

            readline.parse_and_bind("set bind-tty-special-chars off")
            self._readline = readline
        except ImportError:
            self._readline = None

    def print(self, text: str) -> None:
        if threading.current_thread() is threading.main_thread() or not self.active:
            print(text)
            return
        pending = ""
        if self._readline:
            try:
                pending = self._readline.get_line_buffer()
            except Exception:
                pending = ""
        print(f"\r\033[K{text}")
        print(self.PROMPT + pending, end="", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-claude-code",
        description="MiniClaudeCode command-line coding agent",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help=(
            "Workspace and runtime-data directory. "
            "Defaults to the project directory."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="MiniClaudeCode 1.0.0",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(__file__).resolve().parents[1]
    try:
        settings = Settings.from_env(project_dir, args.workdir)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    terminal = Terminal()
    try:
        with Application(
            settings,
            printer=terminal.print,
            input_fn=input,
        ) as application:
            terminal.active = True
            print("MiniClaudeCode")
            print(
                "Enter a question, press Enter to send. Type q to quit.\n"
            )
            while True:
                try:
                    query = input(Terminal.PROMPT)
                except (EOFError, KeyboardInterrupt):
                    break
                if query.strip().lower() in ("q", "exit", ""):
                    break
                for text in application.run(query):
                    terminal.print(text)
                print()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    finally:
        terminal.active = False
    return 0
