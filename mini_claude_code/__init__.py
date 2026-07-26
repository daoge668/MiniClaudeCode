"""MiniClaudeCode.

Importing this package is intentionally side-effect free. Runtime directories,
environment validation, model clients, and background threads are created only
when :class:`Application` is started.
"""

from .config import ConfigurationError, Settings
from .runtime import Application
from .tools import ToolRegistry

__all__ = ["Application", "ConfigurationError", "Settings", "ToolRegistry"]
__version__ = "1.0.0"
