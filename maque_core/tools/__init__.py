"""Tools layer — registry, decorators, execution, and discovery.

Public API:
    from maque_core.tools import ToolRegistry, tool, discover_tools, ToolExecutor
"""

from .decorator import tool
from .discovery import discover_tools, discover_from_path
from .executor import InProcessBackend, ToolExecutor
from .registry import ToolRegistry

__all__ = [
    "ToolRegistry",
    "tool",
    "discover_tools",
    "discover_from_path",
    "ToolExecutor",
    "InProcessBackend",
]
