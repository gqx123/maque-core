"""Global tool registry — holds ToolSpecs registered by @tool decorators.

Why a global dict? Because Python caches imported modules. When
``discover_tools()`` imports a module that's already been imported
elsewhere, the @tool decorator doesn't re-execute. By always registering
to a single global dict at import time, and having ``discover_tools()``
bulk-load from that global, we avoid the "already imported → no tools
found" problem.
"""

from __future__ import annotations

from typing import Dict

from ..contracts.tool import ToolSpec

# The single source of truth for all @tool-decorated functions.
# Populated at module import time by the @tool decorator.
_global_tool_specs: Dict[str, ToolSpec] = {}


def register_global(spec: ToolSpec) -> None:
    """Register a ToolSpec in the global registry.

    Called automatically by the @tool decorator. Can also be called
    manually for programmatic registration.
    """
    _global_tool_specs[spec.name] = spec


def get_global_specs() -> Dict[str, ToolSpec]:
    """Return a copy of all globally registered tool specs."""
    return dict(_global_tool_specs)


def clear_global() -> None:
    """Clear the global registry. Mainly for testing."""
    _global_tool_specs.clear()
