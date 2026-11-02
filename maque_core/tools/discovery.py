"""Tool auto-discovery — scan Python packages for @tool-decorated functions.

Discovers tools by importing modules in a package, which triggers the
@tool decorators to register into the global registry. Then bulk-loads
all globally registered specs into the target registry.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Optional

from ._global_registry import get_global_specs
from .registry import ToolRegistry


def discover_tools(
    registry: ToolRegistry,
    package_name: str,
    search_path: Optional[str] = None,
) -> int:
    """Discover and register all @tool-decorated functions in a package.

    Works by importing every module in the package, which causes the
    @tool decorators to execute and register into the global registry.
    Then all globally registered specs are loaded into ``registry``.

    Args:
        registry: The target ToolRegistry to register discovered tools into.
        package_name: Dotted package name, e.g. "maque_core.tools.impl".
        search_path: Optional filesystem path to search. If provided,
            the path is added to sys.path temporarily.

    Returns:
        Number of newly registered tools.
    """
    # Add search path if provided
    if search_path:
        search_path = str(Path(search_path).resolve())
        if search_path not in sys.path:
            sys.path.insert(0, search_path)

    try:
        # Try to import the package
        package = importlib.import_module(package_name)
    except ImportError:
        # If the package can't be imported, return 0
        return 0

    # Walk through all modules in the package
    package_path = getattr(package, "__path__", None)
    if package_path is not None:
        for module_info in pkgutil.iter_modules(package_path, prefix=f"{package_name}."):
            try:
                importlib.import_module(module_info.name)
            except Exception:
                # Skip modules that fail to import
                continue

    # Load all globally registered specs into the target registry
    global_specs = get_global_specs()
    return registry.register_from_specs(global_specs)


def discover_from_path(
    registry: ToolRegistry,
    directory: str,
    module_prefix: str = "",
) -> int:
    """Discover tools from a filesystem directory.

    Each .py file in the directory is treated as a module.

    Args:
        registry: Target ToolRegistry.
        directory: Path to the directory containing .py files.
        module_prefix: Optional prefix for module names.

    Returns:
        Number of newly registered tools.
    """
    dir_path = Path(directory).resolve()
    dir_str = str(dir_path)

    # Add the directory to path so modules can be imported
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)

    # Import each Python file as a module
    for py_file in sorted(dir_path.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = py_file.stem
        if module_prefix:
            module_name = f"{module_prefix}.{module_name}"
        try:
            importlib.import_module(module_name)
        except Exception:
            continue

    global_specs = get_global_specs()
    return registry.register_from_specs(global_specs)
