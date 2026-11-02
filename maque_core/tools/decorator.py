"""@tool decorator — turn any Python function into a registered tool.

The decorator inspects the function's type hints and docstring to
auto-generate a JSON Schema for the tool's parameters, then registers
the tool in the global registry so it can be discovered later.

Supported parameter types: str, int, float, bool, list, dict, Optional[X].
Parameter descriptions are extracted from Google/NumPy/ReST-style
docstrings when available.
"""

from __future__ import annotations

import inspect
import re
import typing
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from ..contracts.tool import ToolSpec
from ._global_registry import register_global

# Mapping from Python types to JSON Schema types
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _type_to_schema(py_type: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema type dict."""
    # Handle Optional[X] (which is Union[X, None])
    origin = typing.get_origin(py_type)
    args = typing.get_args(py_type)

    if origin is typing.Union:
        # Optional[X] is Union[X, NoneType]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _type_to_schema(non_none[0])
        return {"type": "string"}  # fallback

    if origin is list:
        return {"type": "array"}

    if origin is dict:
        return {"type": "object"}

    if py_type in _TYPE_MAP:
        return {"type": _TYPE_MAP[py_type]}

    # Fallback for unknown types
    return {"type": "string"}


def _extract_description(docstring: Optional[str]) -> str:
    """Extract the first line of a docstring as the tool description."""
    if not docstring:
        return ""
    # Take the first non-empty line
    for line in docstring.strip().split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def _extract_param_descriptions(docstring: Optional[str]) -> Dict[str, str]:
    """Extract parameter descriptions from a docstring.

    Supports common docstring styles:
    - Google style: Args / Parameters section with ``name: description``
    - NumPy style: Parameters section with ``name : type`` then description
    - ReST style: ``:param name: description``

    Returns a dict of param_name -> description.
    """
    if not docstring:
        return {}

    descriptions: Dict[str, str] = {}
    lines = docstring.split("\n")

    # --- ReST style: :param name: description ---
    rest_pattern = re.compile(r"^\s*:param\s+(\w+)\s*:\s*(.+)$")
    for line in lines:
        m = rest_pattern.match(line)
        if m:
            descriptions[m.group(1)] = m.group(2).strip()

    if descriptions:
        return descriptions

    # --- Google / NumPy style: find Args / Parameters section ---
    section_start = -1
    section_indent = 0
    section_headers = ("Args:", "Arguments:", "Parameters:", "Params:")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in section_headers:
            section_start = i
            # Find the indent of the first param line
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip():
                    section_indent = len(next_line) - len(next_line.lstrip())
                    break
            break

    if section_start < 0:
        return {}

    # Parse parameter lines within the section
    current_param = None
    current_desc_parts: List[str] = []

    for i in range(section_start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            # Empty line — if we're building a description, keep going;
            # but if it's followed by a non-indented line, end the section
            if current_param and i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                next_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                if next_stripped and next_indent < section_indent:
                    break
            continue

        indent = len(line) - len(line.lstrip())

        # If indent is less than section indent, we've left the section
        if indent < section_indent and stripped:
            break

        # Check if this line starts a new parameter
        # Pattern: ``param_name (type): description`` or ``param_name: description``
        param_match = re.match(
            r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line
        )
        if param_match and indent == section_indent:
            # Save previous param
            if current_param and current_desc_parts:
                descriptions[current_param] = " ".join(current_desc_parts).strip()

            current_param = param_match.group(1)
            desc_text = param_match.group(2).strip()
            current_desc_parts = [desc_text] if desc_text else []
        elif current_param and indent > section_indent:
            # Continuation line — add to current param's description
            current_desc_parts.append(stripped)
        elif indent == section_indent and not param_match:
            # Not a param line at the same indent — might be end of section
            if current_param and current_desc_parts:
                descriptions[current_param] = " ".join(current_desc_parts).strip()
                current_param = None
                current_desc_parts = []

    # Don't forget the last one
    if current_param and current_desc_parts:
        descriptions[current_param] = " ".join(current_desc_parts).strip()

    return descriptions


def tool(name: Optional[str] = None, description: Optional[str] = None) -> Callable:
    """Decorator that registers a function as a tool.

    Usage::

        @tool
        def echo(text: str) -> str:
            \"\"\"Echo back the input text.\"\"\"
            return text

        @tool(name="my_echo", description="Custom echo tool")
        def echo(text: str, times: int = 1) -> str:
            return text * times

    Args:
        name: Optional custom tool name. Defaults to the function name.
        description: Optional custom description. Defaults to the first
            line of the function's docstring.

    Returns:
        The original function (unwrapped — we just register metadata).
    """
    # Handle bare @tool usage (no parentheses)
    if callable(name) and description is None:
        # name is actually the function being decorated
        func = name
        return _register_tool(func, func.__name__, _extract_description(func.__doc__))

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or _extract_description(func.__doc__)
        return _register_tool(func, tool_name, tool_desc)

    return decorator


def _register_tool(func: Callable, tool_name: str, description: str) -> Callable:
    """Build a ToolSpec from a function and register it globally."""
    # Get type hints
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    # Get function signature
    sig = inspect.signature(func)

    # Extract parameter descriptions from docstring
    param_descs = _extract_param_descriptions(func.__doc__)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        # Skip 'self' for methods
        if param_name == "self":
            continue

        param_type = hints.get(param_name, str)
        schema = _type_to_schema(param_type)

        # Add description from docstring if available
        if param_name in param_descs:
            schema["description"] = param_descs[param_name]

        properties[param_name] = schema

        # Required = no default value
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    spec = ToolSpec(
        name=tool_name,
        description=description,
        parameters=parameters,
        func=func,
    )

    register_global(spec)
    return func
