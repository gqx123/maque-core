"""Tool Registry — central hub for tool management and execution.

The ToolRegistry:
- Stores ToolSpecs (what tools exist and their schemas)
- Returns OpenAI-compatible schemas for the LLM
- Executes tool calls by name (with parameter validation)
- Integrates with CallBudget for rate limiting
- Supports before/after hooks for observability and modification
- Delegates actual execution to a ToolExecutor (pluggable backends)
- Filters extra parameters the LLM might send that the tool doesn't accept
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Callable, Dict, List, Optional

from ..contracts.messages import Observation
from ..contracts.tool import CallBudget, ToolSpec
from .executor import ToolExecutor


class ToolRegistry:
    """Concrete tool registry with execution, validation, and hooks."""

    def __init__(
        self,
        budget: Optional[CallBudget] = None,
        executor: Optional[ToolExecutor] = None,
    ):
        self._tools: Dict[str, ToolSpec] = {}
        self._budget = budget
        self._before_hooks: List[Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]] = []
        self._after_hooks: List[Callable[[str, Dict[str, Any], Any, Optional[Exception]], None]] = []
        self._executor = executor or ToolExecutor()

    # ---- registration ----

    def register(self, spec: ToolSpec) -> None:
        """Register a tool spec."""
        self._tools[spec.name] = spec

    def register_from_specs(self, specs: Dict[str, ToolSpec]) -> int:
        """Bulk-register tools from a dict of name -> ToolSpec.

        Returns the number of tools registered.
        """
        count = 0
        for name, spec in specs.items():
            if name not in self._tools:
                self._tools[name] = spec
                count += 1
        return count

    # ---- schema output ----

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible schemas for all registered tools."""
        return [spec.to_openai_schema() for spec in self._tools.values()]

    @property
    def tool_names(self) -> List[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def get_spec(self, name: str) -> Optional[ToolSpec]:
        """Get a tool spec by name, or None if not found."""
        return self._tools.get(name)

    # ---- hooks ----

    def add_before_hook(
        self, hook: Callable[[str, Dict[str, Any]], Optional[Dict[str, Any]]]
    ) -> None:
        """Add a before-execution hook.

        The hook receives (tool_name, arguments) and can return:
        - None: proceed with original arguments
        - A modified dict: use these arguments instead (short-circuit
          modification, not short-circuit execution)

        To short-circuit execution entirely, raise an exception from the
        hook — it will be caught and returned as an error observation.
        """
        self._before_hooks.append(hook)

    def add_after_hook(
        self, hook: Callable[[str, Dict[str, Any], Any, Optional[Exception]], None]
    ) -> None:
        """Add an after-execution hook.

        The hook receives (tool_name, arguments, result, error).
        Called regardless of success or failure.
        """
        self._after_hooks.append(hook)

    # ---- execution ----

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str = "",
    ) -> Observation:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            tool_call_id: ID for tracking (returned in the Observation).

        Returns:
            An Observation with the result or error.
        """
        # Budget check
        if self._budget is not None and self._budget.exhausted:
            return Observation(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=f"Tool call budget exhausted (max={self._budget.max_calls}).",
                is_error=True,
                metadata={"error_type": "budget_exhausted"},
            )

        # Check tool exists
        spec = self._tools.get(tool_name)
        if spec is None:
            return Observation(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=f"Unknown tool: {tool_name}. Available tools: {', '.join(self.tool_names) or 'none'}",
                is_error=True,
                metadata={"error_type": "unknown_tool"},
            )

        # Validate arguments
        validation_error = self._validate_args(spec, arguments)
        if validation_error:
            return Observation(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=validation_error,
                is_error=True,
                metadata={"error_type": "validation_error"},
            )

        # Before hooks
        args = dict(arguments)
        try:
            for hook in self._before_hooks:
                result = hook(tool_name, args)
                if result is not None:
                    args = result
        except Exception as e:
            return Observation(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=f"Before hook error: {e}",
                is_error=True,
                metadata={"error_type": "before_hook_error"},
            )

        # Consume budget
        if self._budget is not None:
            ok = self._budget.consume()
            if not ok:
                # Budget was exhausted right now
                return Observation(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    content=f"Tool call budget exhausted after this call (max={self._budget.max_calls}).",
                    is_error=True,
                    metadata={"error_type": "budget_exhausted"},
                )

        # Execute
        try:
            # Filter extra parameters — only keep ones the tool actually accepts.
            # LLMs sometimes hallucinate extra parameters; silently dropping them
            # is more robust than failing the whole call.
            accepted_params = set(spec.parameters.get("properties", {}).keys())
            filtered_args = {k: v for k, v in args.items() if k in accepted_params}

            exec_result = self._executor.run(spec.func, filtered_args)

            if exec_result.get("is_error"):
                error_msg = exec_result.get("error", "Unknown execution error")
                original_exc = exec_result.get("exception")
                # After hooks (failure) — pass the original exception if available
                for hook in self._after_hooks:
                    try:
                        hook(tool_name, filtered_args, None, original_exc or Exception(error_msg))
                    except Exception:
                        pass

                return Observation(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    content=error_msg,
                    is_error=True,
                    metadata={
                        "error_type": "execution_error",
                        "traceback": exec_result.get("traceback", ""),
                        "duration_ms": exec_result.get("duration_ms", 0),
                    },
                )

            result = exec_result.get("result")
            # Convert result to string
            if isinstance(result, str):
                content = result
            else:
                content = json.dumps(result, ensure_ascii=False, default=str)

            # After hooks (success)
            for hook in self._after_hooks:
                try:
                    hook(tool_name, filtered_args, result, None)
                except Exception:
                    pass  # hook errors don't fail the execution

            return Observation(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=content,
                is_error=False,
                metadata={"duration_ms": exec_result.get("duration_ms", 0)},
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"

            # After hooks (failure)
            for hook in self._after_hooks:
                try:
                    hook(tool_name, args, None, e)
                except Exception:
                    pass

            return Observation(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                content=error_msg,
                is_error=True,
                metadata={
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                },
            )

    # ---- internal helpers ----

    def _validate_args(self, spec: ToolSpec, arguments: Dict[str, Any]) -> Optional[str]:
        """Validate arguments against the tool spec.

        Returns None if valid, or an error string if invalid.
        """
        params = spec.parameters.get("properties", {})
        required = spec.parameters.get("required", [])

        # Check required params
        for req in required:
            if req not in arguments:
                return f"Missing required parameter: '{req}'"

        # Type check provided params
        for name, value in arguments.items():
            if name not in params:
                # Extra param — warn but don't fail
                continue
            expected_type = params[name].get("type")
            if expected_type and not self._check_type(value, expected_type):
                return (
                    f"Parameter '{name}' expected type '{expected_type}', "
                    f"got '{type(value).__name__}'"
                )

        return None

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        """Check if a value matches the expected JSON Schema type."""
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            # bool is a subclass of int in Python — exclude it
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "array":
            return isinstance(value, list)
        if expected == "object":
            return isinstance(value, dict)
        return True  # unknown type — pass through
