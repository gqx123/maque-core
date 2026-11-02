"""Tool-related contracts: ToolSpec, ToolRegistry protocol, and CallBudget.

ToolSpec describes a tool's interface (name, description, parameters).
ToolRegistry is the protocol for registering and executing tools.
CallBudget tracks how many tool calls remain in a session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .messages import Observation


@dataclass
class ToolSpec:
    """Specification of a tool's interface.

    Attributes:
        name: Unique tool name.
        description: Human-readable description for the LLM.
        parameters: JSON Schema dict describing the tool's arguments.
        func: The callable that implements the tool.
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    func: Any = None  # callable, stored for execution

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible tool schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@runtime_checkable
class ToolRegistry(Protocol):
    """Protocol for tool registries.

    A registry knows about available tools, can return their schemas,
    and can execute a tool call by name.
    """

    def register(self, spec: ToolSpec) -> None:
        """Register a tool spec."""
        ...

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible schemas for all registered tools."""
        ...

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str = "",
    ) -> "Observation":
        """Execute a tool by name with the given arguments.

        Returns:
            An Observation with the result (or error).
        """
        ...

    @property
    def tool_names(self) -> List[str]:
        """Return list of registered tool names."""
        ...

    def __len__(self) -> int:
        """Number of registered tools."""
        ...


@dataclass
class CallBudget:
    """Tracks remaining tool call budget.

    Attributes:
        max_calls: Maximum number of tool calls allowed. None = unlimited.
        used: Number of tool calls already consumed.
    """

    max_calls: Optional[int] = None
    used: int = 0

    def consume(self, count: int = 1) -> bool:
        """Consume ``count`` calls. Returns True if within budget, False if exhausted."""
        if self.max_calls is None:
            self.used += count
            return True
        if self.used + count > self.max_calls:
            # Still count the attempted calls so ``exhausted`` stays True
            # but we report that we're over budget
            self.used += count
            return False
        self.used += count
        return True

    @property
    def exhausted(self) -> bool:
        """Whether the budget has been fully consumed."""
        if self.max_calls is None:
            return False
        return self.used >= self.max_calls

    @property
    def remaining(self) -> Optional[int]:
        """Number of remaining calls, or None if unlimited."""
        if self.max_calls is None:
            return None
        return max(0, self.max_calls - self.used)
