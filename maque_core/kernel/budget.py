"""Run budget — dual protection for agent runs.

Combines two limits:
1. max_iterations — maximum number of LLM calls (reasoning + acting cycles)
2. max_tool_calls — maximum number of tool invocations

Either limit being exhausted triggers a forced final answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunBudget:
    """Budget for a single agent run.

    Attributes:
        max_iterations: Maximum LLM calls. None = unlimited.
        max_tool_calls: Maximum tool calls. None = unlimited.
        iterations_used: Number of iterations consumed so far.
        tool_calls_used: Number of tool calls consumed so far.
    """

    max_iterations: Optional[int] = 10
    max_tool_calls: Optional[int] = None
    iterations_used: int = 0
    tool_calls_used: int = 0

    def consume_iteration(self) -> bool:
        """Consume one iteration. Returns True if within budget, False if exhausted."""
        self.iterations_used += 1
        if self.max_iterations is None:
            return True
        return self.iterations_used <= self.max_iterations

    def consume_tool_call(self, count: int = 1) -> bool:
        """Consume ``count`` tool calls. Returns True if within budget."""
        self.tool_calls_used += count
        if self.max_tool_calls is None:
            return True
        return self.tool_calls_used <= self.max_tool_calls

    @property
    def iterations_exhausted(self) -> bool:
        """Whether the iteration limit has been reached."""
        if self.max_iterations is None:
            return False
        return self.iterations_used >= self.max_iterations

    @property
    def tool_calls_exhausted(self) -> bool:
        """Whether the tool call limit has been reached."""
        if self.max_tool_calls is None:
            return False
        return self.tool_calls_used >= self.max_tool_calls

    @property
    def any_exhausted(self) -> bool:
        """Whether any budget limit has been reached."""
        return self.iterations_exhausted or self.tool_calls_exhausted

    def __repr__(self) -> str:
        return (
            f"RunBudget(iterations={self.iterations_used}/{self.max_iterations}, "
            f"tool_calls={self.tool_calls_used}/{self.max_tool_calls})"
        )
