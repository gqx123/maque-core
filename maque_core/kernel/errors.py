"""Kernel-level error types.

AgentRunError is the exception raised by the kernel loop when an LLM
call fails and ``raise_on_error`` is enabled. It carries both the
human-readable message and the raw :class:`LLMResponse` so upper
layers can inspect provider-level detail (model, usage, etc.) without
having to parse the error text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..contracts.messages import LLMResponse


class AgentRunError(Exception):
    """Raised when the ReAct loop encounters an unrecoverable LLM error.

    This is the kernel-layer error channel — it replaces the old pattern
    where errors were stuffed into ``LLMResponse.content`` and callers
    had to check ``response.is_error`` after every call.

    Attributes:
        message: Human-readable error description.
        response: The raw :class:`LLMResponse` that carried the error.
            ``None`` when the failure happened before a response was
            obtained (e.g. the loop itself detected a structural problem).
    """

    def __init__(
        self,
        message: str,
        *,
        response: Optional["LLMResponse"] = None,
    ):
        super().__init__(message)
        self.message = message
        self.response = response

    def __str__(self) -> str:
        return self.message


__all__ = ["AgentRunError"]
