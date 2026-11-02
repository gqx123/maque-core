"""LLM provider protocol — the interface every LLM backend must implement.

Using a Protocol (structural subtyping) means providers don't need to
inherit from a base class — they just need the right method signature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .messages import LLMResponse, Message, ToolSpec


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM backends.

    Any class with a compatible ``chat`` method is automatically an
    LLMProvider — no explicit inheritance required.
    """

    def chat(
        self,
        messages: List["Message"],
        tools: Optional[List["ToolSpec"]] = None,
        max_tokens: Optional[int] = None,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> "LLMResponse":
        """Send a chat completion request and return the response.

        Args:
            messages: The conversation history.
            tools: Optional list of tool specs the model may call.
            max_tokens: Maximum number of tokens to generate. ``None``
                means "use the provider's configured default".
            raise_on_error: When ``False`` (default), API failures are
                returned as an :class:`LLMResponse` with ``error`` set.
                When ``True``, they are raised as an
                :class:`~maque_core.llm.errors.LLMError` subclass instead.
            **kwargs: Backend-specific parameters (temperature, model, etc.).

        Returns:
            An LLMResponse with content and/or tool_calls.
        """
        ...
