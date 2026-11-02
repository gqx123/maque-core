"""Mock LLM provider — for testing and offline demos.

Returns pre-programmed responses based on a list of LLMResponse objects.
Each call consumes the next response in the sequence.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..contracts.messages import LLMResponse, Message
from .errors import LLMError


class MockLLM:
    """Mock LLM provider for testing.

    Takes a list of LLMResponse objects and returns them in order.
    Useful for unit tests that don't need a real LLM API.

    Usage::

        mock = MockLLM([
            LLMResponse(content="Hello!"),
            LLMResponse(tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "hi"})]),
            LLMResponse(content="Done!"),
        ])
    """

    def __init__(self, responses: Optional[List[LLMResponse]] = None):
        self._responses: List[LLMResponse] = responses or []
        self._index: int = 0
        self.call_count: int = 0
        self.last_messages: Optional[List[Message]] = None
        self.last_tools: Optional[List[Any]] = None
        self.last_max_tokens: Optional[int] = None
        self.all_messages: List[List[Message]] = []

    def add_response(self, response: LLMResponse) -> None:
        """Add a response to the end of the sequence."""
        self._responses.append(response)

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        max_tokens: Optional[int] = None,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return the next mock response in sequence.

        If no more responses are configured, returns an empty response.

        When the queued response carries ``error`` and ``raise_on_error``
        is True, the error is raised instead — mirroring the real provider.
        """
        self.call_count += 1
        self.last_messages = list(messages)
        self.last_tools = tools
        self.last_max_tokens = max_tokens
        self.all_messages.append(list(messages))

        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            if raise_on_error and response.error is not None:
                raise LLMError(response.error, provider="MockLLM", model=response.model)
            return response

        # No more mock responses — return empty
        return LLMResponse(content="", model="mock")

    def reset(self) -> None:
        """Reset the mock to its initial state."""
        self._index = 0
        self.call_count = 0
        self.last_messages = None
        self.last_tools = None
        self.last_max_tokens = None
        self.all_messages = []

    @property
    def exhausted(self) -> bool:
        """Whether all mock responses have been consumed."""
        return self._index >= len(self._responses)

    @property
    def remaining(self) -> int:
        """Number of remaining mock responses."""
        return len(self._responses) - self._index
