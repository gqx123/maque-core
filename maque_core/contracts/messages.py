"""Core message data structures for agent communication.

Defines the universal message types used across all layers:
- Role: who sent a message
- ToolCall: a requested tool invocation
- Message: a chat message (system / user / assistant / tool)
- LLMResponse: what the LLM returns
- Observation: the result of a tool call
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    """Message role — who sent the message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def __str__(self) -> str:
        return self.value


@dataclass
class ToolCall:
    """A single tool invocation requested by the model.

    Attributes:
        id: Unique identifier for this tool call (used to match results).
        name: Name of the tool to invoke.
        arguments: Dict of argument name → value to pass to the tool.
    """

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data.get("arguments", {}),
        )


@dataclass
class Message:
    """A single chat message.

    Use factory methods for readability:
        Message.system("You are helpful.")
        Message.user("Hello")
        Message.assistant("Hi there!", tool_calls=[...])
        Message.tool_result("call_1", "echo", "output text")

    Attributes:
        reasoning: Optional reasoning / chain-of-thought content
            associated with this message. Used for multi-turn
            conversations so that subsequent rounds can see the
            model's previous reasoning basis. None when the message
            has no reasoning (system / user / tool messages, or
            when ``include_reasoning_in_history`` is False).
    """

    role: Role
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    reasoning: Optional[str] = None

    # ---- factory methods ----

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> "Message":
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls or [],
        )

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        tool_name: str,
        content: str,
    ) -> "Message":
        return cls(
            role=Role.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"role": str(self.role), "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_name is not None:
            data["name"] = self.tool_name
        if self.reasoning is not None:
            data["reasoning"] = self.reasoning
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        role = Role(data["role"])
        tool_calls = [
            ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])
        ]
        return cls(
            role=role,
            content=data.get("content", ""),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            tool_name=data.get("name"),
            reasoning=data.get("reasoning"),
        )


@dataclass
class LLMResponse:
    """The result of an LLM chat completion call.

    Attributes:
        content: The text response (may be empty if only tool calls).
            When the call failed and the adapter returned instead of
            raising, ``content`` holds the human-readable error text.
        reasoning: The model's internal reasoning / chain-of-thought.
            None if the model did not produce reasoning content.
        tool_calls: Tool invocations requested by the model.
        model: Name of the model that generated the response.
        usage: Optional token usage info.
        finish_reason: Why the model stopped generating. Passed through
            from the API verbatim (``stop`` / ``length`` / ``tool_calls`` /
            ``content_filter`` ...). ``"length"`` means the answer was
            truncated by the token budget — it is NOT a complete answer.
            None when the provider did not report one (e.g. on failure).
        error: Error description when the call failed. ``None`` means the
            response is a genuine model output. When set, ``content`` also
            carries the message so it can still be displayed, but callers
            MUST NOT treat it as an answer — check this field, or
            ``response.is_error``.
    """

    content: str = ""
    reasoning: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    model: str = ""
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_error(self) -> bool:
        """True when this response represents a failed LLM call."""
        return self.error is not None

    @property
    def is_truncated(self) -> bool:
        """True when the model stopped because it hit the token limit."""
        return self.finish_reason == "length"

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMResponse":
        return cls(
            content=data.get("content", ""),
            reasoning=data.get("reasoning"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            model=data.get("model", ""),
            usage=data.get("usage"),
            finish_reason=data.get("finish_reason"),
            error=data.get("error"),
        )


@dataclass
class Observation:
    """The result of executing a tool call.

    Attributes:
        tool_call_id: Matches the id of the originating ToolCall.
        tool_name: Name of the tool that was executed.
        content: String output of the tool.
        is_error: Whether the tool call resulted in an error.
        metadata: Extra info like execution duration, etc.
    """

    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "content": self.content,
            "is_error": self.is_error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            content=data["content"],
            is_error=data.get("is_error", False),
            metadata=data.get("metadata", {}),
        )

    def to_message(self) -> Message:
        """Convert to a tool-role Message for feeding back to the LLM."""
        return Message.tool_result(
            tool_call_id=self.tool_call_id,
            tool_name=self.tool_name,
            content=self.content,
        )
