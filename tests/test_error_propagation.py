"""Tests for P0 error propagation via exception channel.

Verifies that:
- ``react_loop`` raises ``AgentRunError`` when ``raise_on_error=True``
  and keeps legacy behaviour by default.
- ``Agent.ask()`` raises by default (facade layer) and rolls back
  conversation history on failure (no orphan user message).
- The normal (success) path is unaffected by the changes.
"""

from __future__ import annotations

import pytest

from maque_core.contracts.messages import LLMResponse, Message, Role, ToolCall
from maque_core.kernel import Agent, AgentRunError, react_loop
from maque_core.kernel.budget import RunBudget
from maque_core.llm.mock import MockLLM
from maque_core.tools.registry import ToolRegistry


# ---- helpers ----

def _make_error_response(message: str = "LLM API error: network: Connection refused") -> LLMResponse:
    """Build an LLMResponse that represents a failed call."""
    return LLMResponse(
        content=message,
        error=message,
        model="mock-error",
    )


def _make_ok_response(content: str = "The answer is 42.") -> LLMResponse:
    """Build a normal successful LLMResponse."""
    return LLMResponse(content=content, model="mock-ok")


def _empty_registry() -> ToolRegistry:
    """Return a tool registry with no tools."""
    return ToolRegistry()


# ---- Test 1: react_loop raises on error when enabled ----

class TestReactLoopRaisesOnError:
    """react_loop with raise_on_error=True raises AgentRunError."""

    def test_react_loop_raises_on_error_when_enabled(self):
        """When raise_on_error=True and LLM returns error, raise AgentRunError."""
        mock = MockLLM([_make_error_response()])
        messages = [Message.system("You are helpful."), Message.user("Hi")]

        with pytest.raises(AgentRunError) as exc_info:
            react_loop(
                llm=mock,
                registry=_empty_registry(),
                messages=messages,
                budget=RunBudget(max_iterations=5),
                raise_on_error=True,
            )

        # The exception carries the error message
        assert "LLM API error" in exc_info.value.message
        # The exception carries the raw response
        assert exc_info.value.response is not None
        assert exc_info.value.response.error is not None
        assert exc_info.value.response.model == "mock-error"


# ---- Test 2: react_loop default keeps legacy behaviour ----

class TestReactLoopDefaultLegacy:
    """react_loop default (raise_on_error=False) keeps old behaviour."""

    def test_react_loop_default_keeps_legacy_behavior(self):
        """Default raise_on_error=False: error returned in response content."""
        error_msg = "LLM API error: network: Connection refused"
        mock = MockLLM([_make_error_response(error_msg)])
        messages = [Message.system("You are helpful."), Message.user("Hi")]

        # Should NOT raise — returns the error as content (legacy)
        answer, reasoning = react_loop(
            llm=mock,
            registry=_empty_registry(),
            messages=messages,
            budget=RunBudget(max_iterations=5),
        )

        # Legacy behaviour: error text is the "answer"
        assert answer == error_msg
        # No exception was raised
        assert reasoning is None


# ---- Test 3: Agent.ask raises on LLM error by default ----

class TestAgentAskRaises:
    """Agent.ask() raises AgentRunError by default (facade layer)."""

    def test_agent_ask_raises_on_llm_error(self):
        """Agent.ask default (raise_on_error=True) raises on LLM failure."""
        mock = MockLLM([_make_error_response()])
        agent = Agent(llm=mock, registry=_empty_registry())

        with pytest.raises(AgentRunError) as exc_info:
            agent.ask("Hello")

        assert "LLM API error" in exc_info.value.message
        assert exc_info.value.response is not None
        assert exc_info.value.response.error is not None


# ---- Test 4: Agent.ask rolls back history on error ----

class TestAgentAskRollback:
    """Failed Agent.ask() rolls back history — no orphan user message."""

    def test_agent_ask_rolls_back_history_on_error(self):
        """After a failed ask(), history is rolled back to pre-call state."""
        mock = MockLLM([_make_error_response()])
        agent = Agent(llm=mock, registry=_empty_registry())

        # Record the initial state (just system prompt)
        initial_count = len(agent.messages)
        assert initial_count == 1  # system prompt only
        assert agent.messages[0].role == Role.SYSTEM

        # The call fails
        with pytest.raises(AgentRunError):
            agent.ask("Hello")

        # After rollback: back to just the system prompt —
        # no orphan user message, no assistant message.
        assert len(agent.messages) == initial_count
        assert agent.messages[0].role == Role.SYSTEM

        # All messages should still be just the system prompt
        user_msgs = [m for m in agent.messages if m.role == Role.USER]
        assistant_msgs = [m for m in agent.messages if m.role == Role.ASSISTANT]
        assert len(user_msgs) == 0, "Orphan user message left after error"
        assert len(assistant_msgs) == 0, "Orphan assistant message left after error"


# ---- Test 5: normal path unaffected ----

class TestAgentAskNormal:
    """The normal (success) path still works — returns a string answer."""

    def test_agent_ask_normal_still_returns_str(self):
        """Successful ask() returns the answer string as before."""
        mock = MockLLM([_make_ok_response("Hello there!")])
        agent = Agent(llm=mock, registry=_empty_registry())

        answer = agent.ask("Hi")

        assert isinstance(answer, str)
        assert answer == "Hello there!"
        # History was appended correctly
        assert len(agent.messages) == 3  # system + user + assistant
        assert agent.messages[1].role == Role.USER
        assert agent.messages[1].content == "Hi"
        assert agent.messages[2].role == Role.ASSISTANT
        assert agent.messages[2].content == "Hello there!"
