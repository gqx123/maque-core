"""Tests for the kernel layer — ReAct loop, Agent, budget, and multi-turn."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from maque_core.contracts.messages import LLMResponse, Message, ToolCall, Role
from maque_core.contracts.tool import ToolSpec
from maque_core.kernel.budget import RunBudget
from maque_core.kernel.loop import react_loop
from maque_core.kernel.agent import Agent
from maque_core.llm import MockLLM
from maque_core.tools import ToolRegistry


# ============================================================
# RunBudget tests
# ============================================================

class TestRunBudget:
    def test_default_budget(self):
        budget = RunBudget()
        assert budget.max_iterations == 10
        assert budget.max_tool_calls is None
        assert budget.iterations_used == 0
        assert budget.tool_calls_used == 0

    def test_custom_budget(self):
        budget = RunBudget(max_iterations=5, max_tool_calls=10)
        assert budget.max_iterations == 5
        assert budget.max_tool_calls == 10

    def test_consume_iteration(self):
        budget = RunBudget(max_iterations=3)
        assert budget.consume_iteration() is True
        assert budget.iterations_used == 1
        assert budget.consume_iteration() is True
        assert budget.iterations_used == 2
        assert budget.consume_iteration() is True
        assert budget.iterations_used == 3
        # Fourth call — exhausted
        assert budget.consume_iteration() is False
        assert budget.iterations_used == 4

    def test_consume_tool_call(self):
        budget = RunBudget(max_tool_calls=2)
        assert budget.consume_tool_call() is True
        assert budget.consume_tool_call() is True
        assert budget.consume_tool_call() is False
        assert budget.tool_calls_used == 3

    def test_iterations_exhausted(self):
        budget = RunBudget(max_iterations=2)
        assert budget.iterations_exhausted is False
        budget.consume_iteration()
        assert budget.iterations_exhausted is False
        budget.consume_iteration()
        assert budget.iterations_exhausted is True

    def test_tool_calls_exhausted(self):
        budget = RunBudget(max_tool_calls=1)
        assert budget.tool_calls_exhausted is False
        budget.consume_tool_call()
        assert budget.tool_calls_exhausted is True

    def test_any_exhausted_iterations(self):
        budget = RunBudget(max_iterations=1, max_tool_calls=100)
        assert budget.any_exhausted is False
        budget.consume_iteration()
        assert budget.any_exhausted is True

    def test_any_exhausted_tool_calls(self):
        budget = RunBudget(max_iterations=100, max_tool_calls=1)
        assert budget.any_exhausted is False
        budget.consume_tool_call()
        assert budget.any_exhausted is True

    def test_unlimited_iterations(self):
        budget = RunBudget(max_iterations=None)
        for _ in range(100):
            assert budget.consume_iteration() is True
        assert budget.iterations_exhausted is False

    def test_unlimited_tool_calls(self):
        budget = RunBudget(max_tool_calls=None)
        for _ in range(100):
            assert budget.consume_tool_call() is True
        assert budget.tool_calls_exhausted is False

    def test_repr(self):
        budget = RunBudget(max_iterations=10, max_tool_calls=20)
        r = repr(budget)
        assert "iterations=0/10" in r
        assert "tool_calls=0/20" in r


# ============================================================
# Helper: build messages list for react_loop
# ============================================================

def _build_messages(user_message: str, system_prompt: str = "You are helpful.") -> list:
    """Build a standard messages list for react_loop tests."""
    return [
        Message.system(system_prompt),
        Message.user(user_message),
    ]


# ============================================================
# ReAct loop tests
# ============================================================

class TestReActLoop:
    def _make_registry(self):
        registry = ToolRegistry()
        echo_spec = ToolSpec(
            name="echo",
            description="Echo text back",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            func=lambda text: text,
        )
        registry.register(echo_spec)
        return registry

    def test_single_turn_text_answer(self):
        """LLM returns text directly — no tool calls needed."""
        mock_llm = MockLLM([
            LLMResponse(content="The answer is 42."),
        ])
        registry = self._make_registry()
        messages = _build_messages("What is the answer?")

        result, _ = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
        )

        assert result == "The answer is 42."
        assert mock_llm.call_count == 1

    def test_single_tool_call(self):
        """LLM calls a tool, gets result, then gives final answer."""
        mock_llm = MockLLM([
            # First call: model decides to use echo tool
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="call_1", name="echo", arguments={"text": "Hello!"})
                ],
            ),
            # Second call: model sees result and answers
            LLMResponse(content="The echo tool returned: Hello!"),
        ])
        registry = self._make_registry()
        messages = _build_messages("Echo 'Hello!'")

        result, _ = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
        )

        assert result == "The echo tool returned: Hello!"
        assert mock_llm.call_count == 2

    def test_multiple_tool_calls_in_one_turn(self):
        """LLM calls multiple tools in a single response."""
        mock_llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="echo", arguments={"text": "first"}),
                    ToolCall(id="c2", name="echo", arguments={"text": "second"}),
                ],
            ),
            LLMResponse(content="Both tools executed successfully."),
        ])
        registry = self._make_registry()
        messages = _build_messages("Run two echos")

        result, _ = react_loop(llm=mock_llm, registry=registry, messages=messages)
        assert result == "Both tools executed successfully."
        assert mock_llm.call_count == 2

    def test_max_iterations_forced_end(self):
        """When max iterations reached, loop forces a final answer."""
        # Create a mock that always calls a tool (never ends naturally)
        responses = []
        for i in range(20):  # more than max_iterations
            responses.append(LLMResponse(
                content="",
                tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments={"text": f"msg{i}"})],
            ))
        mock_llm = MockLLM(responses)
        registry = self._make_registry()
        messages = _build_messages("Keep going forever")

        result, _ = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
            budget=RunBudget(max_iterations=5),
        )

        # Should have hit the iteration limit and forced a final answer
        assert mock_llm.call_count == 6  # 5 iterations + 1 final answer call
        assert result is not None

    def test_tool_budget_exhaustion(self):
        """When tool call budget exhausted, loop continues without tools."""
        mock_llm = MockLLM([
            # First iteration: calls tool
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
            ),
            # Second iteration: model sees budget exhausted message and gives answer
            LLMResponse(content="Based on what I have, the answer is ready."),
        ])
        registry = self._make_registry()
        messages = _build_messages("Echo something")

        result, _ = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
            budget=RunBudget(max_iterations=10, max_tool_calls=1),
        )

        # First tool call uses the budget, then model gets budget exhausted message
        assert mock_llm.call_count == 2
        assert isinstance(result, str)
        assert len(result) > 0

    def test_tool_budget_exhaustion_adds_tool_message(self):
        """Kernel layer fix: budget truncation补tool消息 — ensures matched tool_call/tool_result."""
        mock_llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="echo", arguments={"text": "a"}),
                    ToolCall(id="c2", name="echo", arguments={"text": "b"}),
                ],
            ),
            LLMResponse(content="Final answer with partial results."),
        ])
        registry = self._make_registry()
        messages = _build_messages("Call two tools")

        react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
            budget=RunBudget(max_iterations=10, max_tool_calls=1),
        )

        # The messages list should have: system, user, assistant(2 tool_calls),
        # tool_result(c1 executed), tool_result(c2 budget exhausted), system budget msg
        tool_msgs = [m for m in messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 2  # both tool calls have matching tool results
        # Second tool result should have budget exhausted content
        assert "budget exhausted" in tool_msgs[1].content.lower()

    def test_system_prompt_included(self):
        """System prompt is included in the initial messages."""
        mock_llm = MockLLM([LLMResponse(content="ok")])
        messages = _build_messages("Hi", system_prompt="You are a pirate.")

        result, _ = react_loop(
            llm=mock_llm,
            registry=self._make_registry(),
            messages=messages,
        )

        assert mock_llm.last_messages is not None
        assert mock_llm.last_messages[0].role == Role.SYSTEM
        assert mock_llm.last_messages[0].content == "You are a pirate."
        assert mock_llm.last_messages[1].role == Role.USER
        assert mock_llm.last_messages[1].content == "Hi"

    def test_message_history_grows(self):
        """Each iteration adds messages to the history (in-place append)."""
        mock_llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
            ),
            LLMResponse(content="Done!"),
        ])
        registry = self._make_registry()
        messages = _build_messages("Test")

        react_loop(mock_llm, registry, messages)

        # messages list should have grown: system + user + assistant + tool + assistant
        assert len(messages) >= 5
        assert messages[0].role == Role.SYSTEM
        assert messages[1].role == Role.USER
        assert messages[2].role == Role.ASSISTANT
        assert messages[3].role == Role.TOOL
        assert messages[-1].role == Role.ASSISTANT

    def test_messages_modified_in_place(self):
        """react_loop appends to the messages list in-place (same list object)."""
        mock_llm = MockLLM([LLMResponse(content="Hello!")])
        registry = self._make_registry()
        messages = _build_messages("Hi")
        original_id = id(messages)

        react_loop(mock_llm, registry, messages)

        assert id(messages) == original_id  # same list object
        assert len(messages) == 3  # system + user + assistant


# ============================================================
# finish_reason length retry tests
# ============================================================

class TestFinishReasonLengthRetry:
    """Tests for finish_reason='length' truncation retry in react_loop."""

    def test_length_truncation_triggers_retry(self):
        """When finish_reason is 'length', the loop retries with a continue prompt."""
        mock_llm = MockLLM([
            # First call: truncated response
            LLMResponse(
                content="The answer starts with ",
                finish_reason="length",
            ),
            # Retry call: continuation
            LLMResponse(content="42, which is correct."),
        ])
        registry = ToolRegistry()
        messages = _build_messages("What is the answer?")

        result, _ = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
            budget=RunBudget(max_iterations=5),
        )

        # Should have made 2 LLM calls (original + 1 retry)
        assert mock_llm.call_count == 2
        # Content should be combined
        assert "The answer starts with" in result
        assert "42, which is correct." in result

    def test_length_retry_max_one(self):
        """Max 1 retry — even if the retry also truncates, we don't retry again."""
        mock_llm = MockLLM([
            LLMResponse(content="Part 1: ", finish_reason="length"),
            LLMResponse(content="Part 2: ", finish_reason="length"),
            LLMResponse(content="Part 3: final"),
        ])
        registry = ToolRegistry()
        messages = _build_messages("Tell me a long story")

        result, _ = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
            budget=RunBudget(max_iterations=10),
        )

        # Only 1 retry allowed — call count should be 2 (not 3)
        assert mock_llm.call_count == 2
        # The second call's finish_reason=length but no more retries
        assert "Part 1:" in result
        assert "Part 2:" in result
        assert "Part 3" not in result

    def test_no_retry_when_stop_reason(self):
        """Normal finish_reason='stop' does NOT trigger retry."""
        mock_llm = MockLLM([
            LLMResponse(content="Complete answer.", finish_reason="stop"),
        ])
        registry = ToolRegistry()
        messages = _build_messages("Question")

        result, _ = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
        )

        assert mock_llm.call_count == 1
        assert result == "Complete answer."

    def test_no_retry_when_no_budget(self):
        """Don't retry if no iteration budget left."""
        mock_llm = MockLLM([
            LLMResponse(content="Truncated", finish_reason="length"),
        ])
        registry = ToolRegistry()
        messages = _build_messages("Hi")

        result, _ = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
            budget=RunBudget(max_iterations=1),
        )

        # Only 1 call — no budget for retry
        assert mock_llm.call_count == 1


# ============================================================
# Agent class tests
# ============================================================

class TestAgent:
    def _make_agent(self):
        mock_llm = MockLLM([LLMResponse(content="Hello!")])
        registry = ToolRegistry()
        return Agent(
            llm=mock_llm,
            registry=registry,
            system_prompt="You are helpful.",
            max_iterations=5,
        )

    def test_ask_returns_answer(self):
        agent = self._make_agent()
        answer = agent.ask("Hi")
        assert answer == "Hello!"

    def test_ask_verbose_override(self):
        agent = self._make_agent()
        # Should not crash with verbose=True
        answer = agent.ask("Hi", verbose=True)
        assert isinstance(answer, str)

    def test_list_tools_empty(self):
        agent = self._make_agent()
        assert agent.list_tools() == []

    def test_list_tools_with_tools(self):
        mock_llm = MockLLM([LLMResponse(content="ok")])
        registry = ToolRegistry()
        registry.register(ToolSpec(name="echo", description="Echo", func=lambda x: x))
        agent = Agent(llm=mock_llm, registry=registry)
        assert agent.list_tools() == ["echo"]

    def test_create_agent(self):
        from maque_core.config import AppConfig
        from maque_core.factory import create_agent
        config = AppConfig()
        mock_llm = MockLLM([LLMResponse(content="hi")])
        agent = create_agent(
            llm=mock_llm,
            system_prompt=config.agent.system_prompt,
            max_iterations=config.agent.max_iterations,
            max_tool_calls=config.agent.max_tool_calls,
            max_history_turns=config.agent.max_history_turns,
            include_reasoning_in_history=config.agent.include_reasoning_in_history,
            tools_package=config.tools_package,
        )
        assert isinstance(agent, Agent)
        # Should have discovered tools
        assert len(agent.list_tools()) >= 2

    def test_agent_preserves_config_settings(self):
        mock_llm = MockLLM([LLMResponse(content="ok")])
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            system_prompt="Custom prompt",
            max_iterations=3,
            max_tool_calls=5,
            verbose=False,
        )
        assert agent.system_prompt == "Custom prompt"
        assert agent.max_iterations == 3
        assert agent.max_tool_calls == 5

    def test_agent_has_messages_attribute(self):
        """Agent has a messages attribute initialized with system prompt."""
        agent = self._make_agent()
        assert hasattr(agent, "messages")
        assert isinstance(agent.messages, list)
        assert len(agent.messages) == 1
        assert agent.messages[0].role == Role.SYSTEM

    def test_agent_messages_grow_after_ask(self):
        """After ask(), messages list contains the full conversation."""
        agent = self._make_agent()
        initial_len = len(agent.messages)
        agent.ask("Hello")
        # Should have grown: system + user + assistant
        assert len(agent.messages) == initial_len + 2  # user + assistant

    def test_reset_clears_history(self):
        """reset() clears messages and re-adds system prompt."""
        mock_llm = MockLLM([
            LLMResponse(content="First answer"),
            LLMResponse(content="Second answer"),
        ])
        registry = ToolRegistry()
        agent = Agent(llm=mock_llm, registry=registry, system_prompt="Sys prompt")

        agent.ask("First question")
        assert len(agent.messages) > 1

        agent.reset()
        # After reset: only system prompt
        assert len(agent.messages) == 1
        assert agent.messages[0].role == Role.SYSTEM
        assert agent.messages[0].content == "Sys prompt"
        # latest_reasoning also reset
        assert agent.latest_reasoning is None

    def test_reset_then_ask_works(self):
        """After reset(), subsequent ask() works normally."""
        mock_llm = MockLLM([
            LLMResponse(content="Before reset"),
            LLMResponse(content="After reset"),
        ])
        registry = ToolRegistry()
        agent = Agent(llm=mock_llm, registry=registry)

        answer1 = agent.ask("Q1")
        assert answer1 == "Before reset"

        agent.reset()
        answer2 = agent.ask("Q2")
        assert answer2 == "After reset"
        # After reset + 1 ask: system + user + assistant = 3 messages
        assert len(agent.messages) == 3


# ============================================================
# Multi-turn conversation tests (P0-1 fix)
# ============================================================

class TestMultiTurnConversation:
    """Tests for multi-turn conversation memory — P0-1 fix."""

    def test_second_ask_sees_first_question(self):
        """Second ask() includes the first turn's messages in the LLM call."""
        mock_llm = MockLLM([
            LLMResponse(content="First answer"),
            LLMResponse(content="Second answer"),
        ])
        registry = ToolRegistry()
        agent = Agent(llm=mock_llm, registry=registry)

        agent.ask("First question")
        agent.ask("Second question")

        # The second LLM call should include the full conversation history
        assert mock_llm.call_count == 2
        second_call_msgs = mock_llm.all_messages[1]
        # Should have: system + user1 + assistant1 + user2
        assert len(second_call_msgs) >= 4
        assert second_call_msgs[0].role == Role.SYSTEM
        assert second_call_msgs[1].role == Role.USER
        assert second_call_msgs[1].content == "First question"
        assert second_call_msgs[2].role == Role.ASSISTANT
        assert second_call_msgs[2].content == "First answer"
        assert second_call_msgs[3].role == Role.USER
        assert second_call_msgs[3].content == "Second question"

    def test_two_round_qa_second_references_first(self):
        """Two-round Q&A — second round can reference info from first round."""
        mock_llm = MockLLM([
            LLMResponse(content="My favorite color is blue."),
            LLMResponse(content="As I said, my favorite color is blue."),
        ])
        registry = ToolRegistry()
        agent = Agent(llm=mock_llm, registry=registry)

        answer1 = agent.ask("What's your favorite color?")
        assert answer1 == "My favorite color is blue."

        answer2 = agent.ask("What did you just say?")
        assert answer2 == "As I said, my favorite color is blue."

        # Verify second call had access to first call's content
        second_msgs = mock_llm.all_messages[1]
        user_msgs = [m for m in second_msgs if m.role == Role.USER]
        assert len(user_msgs) == 2  # two user messages = two turns

    def test_multi_turn_with_tool_calls(self):
        """Multi-turn with tool calls — tool results persist in history."""
        mock_llm = MockLLM([
            # Turn 1: call echo tool
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="t1", name="echo", arguments={"text": "hello"})],
            ),
            LLMResponse(content="The echo said hello."),
            # Turn 2: answer based on history
            LLMResponse(content="Earlier I echoed 'hello'."),
        ])
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="echo",
            description="Echo text",
            func=lambda text: text,
        ))
        agent = Agent(llm=mock_llm, registry=registry)

        answer1 = agent.ask("Echo hello")
        assert answer1 == "The echo said hello."

        answer2 = agent.ask("What did we do earlier?")
        assert answer2 == "Earlier I echoed 'hello'."

        # Second turn's LLM call should see tool result from first turn
        third_call_msgs = mock_llm.all_messages[2]  # 3rd LLM call = start of turn 2
        tool_msgs = [m for m in third_call_msgs if m.role == Role.TOOL]
        assert len(tool_msgs) >= 1


# ============================================================
# History trimming tests
# ============================================================

class TestHistoryTrimming:
    """Tests for max_history_turns trimming strategy."""

    def test_under_limit_no_trimming(self):
        """When under max_history_turns, no trimming occurs."""
        mock_llm = MockLLM([
            LLMResponse(content="A1"),
            LLMResponse(content="A2"),
            LLMResponse(content="A3"),
        ])
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            max_history_turns=10,
        )

        agent.ask("Q1")
        agent.ask("Q2")
        agent.ask("Q3")

        # 3 turns, limit is 10 — all should be preserved
        user_msgs = [m for m in agent.messages if m.role == Role.USER]
        assert len(user_msgs) == 3

    def test_over_limit_trims_oldest_turns(self):
        """When over max_history_turns, oldest turns are removed."""
        responses = [LLMResponse(content=f"A{i}") for i in range(8)]
        mock_llm = MockLLM(responses)
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            max_history_turns=5,
        )

        for i in range(8):
            agent.ask(f"Q{i}")

        # Should have at most 5 user messages (the most recent 5)
        user_msgs = [m for m in agent.messages if m.role == Role.USER]
        assert len(user_msgs) <= 5
        # System prompt is always preserved
        assert agent.messages[0].role == Role.SYSTEM
        # The most recent messages should be the last few questions
        assert user_msgs[-1].content == "Q7"
        assert user_msgs[-2].content == "Q6"

    def test_system_prompt_always_preserved(self):
        """System prompt (index 0) is never trimmed."""
        responses = [LLMResponse(content=f"Ans{i}") for i in range(20)]
        mock_llm = MockLLM(responses)
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            system_prompt="KEEP ME",
            max_history_turns=3,
        )

        for i in range(15):
            agent.ask(f"Q{i}")

        assert agent.messages[0].role == Role.SYSTEM
        assert agent.messages[0].content == "KEEP ME"

    def test_trim_removes_complete_turns(self):
        """Trimming removes complete user-assistant pairs, not partial turns."""
        # Each "turn" is user + assistant (no tools for simplicity)
        responses = [LLMResponse(content=f"A{i}") for i in range(10)]
        mock_llm = MockLLM(responses)
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            max_history_turns=3,
        )

        for i in range(10):
            agent.ask(f"Q{i}")

        # After trimming, the structure should be consistent:
        # system, then alternating user-assistant pairs
        msgs = agent.messages
        assert msgs[0].role == Role.SYSTEM

        # Check that user messages are always followed by assistant messages
        # (no orphaned user or assistant messages)
        user_indices = [i for i, m in enumerate(msgs) if m.role == Role.USER]
        for idx in user_indices:
            if idx + 1 < len(msgs):
                # The message after a user message should be assistant or tool
                assert msgs[idx + 1].role in (Role.ASSISTANT, Role.TOOL)

    def test_max_history_zero_disabled(self):
        """max_history_turns=0 means no trimming (but system + current turn still work)."""
        responses = [LLMResponse(content=f"A{i}") for i in range(5)]
        mock_llm = MockLLM(responses)
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            max_history_turns=0,
        )

        for i in range(5):
            agent.ask(f"Q{i}")

        # With 0, trimming is disabled — all 5 turns kept
        user_msgs = [m for m in agent.messages if m.role == Role.USER]
        assert len(user_msgs) == 5

    def test_trim_happens_before_each_ask(self):
        """Trimming happens at the start of each ask(), not after."""
        responses = [LLMResponse(content=f"A{i}") for i in range(6)]
        mock_llm = MockLLM(responses)
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            max_history_turns=3,
        )

        # Fill up to exactly 3 turns
        for i in range(3):
            agent.ask(f"Q{i}")

        user_count_before = len([m for m in agent.messages if m.role == Role.USER])
        assert user_count_before == 3

        # 4th ask should trigger trimming before adding the new message
        agent.ask("Q3")
        user_count_after = len([m for m in agent.messages if m.role == Role.USER])
        # After trim + add, should still be at most max_history_turns
        assert user_count_after <= 3


# ============================================================
# Reasoning in history tests (P2-1 fix)
# ============================================================

class TestReasoningInHistory:
    """Tests for reasoning stored in message history — P2-1 fix."""

    def test_reasoning_stored_in_assistant_message(self):
        """When include_reasoning_in_history=True, reasoning is on the assistant message."""
        mock_llm = MockLLM([
            LLMResponse(
                content="The answer is 42.",
                reasoning="Let me think: 6*7=42",
            ),
        ])
        registry = ToolRegistry()
        messages = _build_messages("What is 6*7?")

        react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
            include_reasoning_in_history=True,
        )

        # Find the assistant message
        assistant_msgs = [m for m in messages if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[-1].reasoning == "Let me think: 6*7=42"

    def test_reasoning_not_stored_when_disabled(self):
        """When include_reasoning_in_history=False, reasoning is NOT on messages."""
        mock_llm = MockLLM([
            LLMResponse(
                content="Answer",
                reasoning="Secret thinking",
            ),
        ])
        registry = ToolRegistry()
        messages = _build_messages("Q")

        react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
            include_reasoning_in_history=False,
        )

        assistant_msgs = [m for m in messages if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[-1].reasoning is None

    def test_agent_passes_config_to_loop(self):
        """Agent passes include_reasoning_in_history config to react_loop."""
        mock_llm = MockLLM([
            LLMResponse(content="Hi", reasoning="Thinking"),
        ])
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            include_reasoning_in_history=True,
        )

        agent.ask("Hello")

        assistant_msgs = [m for m in agent.messages if m.role == Role.ASSISTANT]
        assert assistant_msgs[-1].reasoning == "Thinking"

    def test_agent_disabled_reasoning_not_in_history(self):
        """Agent with include_reasoning_in_history=False doesn't store reasoning."""
        mock_llm = MockLLM([
            LLMResponse(content="Hi", reasoning="Thinking"),
        ])
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            include_reasoning_in_history=False,
        )

        agent.ask("Hello")

        assistant_msgs = [m for m in agent.messages if m.role == Role.ASSISTANT]
        assert assistant_msgs[-1].reasoning is None
        # But latest_reasoning still works
        assert agent.latest_reasoning == "Thinking"

    def test_latest_reasoning_always_works(self):
        """latest_reasoning always points to last round's reasoning regardless of config."""
        mock_llm = MockLLM([
            LLMResponse(content="A1", reasoning="Thought 1"),
            LLMResponse(content="A2", reasoning="Thought 2"),
        ])
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            include_reasoning_in_history=False,
        )

        agent.ask("Q1")
        assert agent.latest_reasoning == "Thought 1"

        agent.ask("Q2")
        assert agent.latest_reasoning == "Thought 2"

    def test_reasoning_trimmed_with_messages(self):
        """Reasoning content is trimmed along with its message during history trimming."""
        responses = [
            LLMResponse(content=f"A{i}", reasoning=f"Thought {i}")
            for i in range(10)
        ]
        mock_llm = MockLLM(responses)
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            max_history_turns=3,
            include_reasoning_in_history=True,
        )

        for i in range(10):
            agent.ask(f"Q{i}")

        # All remaining assistant messages should have reasoning
        assistant_msgs = [m for m in agent.messages if m.role == Role.ASSISTANT]
        for msg in assistant_msgs:
            assert msg.reasoning is not None
            assert msg.reasoning.startswith("Thought")

    def test_multi_turn_reasoning_visible_to_next_turn(self):
        """In multi-turn, reasoning from turn 1 is visible to the LLM in turn 2."""
        mock_llm = MockLLM([
            LLMResponse(content="A1", reasoning="Round 1 reasoning"),
            LLMResponse(content="A2", reasoning="Round 2 reasoning"),
        ])
        registry = ToolRegistry()
        agent = Agent(
            llm=mock_llm,
            registry=registry,
            include_reasoning_in_history=True,
        )

        agent.ask("Q1")
        agent.ask("Q2")

        # Second LLM call's messages should include first turn's reasoning
        second_call_msgs = mock_llm.all_messages[1]
        assistant_msgs = [m for m in second_call_msgs if m.role == Role.ASSISTANT]
        # The first assistant message (from turn 1) should have reasoning
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[0].reasoning == "Round 1 reasoning"


# ============================================================
# Reasoning tests — latest_reasoning in react_loop and Agent
# ============================================================

class TestReactLoopReasoning:
    """Tests for latest_reasoning returned by react_loop."""

    def test_no_reasoning_returns_none(self):
        """When LLM responses have no reasoning, latest_reasoning is None."""
        mock_llm = MockLLM([
            LLMResponse(content="The answer is 42."),
        ])
        registry = ToolRegistry()
        messages = _build_messages("What is the answer?")

        answer, reasoning = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
        )

        assert answer == "The answer is 42."
        assert reasoning is None

    def test_single_call_with_reasoning(self):
        """Single LLM call with reasoning — latest_reasoning contains it."""
        mock_llm = MockLLM([
            LLMResponse(
                content="The answer is 42.",
                reasoning="Let me think... 6 * 7 = 42.",
            ),
        ])
        registry = ToolRegistry()
        messages = _build_messages("What is 6 * 7?")

        answer, reasoning = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
        )

        assert answer == "The answer is 42."
        assert reasoning == "Let me think... 6 * 7 = 42."

    def test_reasoning_overwrites_each_iteration(self):
        """Multi-round: latest_reasoning only keeps the LAST round's reasoning."""
        mock_llm = MockLLM([
            LLMResponse(
                content="",
                reasoning="First round thinking: I need to call the echo tool.",
                tool_calls=[
                    ToolCall(id="tc1", name="echo", arguments={"text": "hello"})
                ],
            ),
            LLMResponse(
                content="The result is hello.",
                reasoning="Second round thinking: Now I can answer.",
            ),
        ])
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="echo",
            description="Echo back text",
            func=lambda text: text,
        ))
        messages = _build_messages("Echo hello")

        answer, reasoning = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
        )

        assert answer == "The result is hello."
        # Only the LAST round's reasoning should be kept
        assert reasoning == "Second round thinking: Now I can answer."
        # First round reasoning should NOT be present
        assert "First round" not in reasoning

    def test_three_iterations_only_last_kept(self):
        """Three LLM calls — only the third (last) reasoning is kept."""
        mock_llm = MockLLM([
            LLMResponse(
                content="",
                reasoning="Round 1: need to call add(2, 3)",
                tool_calls=[
                    ToolCall(id="tc1", name="add", arguments={"a": 2, "b": 3})
                ],
            ),
            LLMResponse(
                content="",
                reasoning="Round 2: got 5, now multiply by 4",
                tool_calls=[
                    ToolCall(id="tc2", name="multiply", arguments={"a": 5, "b": 4})
                ],
            ),
            LLMResponse(
                content="The final result is 20.",
                reasoning="Round 3: final answer is 20",
            ),
        ])
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="add",
            description="Add two numbers",
            func=lambda a, b: a + b,
        ))
        registry.register(ToolSpec(
            name="multiply",
            description="Multiply two numbers",
            func=lambda a, b: a * b,
        ))
        messages = _build_messages("Calculate (2+3)*4")

        answer, reasoning = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
        )

        assert answer == "The final result is 20."
        assert reasoning == "Round 3: final answer is 20"
        assert "Round 1" not in reasoning
        assert "Round 2" not in reasoning

    def test_reasoning_not_in_content_when_in_history(self):
        """Reasoning goes to message.reasoning, not into content."""
        mock_llm = MockLLM([
            LLMResponse(
                content="",
                reasoning="I should call the echo tool.",
                tool_calls=[
                    ToolCall(id="tc1", name="echo", arguments={"text": "hi"})
                ],
            ),
            LLMResponse(
                content="Done.",
                reasoning="Final reasoning.",
            ),
        ])
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="echo",
            description="Echo",
            func=lambda text: text,
        ))
        messages = _build_messages("Echo hi")

        react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
            include_reasoning_in_history=True,
        )

        # Reasoning should be in the reasoning field, not in content
        assistant_msgs = [m for m in messages if m.role == Role.ASSISTANT]
        for msg in assistant_msgs:
            assert "I should call the echo tool." not in msg.content
            assert "Final reasoning." not in msg.content
        # But should be in the reasoning attribute
        assert assistant_msgs[-1].reasoning == "Final reasoning."

    def test_mixed_reasoning_and_none(self):
        """Some rounds have reasoning, some don't — last non-None is kept."""
        mock_llm = MockLLM([
            LLMResponse(
                content="",
                reasoning="Round 1 thinking",
                tool_calls=[
                    ToolCall(id="tc1", name="echo", arguments={"text": "x"})
                ],
            ),
            LLMResponse(
                content="Done.",
                reasoning=None,  # no reasoning this round
            ),
        ])
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="echo",
            description="Echo",
            func=lambda text: text,
        ))
        messages = _build_messages("Echo x")

        answer, reasoning = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=messages,
        )

        # Last round had None reasoning, but we keep the last non-None value
        assert reasoning == "Round 1 thinking"


class TestAgentReasoning:
    """Tests for Agent.latest_reasoning attribute."""

    def test_agent_starts_with_none_reasoning(self):
        """New agent has latest_reasoning = None."""
        mock_llm = MockLLM([LLMResponse(content="hi")])
        registry = ToolRegistry()
        agent = Agent(llm=mock_llm, registry=registry)
        assert agent.latest_reasoning is None

    def test_ask_updates_latest_reasoning(self):
        """After ask(), latest_reasoning contains the LLM's reasoning."""
        mock_llm = MockLLM([
            LLMResponse(
                content="The answer is 42.",
                reasoning="Let me calculate: 6 * 7 = 42",
            ),
        ])
        registry = ToolRegistry()
        agent = Agent(llm=mock_llm, registry=registry)

        answer = agent.ask("What is 6*7?")

        assert answer == "The answer is 42."
        assert agent.latest_reasoning == "Let me calculate: 6 * 7 = 42"

    def test_ask_overwrites_previous_reasoning(self):
        """Second ask() overwrites the first ask()'s reasoning."""
        mock_llm = MockLLM([
            LLMResponse(content="First answer", reasoning="First thought"),
            LLMResponse(content="Second answer", reasoning="Second thought"),
        ])
        registry = ToolRegistry()
        agent = Agent(llm=mock_llm, registry=registry)

        agent.ask("First question")
        assert agent.latest_reasoning == "First thought"

        agent.ask("Second question")
        assert agent.latest_reasoning == "Second thought"
        assert agent.latest_reasoning != "First thought"

    def test_ask_with_tool_calls_keeps_latest(self):
        """Multi-tool-call ask — only last round's reasoning kept."""
        mock_llm = MockLLM([
            LLMResponse(
                content="",
                reasoning="I need to use echo",
                tool_calls=[
                    ToolCall(id="tc1", name="echo", arguments={"text": "test"})
                ],
            ),
            LLMResponse(
                content="Echoed: test",
                reasoning="Final answer time",
            ),
        ])
        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="echo",
            description="Echo text",
            func=lambda text: text,
        ))
        agent = Agent(llm=mock_llm, registry=registry)

        answer = agent.ask("Echo test")

        assert answer == "Echoed: test"
        assert agent.latest_reasoning == "Final answer time"
        assert "I need to use echo" not in agent.latest_reasoning

    def test_reset_clears_latest_reasoning(self):
        """reset() clears latest_reasoning back to None."""
        mock_llm = MockLLM([
            LLMResponse(content="Hi", reasoning="Thinking"),
            LLMResponse(content="Hi again", reasoning="Thinking again"),
        ])
        registry = ToolRegistry()
        agent = Agent(llm=mock_llm, registry=registry)

        agent.ask("Hello")
        assert agent.latest_reasoning == "Thinking"

        agent.reset()
        assert agent.latest_reasoning is None

        agent.ask("Hello again")
        assert agent.latest_reasoning == "Thinking again"
