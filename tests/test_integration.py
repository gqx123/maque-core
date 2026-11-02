"""Integration tests — full end-to-end scenarios across all modules."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from maque_core.config import AppConfig
from maque_core.contracts.messages import LLMResponse, Message, ToolCall, Role
from maque_core.contracts.tool import ToolSpec
from maque_core.kernel import Agent, react_loop, RunBudget
from maque_core.llm import MockLLM
from maque_core.tools import ToolRegistry, discover_tools, tool


# ============================================================
# Integration test 1: Full tool call → answer flow
# ============================================================

class TestFullFlow:
    def test_tool_call_then_answer(self):
        """Complete flow: user asks → LLM calls tool → result → LLM answers."""
        mock_llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="call_001", name="echo", arguments={"text": "Hello World"})
                ],
            ),
            LLMResponse(content="The echo tool returned: Hello World"),
        ])

        registry = ToolRegistry()
        discover_tools(registry, "maque_core.tools.impl")

        answer, _ = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=[
                Message.system("You are a helpful assistant."),
                Message.user("Echo 'Hello World'"),
            ],
        )

        assert answer == "The echo tool returned: Hello World"
        assert mock_llm.call_count == 2

        # Verify the conversation history
        last_msgs = mock_llm.last_messages
        assert len(last_msgs) == 4  # system + user + assistant(tool_call) + tool_result
        assert last_msgs[0].role == Role.SYSTEM
        assert last_msgs[1].role == Role.USER
        assert last_msgs[2].role == Role.ASSISTANT
        assert len(last_msgs[2].tool_calls) == 1
        assert last_msgs[3].role == Role.TOOL
        assert last_msgs[3].tool_name == "echo"


# ============================================================
# Integration test 2: Echo tool integration
# ============================================================

class TestEchoToolIntegration:
    def test_echo_tool_discovered(self):
        """Echo tool should be discoverable from the impl package."""
        registry = ToolRegistry()
        count = discover_tools(registry, "maque_core.tools.impl")
        assert count >= 2
        assert "echo" in registry.tool_names

    def test_echo_tool_executes(self):
        """Echo tool should return the input text."""
        registry = ToolRegistry()
        discover_tools(registry, "maque_core.tools.impl")

        obs = registry.execute("echo", {"text": "integration test"}, "tc1")
        assert obs.is_error is False
        assert obs.content == "integration test"

    def test_echo_via_react_loop(self):
        """Echo tool should work end-to-end through the ReAct loop."""
        mock_llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="echo", arguments={"text": "ReAct test"})
                ],
            ),
            LLMResponse(content="Echo successful"),
        ])

        registry = ToolRegistry()
        discover_tools(registry, "maque_core.tools.impl")

        messages = [Message.system("You are helpful."), Message.user("Test echo")]
        result, _ = react_loop(mock_llm, registry, messages)
        assert result == "Echo successful"


# ============================================================
# Integration test 3: Unknown tool handling
# ============================================================

class TestUnknownToolIntegration:
    def test_unknown_tool_returns_error(self):
        """Calling an unknown tool should return an error observation."""
        mock_llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="nonexistent_tool", arguments={})
                ],
            ),
            LLMResponse(content="I don't have access to that tool."),
        ])

        registry = ToolRegistry()
        discover_tools(registry, "maque_core.tools.impl")

        messages = [Message.system("You are helpful."), Message.user("Use the nonexistent tool")]
        result, _ = react_loop(mock_llm, registry, messages)
        # The model should see the error and recover
        assert isinstance(result, str)
        assert mock_llm.call_count == 2

        # Check that the tool error message was passed back
        last_msgs = mock_llm.last_messages
        tool_msgs = [m for m in last_msgs if m.role == Role.TOOL]
        assert len(tool_msgs) == 1


# ============================================================
# Integration test 4: Parameter validation error recovery
# ============================================================

class TestValidationErrorIntegration:
    def test_validation_error_recovery(self):
        """When a tool call has invalid params, model should see error and can retry."""
        mock_llm = MockLLM([
            # First call: missing required param
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="echo", arguments={})  # missing 'text'
                ],
            ),
            # Second call: corrected params
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc2", name="echo", arguments={"text": "fixed"})
                ],
            ),
            # Third call: final answer
            LLMResponse(content="The result is: fixed"),
        ])

        registry = ToolRegistry()
        discover_tools(registry, "maque_core.tools.impl")

        messages = [Message.system("You are helpful."), Message.user("Echo something")]
        result, _ = react_loop(mock_llm, registry, messages)
        assert result == "The result is: fixed"
        assert mock_llm.call_count == 3

        # Should have 2 tool messages: one error, one success
        last_msgs = mock_llm.last_messages
        tool_msgs = [m for m in last_msgs if m.role == Role.TOOL]
        assert len(tool_msgs) == 2


# ============================================================
# Integration test 5: Multiple tools sequence
# ============================================================

class TestMultipleToolsIntegration:
    def test_add_then_echo(self):
        """Agent uses add tool then echo tool in sequence."""
        mock_llm = MockLLM([
            # First: call add
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="add", arguments={"a": 10, "b": 20})
                ],
            ),
            # Second: call echo with the result
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="tc2", name="echo", arguments={"text": "30"})
                ],
            ),
            # Third: final answer
            LLMResponse(content="10 + 20 = 30, and echo confirms: 30"),
        ])

        registry = ToolRegistry()
        discover_tools(registry, "maque_core.tools.impl")

        messages = [Message.system("You are helpful."), Message.user("Add 10 and 20 then echo the result")]
        result, _ = react_loop(mock_llm, registry, messages)
        assert result == "10 + 20 = 30, and echo confirms: 30"
        assert mock_llm.call_count == 3


# ============================================================
# Integration test 6: Budget protection
# ============================================================

class TestBudgetIntegration:
    def test_tool_budget_protects_against_loop(self):
        """Agent with tool budget=1 should not make more than 1 tool call."""
        # Mock LLM that keeps calling tools (would loop forever without budget)
        responses = []
        for i in range(10):
            responses.append(LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id=f"tc{i}", name="echo", arguments={"text": f"msg{i}"})
                ],
            ))
        # Add a final text response for when budget exhausts
        responses.append(LLMResponse(content="Budget exhausted, stopping."))

        mock_llm = MockLLM(responses)

        registry = ToolRegistry()
        discover_tools(registry, "maque_core.tools.impl")

        result, _ = react_loop(
            llm=mock_llm,
            registry=registry,
            messages=[Message.system("You are helpful."), Message.user("Keep echoing forever")],
            budget=RunBudget(max_iterations=10, max_tool_calls=1),
        )

        # Should have made limited number of calls
        assert mock_llm.call_count <= 5  # 1 tool call + budget exhausted handling + final
        assert isinstance(result, str)


# ============================================================
# Integration test 7: Discovery + Agent construction
# ============================================================

class TestAgentConstructionIntegration:
    def test_create_agent_with_discovery(self):
        """create_agent should discover tools and create a working agent."""
        from maque_core.factory import create_agent
        config = AppConfig()
        mock_llm = MockLLM([LLMResponse(content="Hello from config agent!")])

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
        assert len(agent.list_tools()) >= 2  # echo + add
        assert "echo" in agent.list_tools()
        assert "add" in agent.list_tools()

        # Should be able to ask a question
        answer = agent.ask("Hi")
        assert answer == "Hello from config agent!"

    def test_full_demo_scenario(self):
        """Simulate the full demo.py scenario."""
        config = AppConfig()
        registry = ToolRegistry()
        discover_tools(registry, config.tools_package)

        mock_llm = MockLLM([
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="demo_1", name="echo", arguments={"text": "Hello from maque_core!"})
                ],
            ),
            LLMResponse(
                content="The echo tool returned: 'Hello from maque_core!'. "
                        "This demonstrates that the agent can use tools.",
            ),
        ])

        agent = Agent(
            llm=mock_llm,
            registry=registry,
            system_prompt=config.agent.system_prompt,
            max_iterations=config.agent.max_iterations,
        )

        answer = agent.ask("Can you echo 'Hello from maque_core!'?")
        assert "Hello from maque_core" in answer
        assert mock_llm.call_count == 2



# ============================================================
# Integration test: multi-round reasoning only keeps latest
# ============================================================

class TestMultiRoundReasoning:
    """Integration tests verifying latest_reasoning only keeps the last round."""

    def test_three_tool_calls_only_last_reasoning_kept(self):
        """Full scenario: 3 LLM rounds with 2 tool calls → only last reasoning.

        This is the key integration test that verifies the requirement:
        "一次多轮工具调用的场景中，latest_reasoning 确实只保留最后一轮的思考内容"
        """
        mock_llm = MockLLM([
            # Round 1: calls add tool
            LLMResponse(
                content="",
                reasoning="【第一轮思考】用户要计算 (10 + 5) * 3，我需要先调用 add 工具计算 10 + 5。",
                tool_calls=[
                    ToolCall(id="call_add", name="add", arguments={"a": 10, "b": 5})
                ],
            ),
            # Round 2: calls multiply tool
            LLMResponse(
                content="",
                reasoning="【第二轮思考】add 返回了 15，现在调用 multiply 计算 15 * 3。",
                tool_calls=[
                    ToolCall(id="call_mul", name="multiply", arguments={"a": 15, "b": 3})
                ],
            ),
            # Round 3: final answer
            LLMResponse(
                content="(10 + 5) * 3 = 45",
                reasoning="【第三轮思考】multiply 返回了 45，这就是最终答案。",
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

        agent = Agent(
            llm=mock_llm,
            registry=registry,
            system_prompt="You are a math assistant.",
        )

        answer = agent.ask("What is (10 + 5) * 3?")

        # Answer is correct
        assert answer == "(10 + 5) * 3 = 45"
        assert mock_llm.call_count == 3

        # CRITICAL: latest_reasoning only contains the LAST round's reasoning
        latest = agent.latest_reasoning
        assert latest is not None
        assert "第三轮思考" in latest
        assert "最终答案" in latest

        # Earlier rounds' reasoning must NOT be in latest_reasoning
        assert "第一轮思考" not in latest
        assert "第二轮思考" not in latest

    def test_reasoning_not_accumulated_in_messages(self):
        """Verify reasoning content never leaks into conversation history."""
        reasoning_text_1 = "Secret thought number one: use echo"
        reasoning_text_2 = "Secret thought number two: final answer"

        mock_llm = MockLLM([
            LLMResponse(
                content="",
                reasoning=reasoning_text_1,
                tool_calls=[
                    ToolCall(id="tc1", name="echo", arguments={"text": "hello"})
                ],
            ),
            LLMResponse(
                content="The echo said hello.",
                reasoning=reasoning_text_2,
            ),
        ])

        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="echo",
            description="Echo back text",
            func=lambda text: text,
        ))

        agent = Agent(llm=mock_llm, registry=registry)
        agent.ask("Echo hello")

        # Check all message batches sent to LLM
        for msg_batch in mock_llm.all_messages:
            for msg in msg_batch:
                content = str(msg.content)
                # Reasoning text must never appear in message content
                assert reasoning_text_1 not in content, (
                    f"Reasoning leaked into messages: {reasoning_text_1[:50]}"
                )
                assert reasoning_text_2 not in content, (
                    f"Reasoning leaked into messages: {reasoning_text_2[:50]}"
                )

    def test_agent_reasoning_persists_after_ask(self):
        """latest_reasoning is available after ask() returns."""
        mock_llm = MockLLM([
            LLMResponse(
                content="The capital of France is Paris.",
                reasoning="Let me recall... France's capital is Paris.",
            ),
        ])
        registry = ToolRegistry()
        agent = Agent(llm=mock_llm, registry=registry)

        answer = agent.ask("What is the capital of France?")

        # Both answer and reasoning are accessible
        assert answer == "The capital of France is Paris."
        assert agent.latest_reasoning == "Let me recall... France's capital is Paris."
        # Can be read multiple times
        assert agent.latest_reasoning == "Let me recall... France's capital is Paris."
