"""End-to-end demo of the maque_core framework.

Shows the agent working with built-in tools using a mock LLM.
No API key required — perfect for verifying your installation.
"""

from __future__ import annotations

import sys
import os

# Make sure we can import maque_core from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maque_core.config import AppConfig
from maque_core.contracts.messages import LLMResponse, ToolCall
from maque_core.contracts.tool import ToolSpec
from maque_core.kernel import Agent
from maque_core.llm import MockLLM
from maque_core.tools import ToolRegistry, discover_tools


def build_demo_agent() -> Agent:
    """Build an agent with mock LLM and discovered tools."""
    config = AppConfig()

    # Set up tool registry with discovered tools
    registry = ToolRegistry()
    discover_tools(registry, config.tools_package)

    # Create mock LLM with a demo conversation
    mock_llm = MockLLM([
        # First call: model decides to use the echo tool
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_001",
                    name="echo",
                    arguments={"text": "Hello from maque_core!"},
                )
            ],
        ),
        # Second call: model sees the tool result and gives final answer
        LLMResponse(
            content="The echo tool returned: 'Hello from maque_core!'. "
                    "This demonstrates that the agent can use tools to get things done. "
                    "Try adding your own tools with the @tool decorator!",
        ),
    ])

    return Agent(
        llm=mock_llm,
        registry=registry,
        system_prompt=config.agent.system_prompt,
        max_iterations=config.agent.max_iterations,
        verbose=True,
    )


def main() -> None:
    print("=" * 60)
    print("  maque_core Framework — End-to-End Demo")
    print("=" * 60)

    config = AppConfig()
    print(f"\nConfiguration:")
    print(f"  Model: {config.llm.model}")
    print(f"  Base URL: {config.llm.base_url}")
    print(f"  Max iterations: {config.agent.max_iterations}")
    print(f"  Tools package: {config.tools_package}")

    agent = build_demo_agent()

    print(f"\nAvailable tools: {agent.list_tools()}")
    print(f"\n{'─' * 60}")
    print("Demo 1: Echo tool (single tool call)")
    print(f"{'─' * 60}")

    answer = agent.ask("Can you echo 'Hello from maque_core!'?")
    print(f"\nFinal answer: {answer}")

    # Demo 2: Multi-turn with add tool
    print(f"\n{'─' * 60}")
    print("Demo 2: Add tool (math calculation)")
    print(f"{'─' * 60}")

    mock_llm2 = MockLLM([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="call_002", name="add", arguments={"a": 42, "b": 58}),
            ],
        ),
        LLMResponse(
            content="42 + 58 = 100. The add tool confirms this calculation.",
        ),
    ])

    agent2 = Agent(
        llm=mock_llm2,
        registry=agent.registry,
        system_prompt=config.agent.system_prompt,
        verbose=True,
    )

    answer2 = agent2.ask("What is 42 plus 58?")
    print(f"\nFinal answer: {answer2}")

    # Demo 3: Direct text answer (no tools needed)
    print(f"\n{'─' * 60}")
    print("Demo 3: Direct answer (no tools needed)")
    print(f"{'─' * 60}")

    mock_llm3 = MockLLM([
        LLMResponse(content="The capital of France is Paris."),
    ])

    agent3 = Agent(
        llm=mock_llm3,
        registry=agent.registry,
        system_prompt=config.agent.system_prompt,
        verbose=True,
    )

    answer3 = agent3.ask("What is the capital of France?")
    print(f"\nFinal answer: {answer3}")

    # Demo 4: Reasoning (chain-of-thought) display
    print(f"\n{'─' * 60}")
    print("Demo 4: Reasoning (chain-of-thought)")
    print(f"{'─' * 60}")

    mock_llm4 = MockLLM([
        LLMResponse(
            content="",
            reasoning="Let me think... The user wants to calculate (15 + 25) * 2. "
                      "First I need to add 15 and 25 using the add tool, "
                      "then multiply the result by 2.",
            tool_calls=[
                ToolCall(id="call_r1", name="add", arguments={"a": 15, "b": 25}),
            ],
        ),
        LLMResponse(
            content="",
            reasoning="The add tool returned 40. Now I need to multiply 40 by 2 "
                      "using the multiply tool.",
            tool_calls=[
                ToolCall(id="call_r2", name="multiply", arguments={"a": 40, "b": 2}),
            ],
        ),
        LLMResponse(
            content="(15 + 25) * 2 = 80",
            reasoning="The multiply tool returned 80. That's the final answer. "
                      "Let me double-check: 15+25=40, 40*2=80. Correct.",
        ),
    ])

    agent4 = Agent(
        llm=mock_llm4,
        registry=agent.registry,
        system_prompt=config.agent.system_prompt,
        verbose=True,
    )

    agent4.registry.register(ToolSpec(
        name="multiply",
        description="Multiply two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
        func=lambda a, b: a * b,
    ))

    answer4 = agent4.ask("What is (15 + 25) * 2?")
    print(f"\nFinal answer: {answer4}")
    print(f"\nLatest reasoning (only the last round is kept):")
    print(f"  {agent4.latest_reasoning}")
    print(f"\nNote: latest_reasoning contains ONLY the most recent LLM call's")
    print(f"      thinking content — earlier rounds' reasoning is not accumulated.")

    print(f"\n{'=' * 60}")
    print("  Demo complete! All tools working correctly.")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Add your own tools in maque_core/tools/impl/")
    print("  2. Set MAQUE_LLM_API_KEY to use a real LLM")
    print("  3. Run 'python app.py -i' for interactive mode")
    print("  4. Run 'pytest tests/' to run all tests")
    print("  5. Access agent.latest_reasoning after each ask() for chain-of-thought")


if __name__ == "__main__":
    main()
