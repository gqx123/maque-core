"""maque_core CLI — command-line interface for the agent framework.

Usage:
    python app.py "What is the weather?"
    python app.py --interactive
    python app.py --list-tools
    python app.py --verbose "Tell me a joke"
"""

from __future__ import annotations

import argparse
import sys

from maque_core.config import AppConfig
from maque_core.factory import create_agent
from maque_core.kernel import Agent, AgentRunError
from maque_core.llm import OpenAICompatibleProvider


def build_agent(config: AppConfig) -> Agent:
    """Build an Agent instance from config."""
    llm = OpenAICompatibleProvider(
        model=config.llm.model,
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
        temperature=config.llm.temperature,
        timeout=config.llm.timeout,
        max_tokens=config.llm.max_tokens,
    )
    return create_agent(
        llm=llm,
        system_prompt=config.agent.system_prompt,
        max_iterations=config.agent.max_iterations,
        max_tool_calls=config.agent.max_tool_calls,
        max_history_turns=config.agent.max_history_turns,
        include_reasoning_in_history=config.agent.include_reasoning_in_history,
        tools_package=config.tools_package,
    )


def cmd_list_tools(agent: Agent) -> None:
    """List all available tools."""
    tools = agent.list_tools()
    if not tools:
        print("No tools registered.")
    else:
        print(f"Available tools ({len(tools)}):")
        for name in tools:
            print(f"  - {name}")


def cmd_ask(agent: Agent, question: str, verbose: bool) -> None:
    """Ask a single question and print the answer."""
    try:
        answer = agent.ask(question, verbose=verbose)
        print(answer)
    except AgentRunError as e:
        print(f"[error] {e.message}", file=sys.stderr)
        sys.exit(1)


def cmd_interactive(agent: Agent, verbose: bool) -> None:
    """Run in interactive (REPL) mode with multi-turn conversation support.

    The agent maintains conversation history across inputs.
    Use "/reset" to start a fresh conversation.
    """
    print("maque_core interactive mode. Type 'exit' or 'quit' to end.")
    print("Type '/reset' to start a new conversation, '/tools' to list tools.")
    print("-" * 50)
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "/tools":
            cmd_list_tools(agent)
            continue
        if question.lower() == "/reset":
            agent.reset()
            print("Conversation reset. Starting fresh.")
            continue

        print("\nAgent: ", end="", flush=True)
        try:
            answer = agent.ask(question, verbose=verbose)
            print(answer)
        except AgentRunError as e:
            # Print error to stderr but don't break the loop —
            # the user can continue the conversation.
            print(f"[error] {e.message}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="maque_core",
        description="maque_core — a lightweight, extensible Agent framework",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask the agent (single-shot mode)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive (REPL) mode",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all available tools and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (show each step)",
    )
    parser.add_argument(
        "--model",
        help="Override the LLM model name",
    )
    parser.add_argument(
        "--base-url",
        help="Override the LLM API base URL",
    )
    parser.add_argument(
        "--api-key",
        help="Override the LLM API key",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum number of reasoning iterations",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        help="Maximum number of tool calls per run",
    )

    args = parser.parse_args()

    # Build config from environment + overrides
    config = AppConfig.from_env()
    if args.model:
        config.llm.model = args.model
    if args.base_url:
        config.llm.base_url = args.base_url
    if args.api_key:
        config.llm.api_key = args.api_key
    if args.max_iterations:
        config.agent.max_iterations = args.max_iterations
    if args.max_tool_calls:
        config.agent.max_tool_calls = args.max_tool_calls
    if args.verbose:
        config.agent.verbose = True

    agent = build_agent(config)

    if args.list_tools:
        cmd_list_tools(agent)
        return

    if args.interactive:
        cmd_interactive(agent, args.verbose)
        return

    if args.question:
        cmd_ask(agent, args.question, args.verbose)
        return

    # No question and no mode — show help
    parser.print_help()


if __name__ == "__main__":
    main()
