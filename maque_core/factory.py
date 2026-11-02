"""Agent factory — convenient entry point for creating agents.

Provides ``create_agent`` which builds an :class:`Agent` from
explicit parameters, environment variables (via ``AppConfig.from_env``),
or a pre-built LLM provider.
"""

from __future__ import annotations

from .config import AppConfig
from .kernel.agent import Agent
from .llm.openai_compatible import OpenAICompatibleProvider
from .tools.discovery import discover_tools
from .tools.registry import ToolRegistry


def create_agent(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    max_iterations: int | None = None,
    max_tool_calls: int | None = None,
    max_history_turns: int | None = None,
    include_reasoning_in_history: bool | None = None,
    tools_package: str | list[str] | None = None,
    llm=None,
) -> Agent:
    """Create an Agent from explicit parameters and environment defaults.

    Parameters
    ----------
    model:
        Model name. Overrides the environment default when provided.
    base_url:
        Base URL for the OpenAI-compatible endpoint.
    api_key:
        API key for the LLM endpoint.
    system_prompt:
        System prompt to prepend to the message history.
    max_iterations:
        Maximum ReAct iterations per ``ask()`` call.
    max_tool_calls:
        Maximum tool calls per ``ask()`` call.
    max_history_turns:
        Maximum conversation turns to keep in history.
    include_reasoning_in_history:
        Whether to include reasoning content in the message history.
    tools_package:
        Package name(s) to scan for ``@tool``-decorated functions.
        ``None`` means no tools are loaded by default.
        A string loads one package; a list loads multiple packages
        into the same registry.
    llm:
        Pre-built LLM provider. When provided, ``model``/``base_url``/
        ``api_key`` are ignored and this provider is used directly.

    Returns
    -------
    Agent
        A configured Agent instance ready to use.
    """
    config = AppConfig.from_env()

    # LLM
    if llm is None:
        llm = OpenAICompatibleProvider(
            model=model if model is not None else config.llm.model,
            base_url=base_url if base_url is not None else config.llm.base_url,
            api_key=api_key if api_key is not None else config.llm.api_key,
            temperature=config.llm.temperature,
            timeout=config.llm.timeout,
            max_tokens=config.llm.max_tokens,
        )

    # Tools
    registry = ToolRegistry()
    if tools_package is not None:
        if isinstance(tools_package, str):
            discover_tools(registry, tools_package)
        else:
            for pkg in tools_package:
                discover_tools(registry, pkg)

    # Config overrides
    resolved_system_prompt = system_prompt if system_prompt is not None else config.agent.system_prompt
    resolved_max_iterations = max_iterations if max_iterations is not None else config.agent.max_iterations
    resolved_max_tool_calls = max_tool_calls if max_tool_calls is not None else config.agent.max_tool_calls
    resolved_max_history = max_history_turns if max_history_turns is not None else config.agent.max_history_turns
    resolved_include_reasoning = (
        include_reasoning_in_history
        if include_reasoning_in_history is not None
        else config.agent.include_reasoning_in_history
    )

    return Agent(
        llm=llm,
        registry=registry,
        system_prompt=resolved_system_prompt,
        max_iterations=resolved_max_iterations,
        max_tool_calls=resolved_max_tool_calls,
        max_history_turns=resolved_max_history,
        include_reasoning_in_history=resolved_include_reasoning,
    )
