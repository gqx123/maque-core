"""Tests for the public API surface of maque_core."""

import maque_core

EXPECTED = {
    "create_agent", "Agent", "AgentRunError", "tool",
    "LLMProvider", "Message", "Role", "ToolCall", "LLMResponse",
    "ToolSpec", "ToolRegistry",
    "OpenAICompatibleProvider", "MockLLM",
    "__version__",
}


def test_public_api_is_exactly_expected():
    assert set(maque_core.__all__) == EXPECTED


def test_every_exported_name_exists():
    for name in maque_core.__all__:
        assert hasattr(maque_core, name), f"{name} is in __all__ but not importable"


def test_internal_modules_not_in_all():
    internal_names = (
        "AppConfig", "LLMConfig", "AgentConfig", "react_loop",
        "RunBudget", "ToolExecutor", "discover_tools", "classify_llm_error",
    )
    for name in internal_names:
        assert name not in maque_core.__all__, f"{name} should not be in __all__"
