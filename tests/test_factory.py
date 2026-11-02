"""Tests for create_agent factory function."""

from maque_core import create_agent
from maque_core.contracts.messages import LLMResponse
from maque_core.llm.mock import MockLLM


def test_create_agent_no_args():
    a = create_agent(llm=MockLLM([LLMResponse(content="ok")]))
    assert a.list_tools() == []


def test_create_agent_with_tools_package():
    a = create_agent(
        llm=MockLLM([LLMResponse(content="ok")]),
        tools_package="maque_core.tools.impl",
    )
    tool_names = a.list_tools()
    assert "echo" in tool_names
    assert "add" in tool_names


def test_create_agent_with_injected_llm():
    mock = MockLLM([LLMResponse(content="injected")])
    a = create_agent(llm=mock)
    assert a.llm is mock


def test_create_agent_model_override():
    a = create_agent(
        llm=MockLLM([LLMResponse(content="ok")]),
        model="x",
        base_url="y",
    )
    assert a is not None
