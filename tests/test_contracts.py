"""Tests for the contracts layer — messages, LLM protocol, tool protocol."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from maque_core.contracts.messages import Role, ToolCall, Message, LLMResponse, Observation
from maque_core.contracts.llm import LLMProvider
from maque_core.contracts.tool import ToolSpec, ToolRegistry, CallBudget


# ============================================================
# Role tests
# ============================================================

class TestRole:
    def test_role_values(self):
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"

    def test_role_str(self):
        assert str(Role.SYSTEM) == "system"
        assert str(Role.USER) == "user"

    def test_role_from_string(self):
        assert Role("system") == Role.SYSTEM
        assert Role("user") == Role.USER
        assert Role("assistant") == Role.ASSISTANT
        assert Role("tool") == Role.TOOL

    def test_role_is_str_enum(self):
        assert isinstance(Role.SYSTEM, str)
        assert Role.SYSTEM == "system"

    def test_role_comparison(self):
        assert Role.SYSTEM == Role.SYSTEM
        assert Role.SYSTEM != Role.USER

    def test_role_iterate(self):
        roles = list(Role)
        assert len(roles) == 4
        assert Role.SYSTEM in roles


# ============================================================
# ToolCall tests
# ============================================================

class TestToolCall:
    def test_create_basic(self):
        tc = ToolCall(id="call_1", name="echo", arguments={"text": "hello"})
        assert tc.id == "call_1"
        assert tc.name == "echo"
        assert tc.arguments == {"text": "hello"}

    def test_create_default_arguments(self):
        tc = ToolCall(id="call_1", name="test")
        assert tc.arguments == {}

    def test_to_dict(self):
        tc = ToolCall(id="call_1", name="echo", arguments={"text": "hi"})
        d = tc.to_dict()
        assert d == {"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}

    def test_from_dict(self):
        d = {"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}
        tc = ToolCall.from_dict(d)
        assert tc.id == "call_1"
        assert tc.name == "echo"
        assert tc.arguments == {"text": "hi"}

    def test_from_dict_missing_arguments(self):
        d = {"id": "call_1", "name": "test"}
        tc = ToolCall.from_dict(d)
        assert tc.arguments == {}

    def test_roundtrip(self):
        tc = ToolCall(id="abc", name="add", arguments={"a": 1, "b": 2})
        tc2 = ToolCall.from_dict(tc.to_dict())
        assert tc2.id == tc.id
        assert tc2.name == tc.name
        assert tc2.arguments == tc.arguments


# ============================================================
# Message tests
# ============================================================

class TestMessage:
    def test_system_factory(self):
        msg = Message.system("You are helpful.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are helpful."
        assert msg.tool_calls == []

    def test_user_factory(self):
        msg = Message.user("Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_assistant_factory(self):
        msg = Message.assistant("Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"
        assert msg.tool_calls == []

    def test_assistant_with_tool_calls(self):
        tc = ToolCall(id="1", name="echo", arguments={"text": "hi"})
        msg = Message.assistant(tool_calls=[tc])
        assert msg.role == Role.ASSISTANT
        assert msg.content == ""
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "echo"

    def test_tool_result_factory(self):
        msg = Message.tool_result("call_1", "echo", "output")
        assert msg.role == Role.TOOL
        assert msg.content == "output"
        assert msg.tool_call_id == "call_1"
        assert msg.tool_name == "echo"

    def test_to_dict_system(self):
        msg = Message.system("sys")
        d = msg.to_dict()
        assert d == {"role": "system", "content": "sys"}

    def test_to_dict_assistant_with_tool_calls(self):
        tc = ToolCall(id="1", name="t", arguments={"x": 1})
        msg = Message.assistant("text", tool_calls=[tc])
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "text"
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["name"] == "t"

    def test_to_dict_tool_result(self):
        msg = Message.tool_result("cid", "echo", "result")
        d = msg.to_dict()
        assert d["role"] == "tool"
        assert d["content"] == "result"
        assert d["tool_call_id"] == "cid"
        assert d["name"] == "echo"

    def test_from_dict_system(self):
        msg = Message.from_dict({"role": "system", "content": "sys"})
        assert msg.role == Role.SYSTEM
        assert msg.content == "sys"

    def test_from_dict_assistant(self):
        msg = Message.from_dict({"role": "assistant", "content": "hi"})
        assert msg.role == Role.ASSISTANT
        assert msg.content == "hi"
        assert msg.tool_calls == []

    def test_from_dict_with_tool_calls(self):
        d = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "1", "name": "echo", "arguments": {"text": "h"}}],
        }
        msg = Message.from_dict(d)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "1"

    def test_message_roundtrip(self):
        tc = ToolCall(id="abc", name="add", arguments={"a": 1, "b": 2})
        msg = Message.assistant("text", tool_calls=[tc])
        msg2 = Message.from_dict(msg.to_dict())
        assert msg2.role == msg.role
        assert msg2.content == msg.content
        assert len(msg2.tool_calls) == 1
        assert msg2.tool_calls[0].name == "add"

    def test_message_defaults(self):
        msg = Message(role=Role.USER)
        assert msg.content == ""
        assert msg.tool_calls == []
        assert msg.tool_call_id is None
        assert msg.tool_name is None


# ============================================================
# LLMResponse tests
# ============================================================

class TestLLMResponse:
    def test_create_empty(self):
        resp = LLMResponse()
        assert resp.content == ""
        assert resp.tool_calls == []
        assert resp.model == ""
        assert resp.usage is None

    def test_create_with_content(self):
        resp = LLMResponse(content="Hello!", model="gpt-4")
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4"

    def test_create_with_tool_calls(self):
        tc = ToolCall(id="1", name="echo", arguments={"text": "hi"})
        resp = LLMResponse(tool_calls=[tc], usage={"total_tokens": 100})
        assert len(resp.tool_calls) == 1
        assert resp.usage == {"total_tokens": 100}

    def test_to_dict(self):
        tc = ToolCall(id="1", name="t", arguments={"x": 1})
        resp = LLMResponse(content="hi", tool_calls=[tc], model="m", usage={"t": 10})
        d = resp.to_dict()
        assert d["content"] == "hi"
        assert len(d["tool_calls"]) == 1
        assert d["model"] == "m"
        assert d["usage"] == {"t": 10}

    def test_from_dict(self):
        d = {
            "content": "hello",
            "tool_calls": [{"id": "1", "name": "echo", "arguments": {"text": "h"}}],
            "model": "test-model",
            "usage": {"total_tokens": 50},
        }
        resp = LLMResponse.from_dict(d)
        assert resp.content == "hello"
        assert len(resp.tool_calls) == 1
        assert resp.model == "test-model"
        assert resp.usage == {"total_tokens": 50}

    def test_from_dict_minimal(self):
        resp = LLMResponse.from_dict({})
        assert resp.content == ""
        assert resp.tool_calls == []
        assert resp.model == ""
        assert resp.usage is None

    def test_roundtrip(self):
        tc = ToolCall(id="x", name="y", arguments={"z": 1})
        resp = LLMResponse(content="c", tool_calls=[tc], model="m", usage={"k": 5})
        resp2 = LLMResponse.from_dict(resp.to_dict())
        assert resp2.content == resp.content
        assert resp2.model == resp.model
        assert len(resp2.tool_calls) == 1
        assert resp2.usage == resp.usage


# ============================================================
# LLMResponse reasoning field tests
# ============================================================

class TestLLMResponseReasoning:
    def test_default_reasoning_is_none(self):
        resp = LLMResponse()
        assert resp.reasoning is None

    def test_reasoning_field(self):
        resp = LLMResponse(content="answer", reasoning="Let me think about this...")
        assert resp.reasoning == "Let me think about this..."
        assert resp.content == "answer"

    def test_reasoning_in_to_dict(self):
        resp = LLMResponse(content="hi", reasoning="thinking")
        d = resp.to_dict()
        assert d["reasoning"] == "thinking"

    def test_reasoning_in_from_dict(self):
        d = {"content": "hi", "reasoning": "deep thought"}
        resp = LLMResponse.from_dict(d)
        assert resp.reasoning == "deep thought"

    def test_reasoning_none_in_from_dict(self):
        d = {"content": "hi"}
        resp = LLMResponse.from_dict(d)
        assert resp.reasoning is None

    def test_reasoning_roundtrip(self):
        resp = LLMResponse(
            content="final answer",
            reasoning="step by step thinking process",
            model="test-model",
        )
        resp2 = LLMResponse.from_dict(resp.to_dict())
        assert resp2.reasoning == resp.reasoning
        assert resp2.content == resp.content


# ============================================================
# Observation tests
# ============================================================

class TestObservation:
    def test_create_success(self):
        obs = Observation(tool_call_id="c1", tool_name="echo", content="hello")
        assert obs.tool_call_id == "c1"
        assert obs.tool_name == "echo"
        assert obs.content == "hello"
        assert obs.is_error is False
        assert obs.metadata == {}

    def test_create_error(self):
        obs = Observation(
            tool_call_id="c1",
            tool_name="test",
            content="something went wrong",
            is_error=True,
            metadata={"error_type": "ValueError"},
        )
        assert obs.is_error is True
        assert obs.metadata["error_type"] == "ValueError"

    def test_to_dict(self):
        obs = Observation(tool_call_id="c1", tool_name="t", content="out", is_error=False)
        d = obs.to_dict()
        assert d["tool_call_id"] == "c1"
        assert d["tool_name"] == "t"
        assert d["content"] == "out"
        assert d["is_error"] is False
        assert d["metadata"] == {}

    def test_from_dict(self):
        d = {
            "tool_call_id": "c1",
            "tool_name": "echo",
            "content": "hi",
            "is_error": True,
            "metadata": {"k": "v"},
        }
        obs = Observation.from_dict(d)
        assert obs.tool_call_id == "c1"
        assert obs.is_error is True
        assert obs.metadata == {"k": "v"}

    def test_to_message(self):
        obs = Observation(tool_call_id="c1", tool_name="echo", content="result")
        msg = obs.to_message()
        assert msg.role == Role.TOOL
        assert msg.content == "result"
        assert msg.tool_call_id == "c1"
        assert msg.tool_name == "echo"

    def test_roundtrip(self):
        obs = Observation(
            tool_call_id="c1",
            tool_name="test",
            content="output",
            is_error=True,
            metadata={"duration_ms": 100},
        )
        obs2 = Observation.from_dict(obs.to_dict())
        assert obs2.tool_call_id == obs.tool_call_id
        assert obs2.tool_name == obs.tool_name
        assert obs2.content == obs.content
        assert obs2.is_error == obs.is_error
        assert obs2.metadata == obs.metadata


# ============================================================
# ToolSpec tests
# ============================================================

class TestToolSpec:
    def test_create_basic(self):
        spec = ToolSpec(name="echo", description="Echo text")
        assert spec.name == "echo"
        assert spec.description == "Echo text"
        assert spec.parameters == {}
        assert spec.func is None

    def test_create_with_params(self):
        params = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
        spec = ToolSpec(name="echo", description="Echo", parameters=params, func=lambda x: x)
        assert spec.parameters == params
        assert callable(spec.func)

    def test_to_openai_schema(self):
        params = {"type": "object", "properties": {"x": {"type": "integer"}}}
        spec = ToolSpec(name="add", description="Add numbers", parameters=params)
        schema = spec.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"
        assert schema["function"]["description"] == "Add numbers"
        assert schema["function"]["parameters"] == params


# ============================================================
# Protocol runtime check tests
# ============================================================

class TestLLMProviderProtocol:
    def test_protocol_exists(self):
        assert hasattr(LLMProvider, "__protocol_attrs__") or True  # just check it's importable

    def test_mock_llm_is_provider(self):
        from maque_core.llm import MockLLM
        mock = MockLLM()
        assert isinstance(mock, LLMProvider)

    def test_protocol_is_runtime_checkable(self):
        # A class without chat() should not be an LLMProvider
        class NotAProvider:
            pass

        assert not isinstance(NotAProvider(), LLMProvider)

    def test_protocol_with_chat_method(self):
        class HasChat:
            def chat(self, messages, tools=None, **kwargs):
                pass

        assert isinstance(HasChat(), LLMProvider)


class TestToolRegistryProtocol:
    def test_tool_registry_is_protocol(self):
        from maque_core.tools import ToolRegistry as ConcreteRegistry
        reg = ConcreteRegistry()
        assert isinstance(reg, ToolRegistry)

    def test_registry_protocol_methods(self):
        # A class missing execute() should not be a registry
        class IncompleteRegistry:
            def register(self, spec):
                pass
            def get_schemas(self):
                return []
            @property
            def tool_names(self):
                return []
            def __len__(self):
                return 0

        assert not isinstance(IncompleteRegistry(), ToolRegistry)


# ============================================================
# CallBudget tests
# ============================================================

class TestCallBudget:
    def test_unlimited_budget(self):
        budget = CallBudget()
        assert budget.max_calls is None
        assert budget.used == 0
        assert budget.exhausted is False
        assert budget.remaining is None

    def test_limited_budget_initial(self):
        budget = CallBudget(max_calls=5)
        assert budget.max_calls == 5
        assert budget.used == 0
        assert budget.exhausted is False
        assert budget.remaining == 5

    def test_consume_within_budget(self):
        budget = CallBudget(max_calls=5)
        assert budget.consume() is True
        assert budget.used == 1
        assert budget.remaining == 4
        assert budget.exhausted is False

    def test_consume_multiple(self):
        budget = CallBudget(max_calls=10)
        budget.consume(3)
        assert budget.used == 3
        assert budget.remaining == 7

    def test_consume_at_limit(self):
        budget = CallBudget(max_calls=3)
        budget.consume()  # 1
        budget.consume()  # 2
        assert budget.consume() is True  # 3
        assert budget.used == 3
        assert budget.exhausted is True

    def test_consume_past_limit(self):
        budget = CallBudget(max_calls=2)
        assert budget.consume() is True  # 1
        assert budget.consume() is True  # 2
        assert budget.consume() is False  # 3 - over
        assert budget.used == 3
        assert budget.exhausted is True

    def test_exhausted_property(self):
        budget = CallBudget(max_calls=1)
        assert budget.exhausted is False
        budget.consume()
        assert budget.exhausted is True

    def test_remaining_unlimited(self):
        budget = CallBudget()
        assert budget.remaining is None

    def test_remaining_calculation(self):
        budget = CallBudget(max_calls=10)
        budget.consume(4)
        assert budget.remaining == 6

    def test_remaining_never_negative(self):
        budget = CallBudget(max_calls=2)
        budget.consume(5)  # overshoot
        assert budget.remaining == 0


# ============================================================
# Import / dependency tests
# ============================================================

class TestImports:
    def test_contracts_module_imports(self):
        from maque_core import contracts
        assert hasattr(contracts, "Role")
        assert hasattr(contracts, "Message")
        assert hasattr(contracts, "LLMProvider")
        assert hasattr(contracts, "ToolSpec")
        assert hasattr(contracts, "ToolRegistry")
        assert hasattr(contracts, "CallBudget")

    def test_no_circular_imports(self):
        # Import all submodules and make sure no circular import error
        from maque_core.contracts import messages
        from maque_core.contracts import llm
        from maque_core.contracts import tool
        assert messages is not None
        assert llm is not None
        assert tool is not None

    def test_dataclass_not_frozen(self):
        # Messages should be mutable (we add to them during the loop)
        msg = Message.user("test")
        msg.content = "changed"
        assert msg.content == "changed"

    def test_tool_spec_mutable(self):
        spec = ToolSpec(name="t", description="d")
        spec.name = "new_name"
        assert spec.name == "new_name"
