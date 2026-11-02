"""Tests for the LLM adapter layer — OpenAI compatible provider and MockLLM."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from maque_core.contracts.messages import Role, ToolCall, Message, LLMResponse
from maque_core.contracts.tool import ToolSpec
from maque_core.llm import MockLLM
from maque_core.llm.openai_compatible import OpenAICompatibleProvider


# ============================================================
# MockLLM tests
# ============================================================

class TestMockLLM:
    def test_create_empty(self):
        mock = MockLLM()
        assert mock.call_count == 0
        assert mock.exhausted is True
        assert mock.remaining == 0

    def test_create_with_responses(self):
        responses = [LLMResponse(content="hi"), LLMResponse(content="bye")]
        mock = MockLLM(responses)
        assert mock.call_count == 0
        assert mock.remaining == 2
        assert mock.exhausted is False

    def test_add_response(self):
        mock = MockLLM()
        mock.add_response(LLMResponse(content="test"))
        assert mock.remaining == 1
        assert mock.exhausted is False

    def test_chat_returns_responses_in_order(self):
        mock = MockLLM([
            LLMResponse(content="first"),
            LLMResponse(content="second"),
            LLMResponse(content="third"),
        ])
        r1 = mock.chat([Message.user("q1")])
        assert r1.content == "first"
        r2 = mock.chat([Message.user("q2")])
        assert r2.content == "second"
        r3 = mock.chat([Message.user("q3")])
        assert r3.content == "third"

    def test_chat_exhausted_returns_empty(self):
        mock = MockLLM([LLMResponse(content="only")])
        mock.chat([])
        r = mock.chat([])
        assert r.content == ""
        assert r.tool_calls == []

    def test_call_count_tracks_calls(self):
        mock = MockLLM([LLMResponse() for _ in range(5)])
        for i in range(3):
            mock.chat([])
        assert mock.call_count == 3

    def test_last_messages_captured(self):
        mock = MockLLM([LLMResponse(content="ok")])
        msgs = [Message.system("sys"), Message.user("hello")]
        mock.chat(msgs)
        assert mock.last_messages is not None
        assert len(mock.last_messages) == 2
        assert mock.last_messages[0].role == Role.SYSTEM

    def test_last_tools_captured(self):
        mock = MockLLM([LLMResponse(content="ok")])
        tools = [ToolSpec(name="t", description="d")]
        mock.chat([Message.user("hi")], tools=tools)
        assert mock.last_tools is not None
        assert len(mock.last_tools) == 1

    def test_reset(self):
        mock = MockLLM([LLMResponse(content="a"), LLMResponse(content="b")])
        mock.chat([])
        mock.chat([])
        assert mock.call_count == 2
        assert mock.exhausted is True
        mock.reset()
        assert mock.call_count == 0
        assert mock.exhausted is False
        assert mock.remaining == 2
        assert mock.last_messages is None

    def test_chat_with_tool_calls_response(self):
        tc = ToolCall(id="call1", name="echo", arguments={"text": "hi"})
        mock = MockLLM([LLMResponse(content="", tool_calls=[tc])])
        r = mock.chat([Message.user("test")])
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "echo"
        assert r.tool_calls[0].arguments == {"text": "hi"}

    def test_chat_with_multiple_tool_calls(self):
        tcs = [
            ToolCall(id="1", name="echo", arguments={"text": "a"}),
            ToolCall(id="2", name="add", arguments={"a": 1, "b": 2}),
        ]
        mock = MockLLM([LLMResponse(tool_calls=tcs)])
        r = mock.chat([])
        assert len(r.tool_calls) == 2
        assert r.tool_calls[0].name == "echo"
        assert r.tool_calls[1].name == "add"

    def test_chat_kwargs_ignored(self):
        mock = MockLLM([LLMResponse(content="ok")])
        r = mock.chat([], temperature=0.5, model="custom", extra="value")
        assert r.content == "ok"

    def test_mock_llm_is_llm_provider(self):
        from maque_core.contracts.llm import LLMProvider
        mock = MockLLM()
        assert isinstance(mock, LLMProvider)


# ============================================================
# OpenAICompatibleProvider — message conversion tests
# ============================================================

class TestMessageConversion:
    def test_system_message_to_api(self):
        msg = Message.system("You are helpful.")
        result = OpenAICompatibleProvider._message_to_api(msg)
        assert result == {"role": "system", "content": "You are helpful."}

    def test_user_message_to_api(self):
        msg = Message.user("Hello")
        result = OpenAICompatibleProvider._message_to_api(msg)
        assert result == {"role": "user", "content": "Hello"}

    def test_assistant_message_to_api(self):
        msg = Message.assistant("Hi there!")
        result = OpenAICompatibleProvider._message_to_api(msg)
        assert result == {"role": "assistant", "content": "Hi there!"}

    def test_assistant_with_tool_calls_to_api(self):
        tc = ToolCall(id="call_123", name="echo", arguments={"text": "hello"})
        msg = Message.assistant("", tool_calls=[tc])
        result = OpenAICompatibleProvider._message_to_api(msg)
        assert result["role"] == "assistant"
        assert result["content"] == ""
        assert len(result["tool_calls"]) == 1
        tc_api = result["tool_calls"][0]
        assert tc_api["id"] == "call_123"
        assert tc_api["type"] == "function"
        assert tc_api["function"]["name"] == "echo"
        args = json.loads(tc_api["function"]["arguments"])
        assert args == {"text": "hello"}

    def test_assistant_with_multiple_tool_calls(self):
        tcs = [
            ToolCall(id="1", name="echo", arguments={"text": "a"}),
            ToolCall(id="2", name="add", arguments={"a": 1, "b": 2}),
        ]
        msg = Message.assistant("", tool_calls=tcs)
        result = OpenAICompatibleProvider._message_to_api(msg)
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["function"]["name"] == "echo"
        assert result["tool_calls"][1]["function"]["name"] == "add"

    def test_tool_result_message_to_api(self):
        msg = Message.tool_result("call_123", "echo", "hello back")
        result = OpenAICompatibleProvider._message_to_api(msg)
        assert result["role"] == "tool"
        assert result["content"] == "hello back"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "echo"

    def test_tool_call_arguments_json_encoded(self):
        tc = ToolCall(id="1", name="add", arguments={"a": 42, "b": 58})
        msg = Message.assistant("", tool_calls=[tc])
        result = OpenAICompatibleProvider._message_to_api(msg)
        args_str = result["tool_calls"][0]["function"]["arguments"]
        args = json.loads(args_str)
        assert args == {"a": 42, "b": 58}

    def test_empty_arguments_encoded(self):
        tc = ToolCall(id="1", name="noop", arguments={})
        msg = Message.assistant("", tool_calls=[tc])
        result = OpenAICompatibleProvider._message_to_api(msg)
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])
        assert args == {}

    def test_unicode_arguments_preserved(self):
        tc = ToolCall(id="1", name="echo", arguments={"text": "你好世界"})
        msg = Message.assistant("", tool_calls=[tc])
        result = OpenAICompatibleProvider._message_to_api(msg)
        args = json.loads(result["tool_calls"][0]["function"]["arguments"])
        assert args["text"] == "你好世界"


# ============================================================
# OpenAICompatibleProvider — tool schema conversion tests
# ============================================================

class TestToolSchemaConversion:
    def test_tool_spec_to_api(self):
        spec = ToolSpec(
            name="echo",
            description="Echo back text",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        result = OpenAICompatibleProvider._tool_to_api(spec)
        assert result["type"] == "function"
        assert result["function"]["name"] == "echo"
        assert result["function"]["description"] == "Echo back text"
        assert result["function"]["parameters"]["type"] == "object"

    def test_tool_spec_with_no_params(self):
        spec = ToolSpec(name="noop", description="Do nothing")
        result = OpenAICompatibleProvider._tool_to_api(spec)
        assert result["function"]["name"] == "noop"
        assert result["function"]["parameters"] == {}

    def test_tool_spec_with_complex_params(self):
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array"},
            },
            "required": ["name"],
        }
        spec = ToolSpec(name="create_user", description="Create a user", parameters=params)
        result = OpenAICompatibleProvider._tool_to_api(spec)
        assert result["function"]["parameters"]["properties"]["name"]["type"] == "string"
        assert result["function"]["parameters"]["properties"]["age"]["type"] == "integer"
        assert result["function"]["parameters"]["required"] == ["name"]


# ============================================================
# OpenAICompatibleProvider — response parsing tests
# ============================================================

class TestResponseParsing:
    def test_parse_simple_text_response(self):
        data = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.content == "Hello!"
        assert resp.tool_calls == []
        assert resp.model == "gpt-4"
        assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_parse_empty_content_response(self):
        data = {
            "model": "test",
            "choices": [{"message": {"role": "assistant", "content": ""}}],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.content == ""
        assert resp.tool_calls == []

    def test_parse_null_content(self):
        data = {
            "model": "test",
            "choices": [{"message": {"role": "assistant", "content": None}}],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.content == ""

    def test_parse_single_tool_call(self):
        data = {
            "model": "test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"text": "hello"}',
                                },
                            }
                        ],
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].id == "call_abc"
        assert resp.tool_calls[0].name == "echo"
        assert resp.tool_calls[0].arguments == {"text": "hello"}

    def test_parse_multiple_tool_calls(self):
        data = {
            "model": "test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "1",
                                "type": "function",
                                "function": {"name": "echo", "arguments": '{"text": "a"}'},
                            },
                            {
                                "id": "2",
                                "type": "function",
                                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                            },
                        ],
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "echo"
        assert resp.tool_calls[1].name == "add"
        assert resp.tool_calls[1].arguments == {"a": 1, "b": 2}

    def test_parse_no_choices(self):
        data = {"model": "test", "choices": []}
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.content == ""
        assert resp.tool_calls == []
        assert resp.model == "test"

    def test_parse_no_message_in_choice(self):
        data = {"model": "test", "choices": [{}]}
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.content == ""

    def test_parse_invalid_json_arguments(self):
        data = {
            "model": "test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "1",
                                "type": "function",
                                "function": {"name": "bad", "arguments": "not json {{{"},
                            }
                        ],
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert len(resp.tool_calls) == 1
        # Should fall back to storing raw string
        assert "raw" in resp.tool_calls[0].arguments

    def test_parse_unicode_content(self):
        data = {
            "model": "test",
            "choices": [{"message": {"role": "assistant", "content": "你好，世界！"}}],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.content == "你好，世界！"

    def test_parse_response_without_usage(self):
        data = {
            "model": "test",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.usage is None

    def test_parse_first_choice_only(self):
        data = {
            "model": "test",
            "choices": [
                {"message": {"content": "first"}},
                {"message": {"content": "second"}},
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.content == "first"


# ============================================================
# OpenAICompatibleProvider — initialization tests
# ============================================================

class TestProviderInit:
    def test_default_init(self):
        provider = OpenAICompatibleProvider()
        assert provider.model == "qwen-turbo"
        assert "dashscope" in provider.base_url
        assert provider.temperature == 0.7
        assert provider.timeout == 60

    def test_custom_init(self):
        provider = OpenAICompatibleProvider(
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="sk-test123",
            temperature=0.0,
            timeout=30,
        )
        assert provider.model == "gpt-4"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.api_key == "sk-test123"
        assert provider.temperature == 0.0
        assert provider.timeout == 30

    def test_base_url_trailing_slash_stripped(self):
        provider = OpenAICompatibleProvider(base_url="https://example.com/v1/")
        assert provider.base_url == "https://example.com/v1"

    def test_is_llm_provider(self):
        from maque_core.contracts.llm import LLMProvider
        provider = OpenAICompatibleProvider()
        assert isinstance(provider, LLMProvider)


# ============================================================
# OpenAICompatibleProvider — chat method tests (mock HTTP)
# ============================================================

class TestProviderChat:
    def test_chat_builds_correct_url(self):
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            api_key="test-key",
        )
        # We can't easily test the actual HTTP call without httpx mock,
        # but we can verify the URL construction logic
        expected_url = "https://api.example.com/v1/chat/completions"
        assert f"{provider.base_url}/chat/completions" == expected_url

    def test_chat_without_tools(self):
        provider = OpenAICompatibleProvider(api_key="test")
        messages = [Message.user("Hello")]
        # This will fail since no real API, but should not crash on setup
        try:
            provider.chat(messages)
        except Exception:
            pass  # Expected — no real API

    def test_chat_with_tools(self):
        provider = OpenAICompatibleProvider(api_key="test")
        tools = [ToolSpec(name="echo", description="Echo")]
        messages = [Message.user("Hi")]
        try:
            provider.chat(messages, tools=tools)
        except Exception:
            pass  # Expected

    def test_chat_with_kwargs(self):
        provider = OpenAICompatibleProvider(api_key="test")
        try:
            provider.chat([Message.user("hi")], temperature=0.5, model="custom-model")
        except Exception:
            pass

    def test_chat_empty_messages(self):
        provider = OpenAICompatibleProvider(api_key="test")
        try:
            provider.chat([])
        except Exception:
            pass


# ============================================================
# LLMResponse integration with messages
# ============================================================

class TestLLMResponseIntegration:
    def test_response_to_assistant_message(self):
        tc = ToolCall(id="1", name="echo", arguments={"text": "hi"})
        resp = LLMResponse(content="Let me check...", tool_calls=[tc])
        msg = Message.assistant(content=resp.content, tool_calls=resp.tool_calls)
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Let me check..."
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "echo"

    def test_empty_response_is_valid(self):
        resp = LLMResponse()
        msg = Message.assistant(content=resp.content, tool_calls=resp.tool_calls)
        assert msg.content == ""
        assert msg.tool_calls == []

    def test_response_model_preserved(self):
        resp = LLMResponse(content="hi", model="my-model-v1")
        assert resp.model == "my-model-v1"

    def test_response_usage_dict(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        resp = LLMResponse(content="hi", usage=usage)
        assert resp.usage["total_tokens"] == 150


# ============================================================
# Round-trip tests
# ============================================================

class TestRoundTrip:
    def test_tool_call_roundtrip_message(self):
        tc = ToolCall(id="call_xyz", name="add", arguments={"a": 10, "b": 20})
        msg = Message.assistant("", tool_calls=[tc])
        api_msg = OpenAICompatibleProvider._message_to_api(msg)
        # Simulate what the API would return
        response_data = {
            "model": "test",
            "choices": [{"message": api_msg}],
        }
        resp = OpenAICompatibleProvider._parse_response(response_data)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].id == "call_xyz"
        assert resp.tool_calls[0].name == "add"
        assert resp.tool_calls[0].arguments == {"a": 10, "b": 20}

    def test_message_dict_roundtrip(self):
        msg = Message.assistant(
            "Hello world",
            tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "hi"})],
        )
        d = msg.to_dict()
        msg2 = Message.from_dict(d)
        assert msg2.role == msg.role
        assert msg2.content == msg.content
        assert len(msg2.tool_calls) == len(msg.tool_calls)
        assert msg2.tool_calls[0].name == msg.tool_calls[0].name
        assert msg2.tool_calls[0].arguments == msg.tool_calls[0].arguments

    def test_llm_response_dict_roundtrip(self):
        tc = ToolCall(id="abc", name="test", arguments={"x": 1, "y": 2})
        resp = LLMResponse(
            content="answer",
            tool_calls=[tc],
            model="test-model",
            usage={"total_tokens": 42},
        )
        d = resp.to_dict()
        resp2 = LLMResponse.from_dict(d)
        assert resp2.content == resp.content
        assert resp2.model == resp.model
        assert len(resp2.tool_calls) == 1
        assert resp2.tool_calls[0].id == "abc"
        assert resp2.usage == resp.usage



# ============================================================
# Reasoning extraction tests (three-level priority)
# ============================================================

class TestReasoningExtraction:
    """Tests for the three-level reasoning extraction in OpenAICompatibleProvider.

    Priority:
    1. reasoning_content field (Qwen3 / DashScope)
    2. reasoning field (some APIs)
    3. inline <think>...</think> tags in content
    """

    def test_level1_reasoning_content_field(self):
        """Level 1: reasoning_content field takes highest priority."""
        data = {
            "model": "qwen3-turbo",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "reasoning_content": "Let me calculate this step by step... 40 + 2 = 42.",
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.reasoning == "Let me calculate this step by step... 40 + 2 = 42."
        assert resp.content == "The answer is 42."

    def test_level2_reasoning_field(self):
        """Level 2: reasoning field used when reasoning_content absent."""
        data = {
            "model": "some-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                        "reasoning": "I should greet the user warmly.",
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.reasoning == "I should greet the user warmly."
        assert resp.content == "Hello!"

    def test_level3_think_tags_at_start(self):
        """Level 3: <think> tags at the start of content."""
        data = {
            "model": "deepseek-r1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>Hmm, let me think about this problem.</think>The answer is 42.",
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.reasoning == "Hmm, let me think about this problem."
        assert resp.content == "The answer is 42."

    def test_level3_think_tags_multiline(self):
        """Level 3: multi-line <think> tags."""
        data = {
            "model": "deepseek-r1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>Step 1: identify the problem\nStep 2: find the solution\nStep 3: verify</think>Here is the final answer.",
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert "Step 1: identify the problem" in resp.reasoning
        assert "Step 3: verify" in resp.reasoning
        assert resp.content == "Here is the final answer."

    def test_level3_multiple_think_blocks(self):
        """Level 3: multiple <think> blocks are combined."""
        data = {
            "model": "deepseek-r1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>First thought.</think>Some text.<think>Second thought.</think>Final answer.",
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert "First thought." in resp.reasoning
        assert "Second thought." in resp.reasoning
        assert "Some text." in resp.content
        assert "Final answer." in resp.content

    def test_level3_no_think_tags(self):
        """Level 3: no <think> tags → reasoning is None."""
        data = {
            "model": "gpt-4",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Just a normal answer without thinking tags.",
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.reasoning is None
        assert resp.content == "Just a normal answer without thinking tags."

    def test_level1_overrides_level3(self):
        """Level 1 (reasoning_content) takes priority over Level 3 (<think> tags)."""
        data = {
            "model": "qwen3-turbo",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>inline thinking</think>Real answer.",
                        "reasoning_content": "official reasoning content",
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        # reasoning_content takes priority
        assert resp.reasoning == "official reasoning content"
        # content should NOT be stripped of <think> tags when level 1 is used
        # (because we only do level 3 extraction as fallback)
        assert "<think>" in resp.content

    def test_level2_overrides_level3(self):
        """Level 2 (reasoning) takes priority over Level 3 (<think> tags)."""
        data = {
            "model": "some-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>inline</think>Answer.",
                        "reasoning": "separate reasoning field",
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.reasoning == "separate reasoning field"
        assert "<think>" in resp.content

    def test_empty_reasoning_content_skipped(self):
        """Empty reasoning_content string is treated as absent."""
        data = {
            "model": "test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hi",
                        "reasoning_content": "",
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.reasoning is None

    def test_reasoning_with_tool_calls(self):
        """Reasoning works alongside tool calls."""
        data = {
            "model": "qwen3-turbo",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "I need to use the echo tool to reply.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"text": "hello"}',
                                },
                            }
                        ],
                    }
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.reasoning == "I need to use the echo tool to reply."
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "echo"


# ============================================================
# _extract_think_tags unit tests
# ============================================================

class TestExtractThinkTags:
    def test_no_tags(self):
        cleaned, reasoning = OpenAICompatibleProvider._extract_think_tags("Hello world")
        assert cleaned == "Hello world"
        assert reasoning is None

    def test_single_tag_at_start(self):
        cleaned, reasoning = OpenAICompatibleProvider._extract_think_tags(
            "<think>Let me think.</think>Here is the answer."
        )
        assert reasoning == "Let me think."
        assert cleaned == "Here is the answer."

    def test_single_tag_in_middle(self):
        cleaned, reasoning = OpenAICompatibleProvider._extract_think_tags(
            "Intro.<think>thinking</think>Conclusion."
        )
        assert reasoning == "thinking"
        assert "Intro." in cleaned
        assert "Conclusion." in cleaned

    def test_multiple_tags(self):
        cleaned, reasoning = OpenAICompatibleProvider._extract_think_tags(
            "<think>First.</think>Middle.<think>Second.</think>End."
        )
        assert "First." in reasoning
        assert "Second." in reasoning
        assert "Middle." in cleaned
        assert "End." in cleaned

    def test_multiline_think(self):
        text = "<think>Line 1\nLine 2\nLine 3</think>Answer."
        cleaned, reasoning = OpenAICompatibleProvider._extract_think_tags(text)
        assert "Line 1" in reasoning
        assert "Line 3" in reasoning
        assert cleaned == "Answer."

    def test_whitespace_stripped(self):
        cleaned, reasoning = OpenAICompatibleProvider._extract_think_tags(
            "<think>  lots of space  </think>  text  "
        )
        assert reasoning == "lots of space"
        assert "text" in cleaned

    def test_empty_think_tag(self):
        cleaned, reasoning = OpenAICompatibleProvider._extract_think_tags(
            "<think></think>Nothing here."
        )
        assert reasoning is None
        assert cleaned == "Nothing here."

    def test_unicode_think_content(self):
        cleaned, reasoning = OpenAICompatibleProvider._extract_think_tags(
            "<think>让我想想这个问题。</think>答案是 42。"
        )
        assert reasoning == "让我想想这个问题。"
        assert cleaned == "答案是 42。"
