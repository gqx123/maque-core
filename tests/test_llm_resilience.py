"""Tests for LLM adapter resilience.

Covers two fixes:

- **P0-4** — LLM failures are no longer disguised as normal answers:
  ``LLMResponse.error`` is populated, errors are classified into typed
  exceptions, and ``raise_on_error=True`` raises them.
- **P1-2** — ``max_tokens`` is sent with every request, and
  ``finish_reason`` is passed through so a truncated answer
  (``finish_reason='length'``) can be recognised upstream.

HTTP is intercepted with ``httpx.MockTransport`` — no real network calls.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pytest

from maque_core.contracts.llm import LLMProvider
from maque_core.contracts.messages import LLMResponse, Message
from maque_core.llm import (
    DEFAULT_MAX_TOKENS,
    LLMAuthError,
    LLMError,
    LLMHTTPError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServerError,
    LLMTimeoutError,
    OpenAICompatibleProvider,
    classify_llm_error,
)
import maque_core.llm.openai_compatible as oc_module


# ============================================================
# Helpers
# ============================================================

def _install_transport(monkeypatch, handler):
    """Route OpenAICompatibleProvider's httpx.post through a mock transport.

    Returns a list that collects the JSON body of every request, so tests
    can assert on what was actually sent.
    """
    captured = []
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):
        captured.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return client.post(url, json=json, headers=headers)

    monkeypatch.setattr(oc_module.httpx, "post", fake_post)
    return captured


def _ok_handler(finish_reason="stop", content="Hello!", model="test-model"):
    """A handler that returns a well-formed chat completion."""

    def handler(request):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            request=request,
        )

    return handler


def _status_handler(status_code, body=None):
    def handler(request):
        return httpx.Response(
            status_code,
            json=body if body is not None else {"error": {"message": "failure"}},
            request=request,
        )

    return handler


def _make_provider(**kwargs):
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("base_url", "https://api.example.com/v1")
    return OpenAICompatibleProvider(**kwargs)


# ============================================================
# P0-4 — error field is populated instead of posing as an answer
# ============================================================

class TestErrorFieldPopulated:
    """Failures must set `error`, while `content` still carries the text."""

    def test_success_response_has_no_error(self, monkeypatch):
        _install_transport(monkeypatch, _ok_handler())
        provider = _make_provider()
        resp = provider.chat([Message.user("hi")])

        assert resp.content == "Hello!"
        assert resp.error is None
        assert resp.is_error is False

    def test_network_error_sets_error_field(self, monkeypatch):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        _install_transport(monkeypatch, handler)
        provider = _make_provider()
        resp = provider.chat([Message.user("hi")])

        assert resp.is_error is True
        assert resp.error is not None
        assert "connection" in resp.error.lower() or "network" in resp.error.lower()
        # content keeps the message so existing display paths still work
        assert resp.content == resp.error
        assert resp.tool_calls == []
        # No genuine answer must be present
        assert "Hello!" not in resp.content

    def test_timeout_sets_error_field(self, monkeypatch):
        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)

        _install_transport(monkeypatch, handler)
        provider = _make_provider(timeout=5)
        resp = provider.chat([Message.user("hi")])

        assert resp.is_error is True
        assert resp.error is not None
        assert "timeout" in resp.error.lower()

    def test_auth_failure_sets_error_field(self, monkeypatch):
        _install_transport(monkeypatch, _status_handler(401))
        provider = _make_provider(api_key="bad-key")
        resp = provider.chat([Message.user("hi")])

        assert resp.is_error is True
        assert resp.error is not None
        assert "auth" in resp.error.lower()
        assert "401" in resp.error

    def test_error_message_keeps_legacy_prefix(self, monkeypatch):
        """The historical ``LLM API error: ...`` shape is preserved."""
        _install_transport(monkeypatch, _status_handler(500))
        provider = _make_provider()
        resp = provider.chat([Message.user("hi")])

        assert resp.content.startswith("LLM API error: ")
        assert "500" in resp.content

    def test_malformed_body_sets_error_field(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, text="<html>not json</html>", request=request)

        _install_transport(monkeypatch, handler)
        provider = _make_provider()
        resp = provider.chat([Message.user("hi")])

        assert resp.is_error is True
        assert resp.error is not None


# ============================================================
# P0-4 — raise_on_error=True raises typed exceptions
# ============================================================

class TestRaiseOnError:
    """`raise_on_error=True` must raise, with the right subclass."""

    def test_default_does_not_raise(self, monkeypatch):
        """Default behaviour stays backward compatible."""
        def handler(request):
            raise httpx.ConnectError("boom", request=request)

        _install_transport(monkeypatch, handler)
        provider = _make_provider()
        resp = provider.chat([Message.user("hi")])  # no raise
        assert resp.is_error is True

    def test_network_error_raises_network_error(self, monkeypatch):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        _install_transport(monkeypatch, handler)
        provider = _make_provider()

        with pytest.raises(LLMNetworkError) as exc_info:
            provider.chat([Message.user("hi")], raise_on_error=True)

        err = exc_info.value
        assert isinstance(err, LLMError)
        assert err.category == "network"
        assert err.retryable is True
        assert err.original_type == "ConnectError"
        assert isinstance(err.__cause__, httpx.ConnectError)

    def test_timeout_raises_timeout_error(self, monkeypatch):
        def handler(request):
            raise httpx.ReadTimeout("read timeout", request=request)

        _install_transport(monkeypatch, handler)
        provider = _make_provider()

        with pytest.raises(LLMTimeoutError) as exc_info:
            provider.chat([Message.user("hi")], raise_on_error=True)

        err = exc_info.value
        assert err.category == "timeout"
        assert err.retryable is True
        # Timeout is a kind of network error, but must stay distinguishable
        assert isinstance(err, LLMNetworkError)
        assert not isinstance(err, LLMAuthError)

    def test_401_raises_auth_error(self, monkeypatch):
        _install_transport(monkeypatch, _status_handler(401, {"error": "invalid api key"}))
        provider = _make_provider(api_key="bad-key")

        with pytest.raises(LLMAuthError) as exc_info:
            provider.chat([Message.user("hi")], raise_on_error=True)

        err = exc_info.value
        assert err.category == "auth"
        assert err.status_code == 401
        assert err.retryable is False  # retrying a bad key is pointless

    def test_403_raises_auth_error(self, monkeypatch):
        _install_transport(monkeypatch, _status_handler(403))
        provider = _make_provider()

        with pytest.raises(LLMAuthError) as exc_info:
            provider.chat([Message.user("hi")], raise_on_error=True)

        assert exc_info.value.status_code == 403

    def test_429_raises_rate_limit_error(self, monkeypatch):
        _install_transport(monkeypatch, _status_handler(429))
        provider = _make_provider()

        with pytest.raises(LLMRateLimitError) as exc_info:
            provider.chat([Message.user("hi")], raise_on_error=True)

        assert exc_info.value.retryable is True

    def test_500_raises_server_error(self, monkeypatch):
        _install_transport(monkeypatch, _status_handler(500))
        provider = _make_provider()

        with pytest.raises(LLMServerError) as exc_info:
            provider.chat([Message.user("hi")], raise_on_error=True)

        assert exc_info.value.status_code == 500
        assert exc_info.value.retryable is True

    def test_success_does_not_raise_when_raise_on_error_true(self, monkeypatch):
        _install_transport(monkeypatch, _ok_handler())
        provider = _make_provider()
        resp = provider.chat([Message.user("hi")], raise_on_error=True)
        assert resp.content == "Hello!"
        assert resp.error is None

    def test_error_to_dict_is_structured(self, monkeypatch):
        _install_transport(monkeypatch, _status_handler(401))
        provider = _make_provider()

        with pytest.raises(LLMAuthError) as exc_info:
            provider.chat([Message.user("hi")], raise_on_error=True)

        payload = exc_info.value.to_dict()
        assert payload["category"] == "auth"
        assert payload["status_code"] == 401
        assert payload["provider"] == "OpenAICompatibleProvider"
        assert payload["retryable"] is False


# ============================================================
# P0-4 — classify_llm_error unit tests
# ============================================================

class TestClassifyLLMError:
    def test_connect_error_is_network(self):
        err = classify_llm_error(httpx.ConnectError("dns failure"))
        assert isinstance(err, LLMNetworkError)
        assert err.category == "network"

    def test_connect_timeout_is_timeout_not_network(self):
        err = classify_llm_error(httpx.ConnectTimeout("timeout"))
        assert isinstance(err, LLMTimeoutError)
        assert err.category == "timeout"

    def test_builtin_timeout_error(self):
        err = classify_llm_error(TimeoutError("too slow"))
        assert isinstance(err, LLMTimeoutError)

    def test_json_decode_error_is_response_error(self):
        err = classify_llm_error(json.JSONDecodeError("bad", "", 0))
        assert isinstance(err, LLMResponseError)
        assert err.retryable is False

    def test_unknown_exception_falls_back_to_base(self):
        err = classify_llm_error(RuntimeError("weird"))
        assert isinstance(err, LLMError)
        assert err.category == "unknown"

    def test_status_code_boundaries(self):
        """4xx other than 401/403/429 map to generic HTTP error."""
        assert isinstance(classify_llm_error(_status_exc(400)), LLMHTTPError)
        assert isinstance(classify_llm_error(_status_exc(404)), LLMHTTPError)
        assert not isinstance(classify_llm_error(_status_exc(404)), LLMAuthError)

    def test_status_599_is_server_error(self):
        assert isinstance(classify_llm_error(_status_exc(599)), LLMServerError)

    def test_message_carries_category_and_original_type(self):
        err = classify_llm_error(httpx.ConnectError("refused"),
                                 provider="P", model="m")
        assert "network" in err.message
        assert "ConnectError" in err.message
        assert err.provider == "P"
        assert err.model == "m"


def _status_exc(status_code):
    """Build an httpx.HTTPStatusError carrying the given status."""
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


# ============================================================
# P1-2 — max_tokens is sent with the request
# ============================================================

class TestMaxTokens:
    def test_default_max_tokens_sent(self, monkeypatch):
        captured = _install_transport(monkeypatch, _ok_handler())
        provider = _make_provider()
        provider.chat([Message.user("hi")])

        assert captured[0]["json"]["max_tokens"] == DEFAULT_MAX_TOKENS == 2048

    def test_constructor_max_tokens_used(self, monkeypatch):
        captured = _install_transport(monkeypatch, _ok_handler())
        provider = _make_provider(max_tokens=1024)
        provider.chat([Message.user("hi")])

        assert captured[0]["json"]["max_tokens"] == 1024
        assert provider.max_tokens == 1024

    def test_per_call_max_tokens_overrides_constructor(self, monkeypatch):
        captured = _install_transport(monkeypatch, _ok_handler())
        provider = _make_provider(max_tokens=1024)
        provider.chat([Message.user("hi")], max_tokens=256)

        assert captured[0]["json"]["max_tokens"] == 256
        # constructor default must be untouched by the per-call override
        assert provider.max_tokens == 1024

    def test_keyword_max_tokens_still_supported(self, monkeypatch):
        """Legacy ``chat(..., max_tokens=...)`` via kwargs keeps working."""
        captured = _install_transport(monkeypatch, _ok_handler())
        provider = _make_provider()
        provider.chat([Message.user("hi")], max_tokens=512)
        assert captured[0]["json"]["max_tokens"] == 512

    def test_max_tokens_none_falls_back_to_constructor(self, monkeypatch):
        captured = _install_transport(monkeypatch, _ok_handler())
        provider = _make_provider(max_tokens=2048)
        provider.chat([Message.user("hi")], max_tokens=None)
        assert captured[0]["json"]["max_tokens"] == 2048

    def test_none_default_omits_field(self, monkeypatch):
        """max_tokens=None on the constructor restores legacy behaviour."""
        captured = _install_transport(monkeypatch, _ok_handler())
        provider = _make_provider(max_tokens=None)
        provider.chat([Message.user("hi")])

        assert "max_tokens" not in captured[0]["json"]
        assert provider.max_tokens is None

    def test_invalid_max_tokens_ignored(self, monkeypatch):
        captured = _install_transport(monkeypatch, _ok_handler())
        provider = _make_provider(max_tokens=None)
        provider.chat([Message.user("hi")], max_tokens=0)
        assert "max_tokens" not in captured[0]["json"]

    def test_bool_max_tokens_rejected(self, monkeypatch):
        """True is an int subclass — it must not silently become 1."""
        captured = _install_transport(monkeypatch, _ok_handler())
        provider = _make_provider(max_tokens=None)
        provider.chat([Message.user("hi")], max_tokens=True)
        assert "max_tokens" not in captured[0]["json"]

    def test_resolve_max_tokens_negative_skipped(self):
        assert OpenAICompatibleProvider._resolve_max_tokens(-5, 2048) == 2048
        assert OpenAICompatibleProvider._resolve_max_tokens(None, None) is None
        assert OpenAICompatibleProvider._resolve_max_tokens(False, 1024) == 1024


# ============================================================
# P1-2 — finish_reason passthrough
# ============================================================

class TestFinishReason:
    @pytest.mark.parametrize("reason", ["stop", "length", "tool_calls", "content_filter"])
    def test_finish_reason_passed_through(self, reason):
        data = {
            "model": "test",
            "choices": [
                {"message": {"role": "assistant", "content": "hi"}, "finish_reason": reason}
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.finish_reason == reason

    def test_finish_reason_absent_is_none(self):
        data = {"model": "test", "choices": [{"message": {"content": "hi"}}]}
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.finish_reason is None

    def test_finish_reason_null_is_none(self):
        data = {"model": "test", "choices": [{"message": {"content": "hi"}, "finish_reason": None}]}
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.finish_reason is None

    def test_no_choices_has_no_finish_reason(self):
        resp = OpenAICompatibleProvider._parse_response({"model": "test", "choices": []})
        assert resp.finish_reason is None
        assert resp.error is None

    def test_finish_reason_from_http_call(self, monkeypatch):
        """End-to-end through chat(): truncated answers stay identifiable."""
        _install_transport(monkeypatch, _ok_handler(finish_reason="length", content="The answer is 4"))
        provider = _make_provider()
        resp = provider.chat([Message.user("hi")])

        assert resp.finish_reason == "length"
        assert resp.is_truncated is True
        assert resp.is_error is False  # truncated, but not an API error
        assert resp.content == "The answer is 4"

    def test_stop_is_not_truncated(self, monkeypatch):
        _install_transport(monkeypatch, _ok_handler(finish_reason="stop"))
        provider = _make_provider()
        resp = provider.chat([Message.user("hi")])
        assert resp.is_truncated is False

    def test_truncation_flag_with_tool_calls(self):
        """finish_reason is independent of tool_calls presence."""
        data = {
            "model": "test",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "1",
                                "type": "function",
                                "function": {"name": "echo", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        resp = OpenAICompatibleProvider._parse_response(data)
        assert resp.finish_reason == "tool_calls"
        assert len(resp.tool_calls) == 1


# ============================================================
# Backward compatibility
# ============================================================

class TestBackwardCompatibility:
    def test_new_fields_default_to_none(self):
        resp = LLMResponse()
        assert resp.error is None
        assert resp.finish_reason is None
        assert resp.is_error is False
        assert resp.is_truncated is False

    def test_positional_construction_unchanged(self):
        """Legacy positional args keep their meaning."""
        resp = LLMResponse("content", "reasoning", [], "model", {"total_tokens": 1})
        assert resp.content == "content"
        assert resp.reasoning == "reasoning"
        assert resp.model == "model"
        assert resp.error is None
        assert resp.finish_reason is None

    def test_dict_roundtrip_preserves_new_fields(self):
        resp = LLMResponse(
            content="",
            model="m",
            finish_reason="length",
            error="LLM API error: auth: ...",
        )
        restored = LLMResponse.from_dict(resp.to_dict())
        assert restored.finish_reason == "length"
        assert restored.error == resp.error

    def test_from_dict_without_new_keys(self):
        """Old serialized payloads (pre-fix) still load."""
        old = {"content": "hi", "reasoning": None, "tool_calls": [], "model": "m", "usage": None}
        restored = LLMResponse.from_dict(old)
        assert restored.error is None
        assert restored.finish_reason is None

    def test_to_dict_contains_new_keys(self):
        payload = LLMResponse().to_dict()
        assert "error" in payload
        assert "finish_reason" in payload

    def test_provider_is_still_llm_provider(self):
        assert isinstance(OpenAICompatibleProvider(), LLMProvider)

    def test_mock_llm_still_llm_provider(self):
        from maque_core.llm import MockLLM
        assert isinstance(MockLLM(), LLMProvider)

    def test_existing_kwargs_still_work(self, monkeypatch):
        captured = _install_transport(monkeypatch, _ok_handler())
        provider = _make_provider()
        provider.chat([Message.user("hi")], temperature=0.5, model="custom-model")

        body = captured[0]["json"]
        assert body["temperature"] == 0.5
        assert body["model"] == "custom-model"
        assert body["max_tokens"] == 2048



# ============================================================
# Config wiring — max_tokens comes from config
# ============================================================

class TestConfigWiring:
    """The provider must take its default max_tokens from LLMConfig."""

    def test_llm_config_has_max_tokens_default(self):
        from maque_core.config import LLMConfig

        assert LLMConfig().max_tokens == 2048

    def test_llm_config_from_env(self, monkeypatch):
        from maque_core.config import LLMConfig

        monkeypatch.setenv("MAQUE_LLM_MAX_TOKENS", "777")
        assert LLMConfig.from_env().max_tokens == 777

    def test_build_agent_passes_max_tokens(self):
        """app.build_agent wires config.llm.max_tokens into the provider."""
        import app
        from maque_core.config import AppConfig

        cfg = AppConfig()
        cfg.llm.max_tokens = 512
        agent = app.build_agent(cfg)

        assert agent.llm.max_tokens == 512

    def test_agent_request_carries_configured_max_tokens(self, monkeypatch):
        """End-to-end: the configured value lands in the request body."""
        import app
        from maque_core.config import AppConfig

        captured = _install_transport(monkeypatch, _ok_handler())
        cfg = AppConfig()
        cfg.llm.max_tokens = 999
        agent = app.build_agent(cfg)
        agent.llm.chat([Message.user("hi")])

        assert captured[0]["json"]["max_tokens"] == 999
