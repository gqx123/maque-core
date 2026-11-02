"""OpenAI-compatible LLM provider.

Works with any API that follows the OpenAI chat completions format:
- OpenAI
- DashScope (Qwen) — via compatible-mode endpoint
- DeepSeek
- Anthropic (via third-party proxies)
- Local models (via Ollama, vLLM, etc.)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

from ..contracts.messages import LLMResponse, Message, ToolCall
from .errors import LLMError, LLMResponseError, classify_llm_error

DEFAULT_MAX_TOKENS = 2048


class OpenAICompatibleProvider:
    """LLM provider for OpenAI-compatible APIs.

    Uses httpx for HTTP requests (no openai SDK dependency required,
    though the openai package can also be used if available).

    Args:
        model: Model name to use.
        base_url: Base URL for the API.
        api_key: API key for authentication.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        max_tokens: Default cap on generated tokens, sent with every
            request. ``None`` disables the cap and falls back to whatever
            the server defaults to (the legacy behaviour).
    """

    def __init__(
        self,
        model: str = "qwen-turbo",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: str = "",
        temperature: float = 0.7,
        timeout: int = 60,
        max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        max_tokens: Optional[int] = None,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: Conversation history.
            tools: Optional list of ToolSpec objects.
            max_tokens: Per-call cap on generated tokens. ``None`` falls
                back to ``self.max_tokens`` (set from config at construction).
            raise_on_error: When ``False`` (default, backward compatible),
                failures come back as an ``LLMResponse`` with ``content``
                holding the error text and ``error`` set. When ``True``,
                a :class:`~maque_core.llm.errors.LLMError` subclass is raised.
            **kwargs: Additional parameters (temperature, model, etc.).

        Returns:
            LLMResponse with content and/or tool_calls.

        Raises:
            LLMError: Only when ``raise_on_error=True`` — a subclass
                matching the failure (network / timeout / auth / ...).
        """
        url = f"{self.base_url}/chat/completions"

        # Build request body
        body: Dict[str, Any] = {
            "model": kwargs.get("model", self.model),
            "messages": [self._message_to_api(m) for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Resolve max_tokens: explicit argument > kwargs > constructor default.
        effective_max_tokens = self._resolve_max_tokens(
            max_tokens, kwargs.get("max_tokens", self.max_tokens)
        )
        if effective_max_tokens is not None:
            body["max_tokens"] = effective_max_tokens

        # Add tools if provided
        if tools:
            body["tools"] = [self._tool_to_api(t) for t in tools]

        # Headers
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Make request
        if httpx is None:
            error = LLMResponseError(
                "httpx is required for OpenAICompatibleProvider. "
                "Install it with: pip install httpx",
                provider=type(self).__name__,
                model=self.model,
            )
            if raise_on_error:
                raise error
            return LLMResponse(content=f"LLM API error: response: {error.message}",
                               model=self.model, error=error.message)

        try:
            response = httpx.post(
                url,
                json=body,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data)
        except Exception as e:
            error = classify_llm_error(
                e, provider=type(self).__name__, model=self.model
            )
            if raise_on_error:
                raise error from e
            # Keep the error text in `content` so existing callers that
            # display it still work, and flag it via `error` for callers
            # that need to retry or alert.
            return LLMResponse(content=error.message, model=self.model, error=error.message)

    # ---- internal helpers ----

    @staticmethod
    def _resolve_max_tokens(*candidates: Any) -> Optional[int]:
        """Return the first usable max_tokens value, or ``None``.

        Rejects non-positive ints and ``bool`` (``True`` is an ``int``
        subclass in Python and would silently become ``1``).
        """
        for value in candidates:
            if value is None:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value > 0:
                return value
        return None

    @staticmethod
    def _message_to_api(message: Message) -> Dict[str, Any]:
        """Convert a Message to the API's expected format.

        When the message has ``reasoning`` content (assistant messages
        from previous turns in multi-turn conversations), it is sent as
        ``reasoning_content`` — the field used by Qwen3 / DashScope and
        many other OpenAI-compatible reasoning models. This lets the
        model see its own prior reasoning basis in subsequent turns.
        """
        result: Dict[str, Any] = {
            "role": str(message.role),
            "content": message.content,
        }

        if message.reasoning is not None:
            result["reasoning_content"] = message.reasoning

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in message.tool_calls
            ]

        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id

        if message.role.value == "tool" and message.tool_name:
            result["name"] = message.tool_name

        return result

    @staticmethod
    def _tool_to_api(tool_spec: Any) -> Dict[str, Any]:
        """Convert a ToolSpec to the API's tool format."""
        if hasattr(tool_spec, "to_openai_schema"):
            return tool_spec.to_openai_schema()
        # Assume it's already in the right format
        return tool_spec

    @staticmethod
    def _parse_response(data: Dict[str, Any]) -> LLMResponse:
        """Parse the API response into an LLMResponse.

        Reasoning extraction follows a three-level priority:
        1. ``reasoning_content`` field (Qwen3 / DashScope native)
        2. ``reasoning`` field (some other compatible APIs)
        3. Inline ``<think>...</think>`` tags within the content
           (fallback for models that embed thinking in the response text).
           When extracted from <think> tags, the tags are stripped from
           ``content`` and the thinking text goes into ``reasoning``.
        """
        choices = data.get("choices", [])
        if not choices:
            return LLMResponse(model=data.get("model", ""))

        choice = choices[0]
        message_data = choice.get("message", {})

        content = message_data.get("content", "") or ""
        reasoning = None

        # Level 1: reasoning_content field (Qwen3 / DashScope)
        if "reasoning_content" in message_data and message_data["reasoning_content"]:
            reasoning = message_data["reasoning_content"]
        # Level 2: reasoning field (some APIs)
        elif "reasoning" in message_data and message_data["reasoning"]:
            reasoning = message_data["reasoning"]
        # Level 3: inline <think>...</think> tags in content
        else:
            content, extracted = OpenAICompatibleProvider._extract_think_tags(content)
            if extracted:
                reasoning = extracted

        tool_calls_data = message_data.get("tool_calls", [])

        tool_calls: List[ToolCall] = []
        for tc_data in tool_calls_data:
            func = tc_data.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                arguments = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                arguments = {"raw": args_str}

            tool_calls.append(
                ToolCall(
                    id=tc_data.get("id", ""),
                    name=func.get("name", ""),
                    arguments=arguments,
                )
            )

        usage = data.get("usage")

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            model=data.get("model", ""),
            usage=usage,
            finish_reason=choice.get("finish_reason"),
        )

    @staticmethod
    def _extract_think_tags(text: str) -> tuple:
        """Extract <think>...</think> content from text.

        Returns a tuple of (cleaned_content, reasoning_text).
        If no <think> tags are found, returns (text, None).

        Handles:
        - Single pair of <think>...</think> tags
        - Tags at the beginning, middle, or end of text
        - Whitespace normalization around tags
        """
        import re

        # Match <think>...</think> (non-greedy, across newlines)
        pattern = r'<think>(.*?)</think>'
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return text, None

        # Combine all thinking blocks
        reasoning_parts = [m.strip() for m in matches if m.strip()]
        reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None

        # Remove the <think> blocks from content
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
        # Clean up extra whitespace left behind
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        return cleaned, reasoning
