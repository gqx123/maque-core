"""LLM adapters — implementations of the LLMProvider protocol.

Supports:
- OpenAICompatibleProvider: Works with any OpenAI-compatible API
  (OpenAI, DashScope/Qwen, DeepSeek, etc.)
- MockLLM: Pre-programmed responses for tests and offline demos

Failures are typed (see :mod:`maque_core.llm.errors`) so callers can
retry or alert instead of mistaking an error for an answer.
"""

from .openai_compatible import DEFAULT_MAX_TOKENS, OpenAICompatibleProvider
from .mock import MockLLM
from .errors import (
    LLMError,
    LLMNetworkError,
    LLMTimeoutError,
    LLMHTTPError,
    LLMAuthError,
    LLMRateLimitError,
    LLMServerError,
    LLMResponseError,
    classify_llm_error,
)

__all__ = [
    "OpenAICompatibleProvider",
    "MockLLM",
    "DEFAULT_MAX_TOKENS",
    "LLMError",
    "LLMNetworkError",
    "LLMTimeoutError",
    "LLMHTTPError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMServerError",
    "LLMResponseError",
    "classify_llm_error",
]
