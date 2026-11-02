"""Typed exceptions for the LLM adapter layer.

Why this module exists
----------------------
The adapter used to swallow every failure into ``LLMResponse.content``.
The caller had no way to tell "the model answered something wrong" from
"the network was down", so it could neither retry nor alert.

This module gives failures a type:

``LLMError`` (base)
├── ``LLMNetworkError``     connection refused / reset / DNS / proxy
├── ``LLMTimeoutError``     connect / read / write timeout
├── ``LLMHTTPError``        any non-2xx HTTP status
│   ├── ``LLMAuthError``       401 / 403 — bad or missing key
│   ├── ``LLMRateLimitError``  429 — quota / throttling
│   └── ``LLMServerError``     5xx — upstream failure, usually retryable
└── ``LLMResponseError``    payload could not be decoded / had the wrong shape

Usage::

    try:
        resp = llm.chat(messages, raise_on_error=True)
    except LLMError as err:
        if err.retryable:
            ...   # back off and retry
        else:
            ...   # alert

Every error carries the original exception (``__cause__`` and
``original``) so structured logging keeps the full stack context.
"""

from __future__ import annotations

from typing import Any, Optional

try:  # httpx is optional at import time, same as in openai_compatible
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


class LLMError(Exception):
    """Base class for every error raised by an LLM adapter.

    Attributes:
        message: Human-readable text, also used for ``content`` when the
            adapter is configured to return errors instead of raising.
        category: Machine-readable bucket — one of ``network``, ``timeout``,
            ``auth``, ``rate_limit``, ``server``, ``http``, ``response``,
            ``unknown``.
        provider: Name of the adapter that produced the error.
        model: Model name the adapter was configured with.
        status_code: HTTP status code when the failure came from a response.
        original: The original exception, if any.
        original_type: Class name of the original exception, if any.
    """

    category: str = "unknown"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        original: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.original = original
        self.original_type = type(original).__name__ if original is not None else None

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Structured view for logging / telemetry."""
        return {
            "category": self.category,
            "message": self.message,
            "provider": self.provider,
            "model": self.model,
            "status_code": self.status_code,
            "original_type": self.original_type,
            "retryable": self.retryable,
        }


class LLMNetworkError(LLMError):
    """Connection-level failure — no usable response was received."""

    category = "network"
    retryable = True


class LLMTimeoutError(LLMNetworkError):
    """The request exceeded the configured timeout."""

    category = "timeout"
    retryable = True


class LLMHTTPError(LLMError):
    """A response arrived but with a non-2xx status code."""

    category = "http"


class LLMAuthError(LLMHTTPError):
    """401 / 403 — credentials are missing, invalid or not permitted."""

    category = "auth"
    retryable = False


class LLMRateLimitError(LLMHTTPError):
    """429 — quota exhausted or throttled. Retry after backing off."""

    category = "rate_limit"
    retryable = True


class LLMServerError(LLMHTTPError):
    """5xx — upstream failure; usually worth retrying."""

    category = "server"
    retryable = True


class LLMResponseError(LLMError):
    """The payload could not be decoded, or had an unexpected shape."""

    category = "response"
    retryable = False


# Status codes that mean "your credentials are wrong"
_AUTH_STATUS = frozenset({401, 403})


def _is_httpx_instance(exc: BaseException, *classes: str) -> bool:
    """True if ``exc`` is an instance of one of the named httpx classes."""
    if httpx is None:
        return False
    for name in classes:
        cls = getattr(httpx, name, None)
        if cls is not None and isinstance(exc, cls):
            return True
    return False


def _status_code_of(exc: BaseException) -> Optional[int]:
    """Extract an HTTP status code from an exception, if it carries one."""
    if _is_httpx_instance(exc, "HTTPStatusError"):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
        return None
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status
    return None


def _resolve_class(
    exc: BaseException,
    status_code: Optional[int],
) -> type[LLMError]:
    """Pick the most specific error class for ``exc``."""
    # HTTP status errors are not transport errors — decide by status first.
    if status_code is not None:
        if status_code in _AUTH_STATUS:
            return LLMAuthError
        if status_code == 429:
            return LLMRateLimitError
        if 500 <= status_code <= 599:
            return LLMServerError
        return LLMHTTPError

    # Timeout must be checked before the wider transport/network branch:
    # httpx.TimeoutException is a subclass of httpx.TransportError.
    if isinstance(exc, TimeoutError) or _is_httpx_instance(
        exc, "TimeoutException", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout"
    ):
        return LLMTimeoutError

    if isinstance(exc, (ConnectionError, OSError)) or _is_httpx_instance(
        exc, "NetworkError", "TransportError", "ProxyError"
    ):
        return LLMNetworkError

    # Payload problems: json.JSONDecodeError is a ValueError, and a non-dict
    # body surfaces as AttributeError / TypeError.
    if isinstance(exc, (ValueError, KeyError, TypeError, AttributeError)):
        return LLMResponseError

    return LLMError


def classify_llm_error(
    exc: BaseException,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMError:
    """Wrap an arbitrary exception into the matching :class:`LLMError`.

    The resulting message keeps the adapter's historical
    ``LLM API error: <Type>: <detail>`` shape, with the category inserted
    so the text alone is enough to triage::

        LLM API error: auth: HTTPStatusError: Client error '401 Unauthorized'

    Args:
        exc: The exception caught by the adapter.
        provider: Adapter name, recorded on the error.
        model: Model name, recorded on the error.

    Returns:
        An :class:`LLMError` subclass instance; ``original`` / ``__cause__``
        point back at ``exc``.
    """
    status_code = _status_code_of(exc)
    error_class = _resolve_class(exc, status_code)

    detail = f"{type(exc).__name__}: {exc}"
    message = f"LLM API error: {error_class.category}: {detail}"

    error = error_class(
        message,
        provider=provider,
        model=model,
        status_code=status_code,
        original=exc,
    )
    error.__cause__ = exc
    return error


__all__ = [
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
