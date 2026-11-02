"""Web UI — lightweight HTTP interface for the agent framework.

Provides a simple JSON API for chatting with the agent, plus a
RecordingProvider wrapper that captures every LLM call for trace
display in the frontend.

Error handling:
    LLM failures travel through the exception channel (AgentRunError).
    The API handler catches them and returns a structured error
    response: ``{"ok": false, "error": "...", "hint": "..."}``.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

from maque_core.config import AppConfig
from maque_core.contracts.messages import LLMResponse, Message
from maque_core.factory import create_agent
from maque_core.kernel import Agent, AgentRunError
from maque_core.llm import OpenAICompatibleProvider


class RecordingProvider:
    """Wraps an LLM provider to record every call for trace display.

    Each call's input messages, tools, and output response are saved
    so the web UI can render a step-by-step trace of the agent's
    reasoning process.

    Note: error detection is handled by the exception channel
    (AgentRunError). This class only records — it does not inspect
    ``response.error`` or maintain a ``last_error`` flag.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self.trace: List[dict] = []

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        max_tokens: Optional[int] = None,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        # Record inputs
        step = {
            "messages": [m.to_dict() for m in messages],
            "tools": [t.to_dict() if hasattr(t, "to_dict") else str(t) for t in (tools or [])],
        }

        # Delegate to the inner provider
        response = self._inner.chat(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            raise_on_error=raise_on_error,
            **kwargs,
        )

        # Record output
        step["response"] = response.to_dict()
        self.trace.append(step)

        return response

    def reset_trace(self) -> None:
        """Clear the recorded trace."""
        self.trace = []


def build_agent(config: AppConfig) -> Agent:
    """Build an Agent instance with a RecordingProvider for tracing."""
    inner_llm = OpenAICompatibleProvider(
        model=config.llm.model,
        base_url=config.llm.base_url,
        api_key=config.llm.api_key,
        temperature=config.llm.temperature,
        timeout=config.llm.timeout,
        max_tokens=config.llm.max_tokens,
    )
    llm = RecordingProvider(inner_llm)
    return create_agent(
        llm=llm,
        system_prompt=config.agent.system_prompt,
        max_iterations=config.agent.max_iterations,
        max_tool_calls=config.agent.max_tool_calls,
        max_history_turns=config.agent.max_history_turns,
        include_reasoning_in_history=config.agent.include_reasoning_in_history,
        tools_package=config.tools_package,
    )


def chat_endpoint(agent: Agent, question: str) -> dict:
    """Handle a chat request and return a JSON-serializable response.

    On success: ``{"ok": true, "answer": "...", "trace": [...]}``
    On error:   ``{"ok": false, "error": "...", "hint": "..."}``
    """
    provider = getattr(agent.llm, "reset_trace", None)
    if callable(provider):
        agent.llm.reset_trace()

    try:
        answer = agent.ask(question)
    except AgentRunError as e:
        trace = getattr(agent.llm, "trace", [])
        return {
            "ok": False,
            "error": e.message,
            "hint": (
                "The LLM service returned an error. Check your API key, "
                "network connection, and model availability."
            ),
            "trace": trace,
        }

    trace = getattr(agent.llm, "trace", [])
    return {
        "ok": True,
        "answer": answer,
        "trace": trace,
    }


# ---- Simple WSGI-compatible entry point ----

def create_app(config: Optional[AppConfig] = None) -> dict:
    """Create the web UI application state.

    Returns a dict with the agent instance and a ``handle`` function
    that processes chat requests. This is a minimal setup — in
    production you'd wrap this in Flask / FastAPI / etc.
    """
    if config is None:
        config = AppConfig.from_env()
    agent = build_agent(config)

    def handle(body: str) -> str:
        try:
            data = json.loads(body)
            question = data.get("question", "")
        except (json.JSONDecodeError, AttributeError):
            return json.dumps({
                "ok": False,
                "error": "Invalid request body",
                "hint": "Send a JSON object with a 'question' field.",
            })

        if not question:
            return json.dumps({
                "ok": False,
                "error": "Question is required",
                "hint": "Provide a non-empty 'question' string.",
            })

        result = chat_endpoint(agent, question)
        return json.dumps(result, ensure_ascii=False)

    return {
        "agent": agent,
        "handle": handle,
    }
