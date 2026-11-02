"""Configuration for the agent framework.

All settings can be overridden via environment variables with the
``MAQUE_`` prefix, e.g. ``MAQUE_LLM_MODEL=qwen3-turbo``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM backend configuration."""

    model: str = "qwen-turbo"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    temperature: float = 0.7
    timeout: int = 60
    max_tokens: int = 2048

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            model=os.environ.get("MAQUE_LLM_MODEL", cls.model),
            base_url=os.environ.get("MAQUE_LLM_BASE_URL", cls.base_url),
            api_key=os.environ.get("MAQUE_LLM_API_KEY", cls.api_key),
            temperature=float(os.environ.get("MAQUE_LLM_TEMPERATURE", cls.temperature)),
            timeout=int(os.environ.get("MAQUE_LLM_TIMEOUT", cls.timeout)),
            max_tokens=int(os.environ.get("MAQUE_LLM_MAX_TOKENS", cls.max_tokens)),
        )


@dataclass
class AgentConfig:
    """Agent runtime configuration."""

    system_prompt: str = "You are a helpful assistant with access to tools. Use tools when needed to answer questions accurately."
    max_iterations: int = 10
    max_tool_calls: Optional[int] = None
    verbose: bool = False
    max_history_turns: int = 20
    include_reasoning_in_history: bool = True

    @classmethod
    def from_env(cls) -> "AgentConfig":
        max_tool_calls_str = os.environ.get("MAQUE_MAX_TOOL_CALLS", "")
        max_tool_calls = int(max_tool_calls_str) if max_tool_calls_str else None
        max_history_str = os.environ.get("MAQUE_MAX_HISTORY_TURNS", "")
        max_history_turns = int(max_history_str) if max_history_str else cls.max_history_turns
        return cls(
            system_prompt=os.environ.get("MAQUE_SYSTEM_PROMPT", cls.system_prompt),
            max_iterations=int(os.environ.get("MAQUE_MAX_ITERATIONS", cls.max_iterations)),
            max_tool_calls=max_tool_calls,
            verbose=os.environ.get("MAQUE_VERBOSE", "").lower() in ("1", "true", "yes"),
            max_history_turns=max_history_turns,
            include_reasoning_in_history=os.environ.get(
                "MAQUE_INCLUDE_REASONING_IN_HISTORY", "true"
            ).lower() in ("1", "true", "yes"),
        )


@dataclass
class AppConfig:
    """Top-level application configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools_package: str = "maque_core.tools.impl"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            llm=LLMConfig.from_env(),
            agent=AgentConfig.from_env(),
            tools_package=os.environ.get("MAQUE_TOOLS_PACKAGE", cls.tools_package),
        )
