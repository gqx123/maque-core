"""maque_core — a lightweight, extensible Agent framework."""

from .contracts.llm import LLMProvider
from .contracts.messages import LLMResponse, Message, Role, ToolCall
from .contracts.tool import ToolSpec
from .factory import create_agent
from .kernel import Agent, AgentRunError
from .llm import MockLLM, OpenAICompatibleProvider
from .tools import ToolRegistry, tool

__version__ = "1.3.4"

__all__ = [
    "create_agent",
    "Agent",
    "AgentRunError",
    "tool",
    "LLMProvider",
    "Message",
    "Role",
    "ToolCall",
    "LLMResponse",
    "ToolSpec",
    "ToolRegistry",
    "OpenAICompatibleProvider",
    "MockLLM",
    "__version__",
]
